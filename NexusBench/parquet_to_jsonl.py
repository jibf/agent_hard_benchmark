#!/usr/bin/env python3
"""
Convert NexusBench parquet intermediate result files to the unified *.jsonl format
expected by sglang evaluation.  One output JSONL file is produced **per model**.

Example mapping of input → output path (under --output-dir):
    /path/.../ABC_08182318_openai-gpt-4.1-OpenAIFC.parquet
        → openai/gpt-4.1.jsonl
    /path/.../CVECPEBenchmark_08182347_deepseek-ai-DeepSeek-V3-0324-OpenAIFC.parquet
        → deepseek_ai/DeepSeek-V3-0324.jsonl

Usage
-----
    python parquet_to_jsonl.py \
        --parquet-dir /path/to/NexusBench/parquets \
        --output-dir  /path/to/jsonl_out

Notes
-----
* All parquet files inside --parquet-dir (recursively) that end with ``.parquet``
  are processed.
* For each *row* in a parquet file we emit *one* JSON object.
* The script attempts to map the generic NexusBench column pattern to the
  sglang message trajectory automatically (prompt_i/query_i/... columns).
  Feel free to adapt the mapping logic if needed for newly–added benchmarks.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Helpers for filename → (benchmark, model_path)
# ---------------------------------------------------------------------------

# Matches e.g. "_08190020_openai-o3-high-OpenAIFC.parquet"
# Captures model segment between the digits and the ending "-OpenAIFC.parquet"
MODEL_PART_RE = re.compile(r"_(?:\d{6,})_(.*?)-OpenAIFC\.parquet$")


def extract_benchmark_and_model(filename: str) -> Tuple[str, str]:
    """Return (benchmark_name, model_path) from an input parquet filename."""
    basename = Path(filename).name
    # benchmark name = everything before first underscore
    benchmark = basename.split("_")[0]

    m = MODEL_PART_RE.search(basename)
    if not m:
        raise ValueError(f"Cannot parse model part from filename: {basename}")

    model_part = m.group(1)  # e.g. "openai-gpt-4.1" or "deepseek-ai-DeepSeek-V3-0324"
    tokens = model_part.split("-")
    if len(tokens) >= 2 and tokens[1] == "ai":
        # special pattern like "deepseek-ai-DeepSeek-R1-0528" → owner "deepseek_ai"
        owner = f"{tokens[0]}_{tokens[1]}"
        model_name = "-".join(tokens[2:]) if len(tokens) > 2 else "unknown"
    else:
        owner = tokens[0].replace("-", "_")
        model_name = "-".join(tokens[1:]) if len(tokens) > 1 else "unknown"
    return benchmark, f"{owner}/{model_name}"


# ---------------------------------------------------------------------------
# Column utilities
# ---------------------------------------------------------------------------

PROMPT_RE = re.compile(r"^(prompt|query)_(\d+)$")
RESPONSE_RE = re.compile(r"^response_(\d+)$")
EXEC_RE = re.compile(r"^executed_(\d+)$")
RESULT_RE = re.compile(r"^result_(\d+)$")


class RowConverter:
    """Convert a single record (dict) from parquet → sglang JSON object."""

    def __init__(
        self,
        row: Dict[str, object],
        model_path: str,
        benchmark_name: str = "nexusbench",
        default_task: str | None = None,
    ):
        self.row = row
        self.benchmark = benchmark_name
        self.model_path = model_path
        # Attempt to get task/subtask name.  Some benchmarks store it in GT.
        self.task_name: str | None = self._infer_task_name() or default_task

    # ---------------------------------------------------------------------
    def _infer_task_name(self) -> str | None:
        # Heuristic: look for a field that clearly encodes a task identifier.
        VALID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")
        for key in ("task", "subtask", "id"):
            val = self.row.get(key)
            if isinstance(val, str) and VALID_RE.match(val):
                return val.split("-")[0]
        return None

    # ---------------------------------------------------------------------
    def build_messages(self) -> List[Dict[str, object]]:
        messages: List[Dict[str, object]] = []
        # Gather all turn indices present in row
        turn_indices = set()
        for col in self.row:
            for regex in (PROMPT_RE, RESPONSE_RE):
                m = regex.match(col)
                if m:
                    # capture group holding the numeric index is ALWAYS the last one
                    idx_str = m.group(m.lastindex)
                    if idx_str is not None:
                        turn_indices.add(int(idx_str))
        if not turn_indices:
            return messages

        for idx in sorted(turn_indices):
            # user message
            user_content = self.row.get(f"prompt_{idx}") or self.row.get(f"query_{idx}")
            if user_content is not None:
                messages.append({"role": "user", "content": user_content})
            # assistant message
            assistant_content = self.row.get(f"response_{idx}")
            assistant_msg: Dict[str, object] = {
                "role": "assistant",
                "content": assistant_content,
                "turn_idx": idx + 1,
            }
            # If any tool/execution metadata exists, attach it
            exec_data = self.row.get(f"executed_{idx}")
            result_data = self.row.get(f"result_{idx}")
            tool_calls = []
            if exec_data is not None:
                tool_calls.append({"executed": exec_data})
            if result_data is not None:
                tool_calls.append({"result": result_data})
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)
        return messages

    # ---------------------------------------------------------------------
    def build_eval_result(self) -> Dict[str, object]:
        # Use "Final Accuracy" or "score" if present.
        score_value = self.row.get("Final Accuracy") or self.row.get("score") or 0

        # Normalise different representations to a float in [0,1].
        if isinstance(score_value, (int, float)):
            score = float(score_value)
        elif isinstance(score_value, str):
            val = score_value.strip().lower()
            if val in {"true", "correct", "yes"}:
                score = 1.0
            elif val in {"false", "incorrect", "no"}:
                score = 0.0
            else:
                # Remove % if present
                if val.endswith("%"):
                    val = val.rstrip("%")
                    try:
                        score = float(val) / 100.0
                    except ValueError:
                        score = 0.0
                else:
                    try:
                        score = float(val)
                    except ValueError:
                        score = 0.0
        else:
            # Unknown type → default to 0
            score = 0.0

        # Clamp to [0,1]
        if score > 1:
            score /= 100.0
        score = max(0.0, min(1.0, score))
        return {"score": score}

    # ---------------------------------------------------------------------
    def build_meta(self) -> Dict[str, object]:
        # Dump remaining key/values that are JSON-serialisable.
        meta: Dict[str, object] = {}
        for k, v in self.row.items():
            if k.startswith(("prompt_", "query_", "response_", "executed_", "result_")):
                continue
            if k in ("Final Accuracy", "score"):
                continue
            if isinstance(v, (str, int, float, bool, type(None))):
                meta[k] = v
        return meta

    # ---------------------------------------------------------------------
    def to_json_obj(self) -> Dict[str, object]:
        obj: Dict[str, object] = {
            "model_path": self.model_path,
            "benchmark_name": self.benchmark,
            "sampling_params": {"max_tokens": 16384, "temperature": 0.0},
            "messages": self.build_messages(),
            "eval_result": self.build_eval_result(),
        }
        if self.task_name:
            obj["task_name"] = self.task_name
        meta = self.build_meta()
        if meta:
            obj["meta"] = meta
        return obj


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def process_parquet(
    path: Path,
    out_root: Path,
    overwrite: bool = False,
    opened_once: set[str] | None = None,
):
    bench_token, model_path = extract_benchmark_and_model(path.name)
    out_path = out_root / model_path  # e.g. deepseek_ai/DeepSeek-V3-0324
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = out_path.with_suffix(".jsonl")

    # Determine file mode: truncate only the very first time we write to a model file
    first_time = model_path not in opened_once if opened_once is not None else not out_file.exists()
    mode = "w" if (overwrite and first_time) or (not out_file.exists()) else "a"
    with out_file.open(mode, encoding="utf-8") as fout:
        pf = pq.ParquetFile(str(path))
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx)
            batch_dict = table.to_pydict()
            num_rows = len(next(iter(batch_dict.values())))
            for i in range(num_rows):
                row = {k: v[i] for k, v in batch_dict.items()}
                obj = RowConverter(row, model_path, default_task=bench_token).to_json_obj()
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if opened_once is not None:
        opened_once.add(model_path)
    print(f"✓ {path.name} → {out_file} (+{pf.metadata.num_rows} lines)")


# ---------------------------------------------------------------------------

def discover_parquet_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.parquet") if p.is_file())


# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Convert NexusBench parquet files to sglang-style JSONL.")
    ap.add_argument("--parquet-dir", required=True, type=Path, help="Directory containing *.parquet files")
    ap.add_argument("--output-dir", required=True, type=Path, help="Directory to write JSONL files")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing JSONL files")
    return ap.parse_args()


def main():
    args = parse_args()
    parquet_dir: Path = args.parquet_dir.expanduser().resolve()
    output_dir: Path = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_parquet_files(parquet_dir)
    if not files:
        print(f"No parquet files found under {parquet_dir}")
        sys.exit(1)

    opened_once: set[str] = set()
    for path in files:
        try:
            process_parquet(path, output_dir, overwrite=args.overwrite, opened_once=opened_once)
        except Exception as e:
            print(f"! Failed to process {path}: {e}")


if __name__ == "__main__":
    main()
