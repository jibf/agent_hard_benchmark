#!/usr/bin/env python3
"""
convert_outputs.py
------------------
A light-weight CLI utility for converting the raw agent-benchmark output that
lives under the local `data/` directory into the SGLang-compatible jsonl format
outlined in the prompt.

Usage
-----
$ python convert_outputs.py \
    --data-dir /absolute/path/to/data \
    --out-dir  /absolute/path/for/jsonl

The script crawls every
    <data>/<benchmark>/<agent>_user_<provider>/<run_timestamp>/trajectories/<scenario>/conversation.json
file and emits one json line per scenario to
    <out-dir>/<agent>/<scenario>.jsonl
creating directories as required.  If an output file already exists the new
line will be appended.

The **agent** (model under evaluation) is the part of the directory name before
`_user_`, e.g. `claude-4-sonnet-thinking-off`.
The **user model** is fixed to GPT-4o – the version (e.g. `20240806`) is
inferred from the run directory (`gpt-4o-20240806_…`).

The script tries to keep as much information as is available:
* the full `conversation.json` array is copied into the `messages` field
* the matching scenario entry from `result_summary.json` (if present) becomes
  `eval_result`; the numeric `similarity` is mapped to `score`
* sensible defaults are supplied when optional data is absent

The produced json strictly follows the schema given in the prompt so it can be
consumed directly by SGLang.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_SAMPLING_PARAMS = {"max_tokens": 16_384, "temperature": 0.0}
USER_MODEL_PROVIDER = "openai"
USER_MODEL_NAME = "gpt-4o"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _derive_paths(conv_path: Path):
    """Return benchmark, agent, run_dir, scenario names parsed from a conversation path."""
    # Expected layout: data/<benchmark>/<agent_dir>/<run_dir>/trajectories/<scenario>/conversation.json
    parts = conv_path.parts
    # Use *last* occurrence of the "data" directory so we ignore any leading
    # directories that coincidentally contain that name (e.g. /eic/data).
    try:
        data_idx = max(i for i, p in enumerate(parts) if p == "data")
    except ValueError:
        raise ValueError(f"Path {conv_path} does not contain a 'data' folder component")

    # Find the index of the obligatory "trajectories" segment to be resilient
    # to agents that introduce extra nesting levels (e.g. vendor / model-name).
    try:
        traj_idx = parts.index("trajectories", data_idx)
    except ValueError:
        raise ValueError(f"Path {conv_path} does not contain a 'trajectories' directory component")

    if traj_idx < data_idx + 4:
        raise ValueError(
            f"Unexpected directory structure for {conv_path}: not enough components before 'trajectories'"
        )

    benchmark = parts[data_idx + 1]
    agent_dir = parts[traj_idx - 2]  # component two levels above 'trajectories'
    run_dir = parts[traj_idx - 1]    # component just above 'trajectories'
    scenario = parts[traj_idx + 1]   # component inside 'trajectories'
    return benchmark, agent_dir, run_dir, scenario


def _agent_name_from_dir(agent_dir: str) -> str:
    """Strip trailing `_user_…` if present to get the agent model name."""
    return agent_dir.split("_user")[0]


def _user_model_path(run_dir: str) -> str:
    """Extract `openai/gpt-4o-YYYYMMDD` from run directory name like
    `gpt-4o-20240806_08_15_2025_03_02_49`."""
    # The date coded substring precedes the first underscore after model name.
    first_segment = run_dir.split("_")[0]  # -> 'gpt-4o-20240806'
    return f"{USER_MODEL_PROVIDER}/{first_segment}"


def _load_json(path: Path) -> Any:
    with path.open() as fp:
        return json.load(fp)


def _load_result_summary(run_root: Path) -> Dict[str, Any]:
    """Return a mapping of scenario name -> result dict."""
    summary_path = run_root / "result_summary.json"
    if not summary_path.exists():
        return {}
    data = _load_json(summary_path)
    mapping: Dict[str, Any] = {}
    for entry in data.get("per_scenario_results", []):
        # Copy the whole entry but promote `similarity` to `score`
        clean = dict(entry)
        clean["score"] = clean.get("similarity")
        mapping[entry.get("name")] = clean
    return mapping


def _build_jsonl_obj(
    model_path: str,
    user_model_path: str,
    benchmark: str,
    task_name: str,
    messages: List[Dict[str, Any]],
    eval_res: Optional[Dict[str, Any]] = None,
    run_timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    obj: Dict[str, Any] = {
        "model_path": model_path,
        "user_model_path": user_model_path,
        "benchmark_name": benchmark,
        "task_name": task_name,
        "sampling_params": DEFAULT_SAMPLING_PARAMS.copy(),
        "messages": messages,
        "eval_result": {
            "score": (eval_res.get("score") if eval_res else None) or 0.0,
        },
    }
    # Extend eval_result with additional fields if present
    if eval_res:
        extra = {k: v for k, v in eval_res.items() if k not in {"name", "score"}}
        obj["eval_result"].update(extra)

    # Optional meta
    meta: Dict[str, Any] = {}
    if run_timestamp:
        meta["run_timestamp"] = run_timestamp
    if meta:
        obj["meta"] = meta
    return obj

# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def convert(data_dir: Path, out_dir: Path) -> None:
    # Cache result summaries per run directory to avoid re-parsing.
    summary_cache: Dict[Path, Dict[str, Any]] = {}

    conv_files = list(data_dir.glob("**/trajectories/*/conversation.json"))
    if not conv_files:
        print(f"No conversation.json files found under {data_dir}.", file=sys.stderr)
        return

    for conv_path in conv_files:
        benchmark, agent_dir, run_dir, scenario = _derive_paths(conv_path)
        run_root = conv_path.parents[2]  # …/<run_dir>

        # Load conversation
        messages = _load_json(conv_path)

        # Ensure assistant messages include turn_idx (1-based within the conversation)
        assistant_counter = 0
        for msg in messages:
            if msg.get("role") == "assistant":
                assistant_counter += 1
                msg.setdefault("turn_idx", assistant_counter)

        # Load result summary (cached)
        if run_root not in summary_cache:
            summary_cache[run_root] = _load_result_summary(run_root)
        scenario_results = summary_cache[run_root].get(scenario)

        # If no direct match, try a fuzzy fallback: many benchmarks append
        # or drop suffixes like "_all_tools".  Pick the first entry whose
        # key contains the scenario name or vice-versa.
        if scenario_results is None:
            for k, v in summary_cache[run_root].items():
                if scenario in k or k in scenario:
                    scenario_results = v
                    break

        # Build model_path: provider/optional_org/.../model
        parts = list(conv_path.parts)
        # Find positions again
        try:
            data_idx = max(i for i, p in enumerate(parts) if p == "data")
            traj_idx = parts.index("trajectories", data_idx)
        except ValueError:
            continue  # should not happen

        provider_dir = parts[data_idx + 1]  # e.g. 'agent_togetherai'
        provider = provider_dir.replace("agent_", "")

        org_model_parts = list(parts[data_idx + 2 : traj_idx - 1])
        if org_model_parts:
            org_model_parts[-1] = org_model_parts[-1].split("_user")[0]

        model_path_str = "/".join([provider] + org_model_parts) if org_model_parts else provider

        user_model_path = _user_model_path(run_dir)
        run_timestamp = None
        # Try to parse timestamp from the remainder of run_dir after first '_'
        if "_" in run_dir:
            ts_part = run_dir.split("_", 1)[1]
            try:
                # Expect format MM_DD_YYYY_HH_MM_SS
                dt = datetime.strptime(ts_part, "%m_%d_%Y_%H_%M_%S")
                run_timestamp = dt.isoformat() + "Z"
            except ValueError:
                pass  # silently ignore if format unexpected

        obj = _build_jsonl_obj(
            model_path=model_path_str,
            user_model_path=user_model_path,
            benchmark="toolsandbox",
            task_name=scenario,
            messages=messages,
            eval_res=scenario_results,
            run_timestamp=run_timestamp,
        )

        # ----------------------------------------------------------------
        # Write (append) to the appropriate jsonl file
        #   <out-dir>/<agent>/<scenario>.jsonl
        # ----------------------------------------------------------------
        # Save ONE file per model (llm). Keep provider/org directory tree, but
        # filename is just the model (last component) plus .jsonl
        if org_model_parts:
            *org_parts, model_name = org_model_parts
        else:
            org_parts = []
            model_name = provider  # degenerate case

        model_dir = out_dir.joinpath(provider, *org_parts)
        model_dir.mkdir(parents=True, exist_ok=True)
        out_path = model_dir / f"{model_name}.jsonl"

        with out_path.open("a", encoding="utf-8") as fp:
            json.dump(obj, fp, ensure_ascii=False)
            fp.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert raw benchmark outputs to SGLang-compatible jsonl format.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Root directory that contains the raw outputs (default: ./data)")
    parser.add_argument("--out-dir", type=Path, default=Path("converted"), help="Destination directory for the produced jsonl files (default: ./converted)")

    args = parser.parse_args()

    convert(args.data_dir.resolve(), args.out_dir.resolve())


if __name__ == "__main__":
    main()
