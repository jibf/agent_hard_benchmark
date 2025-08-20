#!/usr/bin/env python3
"""
create_scores_matrix.py
-----------------------
Scan a directory produced by `convert_outputs.py` (default ./converted) and
assemble a CSV matrix of evaluation scores:
    rows   = unique task_name (subtask) across all models
    columns= unique model_path (provider/org/.../model)
    cell   = eval_result.score (empty if missing)

Usage
-----
$ python create_scores_matrix.py \
         --in-dir  converted \
         --out-csv scores_matrix.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, DefaultDict, Set


def collect_scores(root: Path):
    """Traverse *root* and collect scores into nested dict[task][model] = score"""
    task_to_scores: DefaultDict[str, Dict[str, float]] = DefaultDict(dict)
    models_seen: Set[str] = set()

    for jsonl_path in root.rglob("*.jsonl"):
        with jsonl_path.open(encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip malformed lines

                task = obj.get("task_name") or obj.get("benchmark_subtask") or ""
                model = obj.get("model_path", "")
                score = obj.get("eval_result", {}).get("score")
                if task and model and score is not None:
                    task_to_scores[task][model] = score
                    models_seen.add(model)
    return task_to_scores, sorted(models_seen)


def write_csv(task_to_scores: Dict[str, Dict[str, float]], models: list[str], out_csv: Path):
    tasks = sorted(task_to_scores)
    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["task_name"] + models)
        for task in tasks:
            row = [task]
            scores_map = task_to_scores[task]
            for m in models:
                val = scores_map.get(m)
                row.append("" if val is None else val)
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Create CSV matrix of scores from converted jsonl files.")
    parser.add_argument("--in-dir", type=Path, default=Path("converted"), help="Directory containing converted jsonl files")
    parser.add_argument("--out-csv", type=Path, default=Path("scores_matrix.csv"), help="Destination CSV path")
    args = parser.parse_args()

    task_to_scores, models = collect_scores(args.in_dir.resolve())
    write_csv(task_to_scores, models, args.out_csv.resolve())
    print(f"Wrote {len(task_to_scores)} tasks × {len(models)} models to {args.out_csv}")


if __name__ == "__main__":
    main()
