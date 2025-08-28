#!/usr/bin/env python3
"""
download_agenthard.py

Download multiple AgentHard evaluation datasets from Hugging Face in one go,
optionally filtering which files to pull (e.g., only *.jsonl).

Usage:
  python3 download_agenthard.py
  python3 download_agenthard.py --patterns "*.jsonl"
  python3 download_agenthard.py --repo AgentHard/NexusBench-evaluation --patterns "*.jsonl"

Note: All AgentHard datasets are public and can be downloaded without authentication.
If you encounter issues with other gated datasets, run:
  huggingface-cli login
and paste your HF token first.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

try:
    from huggingface_hub import snapshot_download
    import huggingface_hub
except Exception as e:
    print("ERROR: huggingface_hub is not installed. Run: pip install -U huggingface_hub")
    raise

DEFAULT_DATASETS = [
    "AgentHard/NexusBench-evaluation",
    "AgentHard/DrafterBench-evaluation",
    "AgentHard/complex-func-bench-evaluation",
    "AgentHard/multi_challenge-evaluation",
    "AgentHard/BFCL-evaluation",
    "AgentHard/ToolSandbox-evaluation",
]

def download_dataset(repo_id: str, allow_patterns: Optional[List[str]] = None, base_dir: Path = Path(".")) -> Path:
    """
    Download a single HF dataset repo into a local folder named after the repo_id suffix.
    Returns the local directory path.
    """
    local_dir = base_dir / repo_id.split("/")[-1]
    print(f"\n=== Downloading {repo_id} → {local_dir} ===")
    if allow_patterns:
        print(f"    allow_patterns={allow_patterns}")

    # snapshot_download will create the directory and populate it.
    # local_dir_use_symlinks=False ensures you get actual files, not symlinks.
    cache_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,  # e.g., ["*.jsonl"]
        # You can tune max_workers if you want: max_workers=8,
        # resume_download=True is default behavior.
    )

    print(f"    Done. Cached at: {cache_dir}")
    return local_dir

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download AgentHard HF datasets.")
    p.add_argument(
        "--repo",
        action="append",
        help="HF dataset repo_id (e.g., AgentHard/NexusBench-evaluation). Repeat to add multiple. "
             "If omitted, defaults to the six AgentHard evaluation repos.",
    )
    p.add_argument(
        "--patterns",
        nargs="+",
        help='Optional list of glob patterns to limit which files to download, e.g.: --patterns "*.jsonl" "scores/*.csv"',
    )
    p.add_argument(
        "--out",
        default=".",
        help="Base output directory (default: current directory).",
    )
    return p.parse_args()

def main():
    args = parse_args()
    repos = args.repo if args.repo else DEFAULT_DATASETS
    patterns = args.patterns
    base_dir = Path(args.out).resolve()

    print(f"Hugging Face Hub version: {huggingface_hub.__version__}")
    print(f"Output directory: {base_dir}")
    base_dir.mkdir(parents=True, exist_ok=True)

    errors = []
    for repo_id in repos:
        try:
            download_dataset(repo_id, allow_patterns=patterns, base_dir=base_dir)
        except Exception as e:
            errors.append((repo_id, str(e)))
            print(f"!! ERROR downloading {repo_id}: {e}", file=sys.stderr)

    if errors:
        print("\nSome downloads failed:")
        for repo_id, msg in errors:
            print(f" - {repo_id}: {msg}")
        sys.exit(1)

    print("\nAll requested datasets downloaded successfully.")

if __name__ == "__main__":
    main()
