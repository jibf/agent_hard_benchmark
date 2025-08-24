#!/usr/bin/env python3
"""
Download .parquet shards from every Hugging Face dataset repo under an owner
and merge them into ONE file per dataset named <dataset_name>.parquet.

- Works with older/newer pyarrow (uses ParquetFile.iter_batches, no Dataset.scan()).
- Prefers files under data/ but can fall back to any .parquet.
- Cleans up all temporary directories (no _tmp_* left behind).
- Skips existing outputs unless --overwrite is supplied.

Requirements:
  pip install --upgrade huggingface_hub pyarrow

Auth for private datasets:
  export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, List

from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--owner", required=True, help="HF username/org")
    p.add_argument("--out", default="hf_parquets", help="Output directory for merged files")
    p.add_argument("--max", type=int, default=10000, help="Max repos to list")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing merged files")
    p.add_argument(
        "--only-data-folder",
        action="store_true",
        help="Only consider parquet files under data/ (default: prefer data/ then fallback to any)",
    )
    return p.parse_args()


def as_list(x: Iterable) -> List:
    try:
        return list(x)
    except TypeError:
        return [*x]


def merge_parquet_files(input_paths: List[Path], output_path: Path):
    """
    Merge shards into a single parquet by streaming record batches from each shard.
    Single-shard datasets are just copied/renamed.
    """
    input_paths = sorted(input_paths, key=lambda p: p.name)

    # Fast path for single shard
    if len(input_paths) == 1:
        tmp = output_path.with_suffix(output_path.suffix + ".tmp")
        tmp.write_bytes(input_paths[0].read_bytes())
        tmp.replace(output_path)
        return

    # Multi-shard streaming merge
    first_pf = pq.ParquetFile(str(input_paths[0]))
    schema = first_pf.schema_arrow
    tmp_out = output_path.with_suffix(output_path.suffix + ".tmp")

    with pq.ParquetWriter(str(tmp_out), schema) as writer:
        # first file
        for batch in first_pf.iter_batches():
            writer.write_batch(batch)

        # remaining files
        for shard in input_paths[1:]:
            pf = pq.ParquetFile(str(shard))
            for batch in pf.iter_batches():
                if batch.schema != schema:
                    # Cast to base schema if needed (unsafe cast allows common promotions)
                    batch = batch.cast(schema, safe=False)
                writer.write_batch(batch)

    tmp_out.replace(output_path)


def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    api = HfApi()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = as_list(api.list_datasets(author=args.owner, limit=args.max))
    print(f"Found {len(datasets)} dataset repos for owner {args.owner}")

    for ds_info in datasets:
        repo_id = getattr(ds_info, "id", None) or getattr(ds_info, "repo_id", None) or str(ds_info)
        ds_name = repo_id.split("/")[-1]
        target_path = out_dir / f"{ds_name}.parquet"

        if target_path.exists() and not args.overwrite:
            print(f"✓ {repo_id}: exists → {target_path.name} (use --overwrite to rebuild)")
            continue

        print(f"\n→ Processing {repo_id}")
        try:
            files = as_list(api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token))
        except Exception as e:
            print(f"  ! list_repo_files failed: {e}")
            continue

        parquets_all = [f for f in files if f.endswith(".parquet")]
        parquets_data = [f for f in parquets_all if f.startswith("data/")]
        candidates = parquets_data if (args.only_data_folder or parquets_data) else parquets_all

        if not candidates:
            print("  ! No parquet files found, skipping.")
            continue

        # Download shards to a temp dir
        tmp_dir = out_dir / f"_tmp_{ds_name}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        local_paths: List[Path] = []

        try:
            for rel in sorted(candidates):
                local_file = hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=rel,
                    local_dir=str(tmp_dir),
                    local_dir_use_symlinks=False,
                    token=token,
                )
                local_paths.append(Path(local_file))

            print(f"  Merging {len(local_paths)} shard(s) → {target_path.name}")
            merge_parquet_files(local_paths, target_path)
            print(f"  ✓ Saved {target_path}")

        except Exception as e:
            print(f"  ! Failed while downloading/merging: {e}")

        finally:
            # Always remove temp directory recursively (no leftovers)
            try:
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
            except Exception as e:
                print(f"  ! Cleanup failed for {tmp_dir}: {e}")


if __name__ == "__main__":
    main()
