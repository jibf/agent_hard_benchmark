import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import polars as pl
from attrs import asdict
from types import FunctionType
import networkx as nx  # type: ignore

# We purposefully do not perform any filtering, processing or value munging.  The
# goal is to expose the benchmark definitions "as-is" in a JSON file that can be
# consumed by downstream tooling for analysis, visualisation or further data
# crunching.

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _serialize_df(df: pl.DataFrame) -> List[Dict[str, Any]]:  # noqa: D401 – simple function name.
    """Return the dataframe as a list-of-dict rows so it is JSON serialisable."""
    return df.to_dicts()


def _json_default(obj):  # noqa: ANN001 – generic serializer.
    """Fallback converter for otherwise non-serialisable objects.

    We keep the representation lossless where feasible.  For enums we output the
    symbolic name (e.g. ``"USER"``).  `pathlib.Path` objects are converted to
    their string representation.
    """
    # polars objects
    if isinstance(obj, pl.DataFrame):
        return _serialize_df(obj)
    if isinstance(obj, pl.Series):
        return obj.to_list()

    # Enums
    try:
        # Most enums derive from `Enum` including StrEnum.  We treat all of them
        # the same.
        from enum import Enum

        if isinstance(obj, Enum):
            return obj.name
    except Exception:  # pragma: no cover – defensive guard, should not happen.
        pass

    # ExecutionContext – serialize via its own helper to capture full starting state.
    try:
        from tool_sandbox.common.execution_context import ExecutionContext  # noqa: WPS433 – runtime import

        if isinstance(obj, ExecutionContext):
            # Exclude interactive console bytes for readability / size.
            return obj.to_dict(serialize_console=False)
    except Exception:  # pragma: no cover – import errors or other issues
        pass

    # Callables/functions – use a deterministic dotted path.
    if isinstance(obj, FunctionType):
        return f"{obj.__module__}.{obj.__qualname__}"

    # networkx graph – convert to simple edge/node lists for transparency.
    if isinstance(obj, nx.Graph):  # includes DiGraph subclasses
        return {
            "type": obj.__class__.__name__,
            "nodes": list(obj.nodes),
            "edges": list(obj.edges),
        }

    # pathlib
    if isinstance(obj, Path):
        return str(obj)

    # Fallback to stringification – this guarantees we never fail with "Object of
    # type … is not JSON serialisable".  The resulting value is still the raw
    # *data* albeit without structure.  Crucially we *do not* mutate or drop
    # anything.
    return str(obj)


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------

def build_prompts_groundtruths() -> List[Dict[str, Any]]:
    """Load *all* ToolSandbox benchmark scenarios and return a serialisable list."""
    # Runtime import avoids the reasonably heavy module import when one merely
    # runs ``--help``.
    from tool_sandbox.common.tool_discovery import ToolBackend
    from tool_sandbox.scenarios import named_scenarios

    # Collect every defined scenario across all categories and difficulty tiers.
    scenarios = named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT)

    result: List[Dict[str, Any]] = []
    for name, scenario in sorted(scenarios.items()):
        # The attrs classes used throughout the benchmark make `asdict`
        # convenient – it recursively converts the structure to Python
        # primitives *except* for polars which we handle in the JSON fallback.
        scenario_dict = asdict(scenario)
        scenario_entry: Dict[str, Any] = {
            "scenario_name": name,
            "data": scenario_dict,
        }
        result.append(scenario_entry)
    return result


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401 – simple name.
    parser = argparse.ArgumentParser(
        description=(
            "Dump the raw prompts and ground truths of every ToolSandbox "
            "benchmark task into a JSON file.  *No* filtering, cleaning or "
            "pre-processing is applied – the output precisely reflects the "
            "in-code benchmark definitions."
        )
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=Path("prompts_groundtruths.json"),
        help="Destination JSON file (default: ./prompts_groundtruths.json)",
    )
    args = parser.parse_args()

    data = build_prompts_groundtruths()
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False, default=_json_default)

    print(f"Wrote {len(data)} scenarios to {args.out_file}")


if __name__ == "__main__":
    main()
