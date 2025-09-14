"""Utility script to quickly print a NexusBench filtering prompt.

Run it from any directory (e.g. `python -m src.print_nexusbench_prompt`).
The script dynamically injects the project root into ``sys.path`` so that the
internal package imports resolve no matter where it is executed from.
"""

import os
import sys

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)


from src.bench_loaders.nexus_bench_loader import NexusBenchLoader
from src.utils.format_judge_prompt import format_judge_prompt
from src.utils.types import LLMJudgeStep


def main():
    """Load the first NexusBench question and print its filtering prompt.

    Running the script gives an immediate view of the fully-rendered evaluator prompt, which is useful for manual
    sanity-checking during development.
    """
    loader = NexusBenchLoader()
    # Load only the benchmark_name benchmark to speed things up
    benchmark_name = "LangChainMath"
    # First attempt: use specific benchmark loader (requires nexusbench package)
    questions = loader.load_specific_benchmark(benchmark_name)

    # Fallback: load all questions via HF datasets and filter
    if not questions:
        all_questions = loader.load_questions()
        questions = [q for q in all_questions if getattr(q, "benchmark_name", None) == benchmark_name]

    if not questions:
        raise RuntimeError(f"No {benchmark_name} questions could be loaded.")

    # Generate the filtering prompt for the first benchmark sample
    prompt = format_judge_prompt(questions[0], LLMJudgeStep.SPECIFIC_FILTER)
    print(prompt)


if __name__ == "__main__":
    main()
