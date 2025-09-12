import typing
import os

from src.utils.types import Benchmark
from src.bench_loaders import get_bench_loader

BENCHMARKS = [ 
    # Benchmark.TAU2_BENCH,
    # Benchmark.TAU_BENCH,
    Benchmark.ACE_BENCH,
    # Benchmark.NEXUS_BENCH,
    # Benchmark.TOOL_SANDBOX,
    # Benchmark.COMPLEX_FUNC_BENCH,
    # Benchmark.DRAFTER_BENCH,
    # Benchmark.BFCL,
    # Benchmark.MULTI_CHALLENGE
    ]

def test_bench_loader(benchmark: Benchmark):
    loader = get_bench_loader(benchmark)
    try:
        questions = loader().load_questions()
        print(f"Loaded {len(questions)} questions from {benchmark}")
        assert isinstance(questions, list)
        assert len(questions) > 0

        prev_task_name = None
        for idx, question in enumerate(questions):
            task_name = getattr(question, "task_name", None)
            if task_name != prev_task_name:
                print(f"\n--- task_name changed at index {idx}: {task_name} ---")
                question_dict = question.model_dump()
                for field, value in question_dict.items():
                    value_str = str(value)
                    print(f"=== {field} ===:\n {value_str}")
                prev_task_name = task_name
                input()
            else:
                continue
        question = questions[100]
        question_dict = question.model_dump()
        for field, value in question_dict.items():
            value_str = str(value)
            if len(value_str) > 100:
                print(f"=== {field} ===:\n {value_str[:500]}...")
            else:
                print(f"=== {field} ===:\n {value_str}")

    except NotImplementedError:
        print(f"Loader for {benchmark} is not implemented.")


if __name__ == "__main__":
    for benchmark in BENCHMARKS:
        print(f"Testing benchmark: {benchmark}")
        test_bench_loader(benchmark)