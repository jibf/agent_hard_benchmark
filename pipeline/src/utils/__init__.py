import typing
from typing import List, Dict
from collections import defaultdict
from .types import UniqueQuestionID, Benchmark

def get_benchmark_from_name(benchmark_name: str) -> Benchmark:
    def normalize_name(name: str) -> str:
        return name.lower().replace("-", "").replace("_", "")

    benchmark_name_normlized = normalize_name(benchmark_name)
    for benchmark in Benchmark:
        if benchmark_name_normlized == normalize_name(benchmark.value):
            return benchmark
    raise ValueError(f"No benchmark with name {benchmark_name}")


def group_responses_by_question(responses: List[Dict]) -> Dict[UniqueQuestionID, List[Dict]]:
    result = defaultdict(list)
    for response in responses:
        unique_question_id = UniqueQuestionID(
            benchmark=get_benchmark_from_name(response["benchmark_name"]),
            task_name=response.get("task_name", None),
            question_id=response["meta"]["id"]
        )
        result[unique_question_id].append(response)
    return dict(result)