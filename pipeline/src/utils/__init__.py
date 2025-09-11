from typing import List, Dict
from collections import defaultdict
from enum import Enum
import json
from .types import UniqueQuestionID, Benchmark

def normalize_benchmark_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "")

def get_benchmark_from_name(benchmark_name: str) -> Benchmark:
    benchmark_name_normlized = normalize_benchmark_name(benchmark_name)
    for benchmark in Benchmark:
        if benchmark_name_normlized == normalize_benchmark_name(benchmark.value):
            return benchmark
    raise ValueError(f"No benchmark with name {benchmark_name}")


def group_responses_by_question(responses: List[Dict]) -> Dict[UniqueQuestionID, List[Dict]]:
    result = defaultdict(list)
    for response in responses:
        unique_question_id = UniqueQuestionID(
            benchmark=response["benchmark_name"],  
            task_name=response.get("task_name", None),
            question_id=response["meta"]["id"]
        )
        result[unique_question_id].append(response)
    return dict(result)


class EnumJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts Enum values to their string representation."""
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)