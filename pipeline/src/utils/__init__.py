from typing import List, Dict
from collections import defaultdict
from enum import Enum
import json
import logging
from .types import UniqueQuestionID, Benchmark

logger = logging.getLogger(__name__)

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


def compute_confusion_matrix(problematic_ids: set, passed_ids: set, total_num: int) -> None:
    """Compute and log confusion matrix for filtering performance.

    Args:
        problematic_ids: Set of question IDs that are known to be problematic
        passed_ids: Set of question IDs that passed all filters
        total_num: Total number of samples
    """

    # Calculate confusion matrix components
    tp = len(problematic_ids - passed_ids)  # True Positive: problematic samples that were filtered
    fn = len(problematic_ids & passed_ids)  # False Negative: problematic samples that passed

    total_filtered = total_num - len(passed_ids)  # Total samples that were filtered
    fp = total_filtered - tp  # False Positive: normal samples that were filtered
    tn = len(passed_ids) - fn  # True Negative: normal samples that passed

    # Calculate total samples for verification
    total_samples = tp + fp + fn + tn
    total_problematic = len(problematic_ids)
    total_normal = total_num - total_problematic

    logger.info("=== Confusion Matrix ===")
    logger.info(f"               Filtered | Passed")
    logger.info(f" Problematic     {tp:4d}   |  {fn:4d} = {total_problematic}")
    logger.info(f" Normal          {fp:4d}   |  {tn:4d} = {total_normal}")
    logger.info(f"                 {tp+fp:4d}   |  {fn+tn:4d} = {total_samples}")
    logger.info("=" * 45)

    # Calculate performance metrics
    if total_problematic > 0:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        logger.info(f"  Precision: {precision:.3f} ({tp}/{tp + fp})")
        logger.info(f"  Recall:    {recall:.3f} ({tp}/{tp + fn})")
        logger.info(f"  F1-Score:  {f1_score:.3f}")
        logger.info(f"  Accuracy:  {(tp + tn) / total_samples:.3f} ({tp + tn}/{total_samples})")



def log_confusion_matrix(problematic_issues: Dict, passed_ids: set, total_num: int) -> None:
    """Log confusion matrix for all problematic issues and manually annotated ones.

    Args:
        problematic_issues: Dict mapping benchmark names to dicts of {UniqueQuestionID: {"reason": str, "source": str}}
        passed_ids: Set of question IDs that passed all filters
        total_num: Total number of samples
    """
    # Extract all problematic question IDs
    all_problematic_ids = set(problematic_issues.keys())

    # Log confusion matrix for all problematic issues
    logger.info("=== All Problematic Issues ===")
    compute_confusion_matrix(all_problematic_ids, passed_ids, total_num)

    # Extract manually annotated problematic IDs
    manually_ids = set()
    for question_id, info in problematic_issues.items():
        if info.get("source") == "manually":
            manually_ids.add(question_id)

    # Log confusion matrix for manually annotated issues
    if manually_ids:
        logger.info("=== Manually Annotated Issues ===")
        compute_confusion_matrix(manually_ids, passed_ids, total_num)
    else:
        logger.info("No manually annotated problematic issues found")