from typing import List, Dict
from collections import defaultdict
from enum import Enum
import json
import logging
from .types import UniqueQuestionID, Benchmark
from .bfcl_v4 import normalize_bfcl_v4_task_info

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

        # align question id manually
        task_name = response.get("task_name", None)
        benchmark = response["benchmark_name"]
        question_id = response["meta"]["id"]
        if isinstance(benchmark, str):
            try:
                benchmark = get_benchmark_from_name(benchmark)
            except ValueError:
                pass
        if benchmark == Benchmark.BFCL_V4:
            task_name, question_id = normalize_bfcl_v4_task_info(
                task_name or "", question_id
            )
            if task_name == "format_sensitivity" or question_id.startswith("format_sensitivity"):
                continue
        if task_name and not question_id.startswith(task_name):
            question_id = task_name + "_" + question_id

        unique_question_id = UniqueQuestionID(
            benchmark=benchmark,
            task_name=task_name,
            question_id=question_id
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


def log_confusion_matrix_human_labelled(human_labelled_questions: set, human_labelled_details: Dict, passed_ids: set, input_ids: set) -> Dict:
    """Log confusion matrix for human labelled ground truth data and return metrics.

    Args:
        human_labelled_questions: Set of current human labelled question IDs (filtered by previous steps)
        human_labelled_details: Dict mapping question IDs to their details {"is_issue": "0"/"1", "issue_type": "..."}
        passed_ids: Set of question IDs that passed all filters
        input_ids: Set of question IDs that were input to the current step

    Returns:
        Dict: Dictionary containing precision, recall, f1, tp, fp, fn, tn values
    """
    # Find which human labelled questions are in the current step's input
    labelled_questions_in_input = human_labelled_questions & input_ids

    if not labelled_questions_in_input:
        logger.info("=== Human Labelled Ground Truth Evaluation ===")
        logger.info("No human labelled questions found in current step input")
        return {"precision": None, "recall": None, "f1": None, "tp": 0, "fp": 0, "fn": 0, "tn": 0}

    # Split into problematic (is_issue=1) and normal (is_issue=0) questions using human_labelled_details
    problematic_questions_in_input = {
        qid for qid in labelled_questions_in_input
        if human_labelled_details.get(qid, {}).get("is_issue") == "1"
    }
    normal_questions_in_input = labelled_questions_in_input - problematic_questions_in_input

    # Calculate confusion matrix components
    # TN: Normal questions that passed (correctly kept)
    tn = len(normal_questions_in_input & passed_ids)

    # TP: Problematic questions that were filtered (correctly removed)
    tp = len(problematic_questions_in_input - passed_ids)

    # FP: Normal questions that were filtered (incorrectly removed)
    fp = len(normal_questions_in_input - passed_ids)

    # FN: Problematic questions that passed (incorrectly kept)
    fn = len(problematic_questions_in_input & passed_ids)

    # Log confusion matrix
    logger.info("=== Confusion Matrix ===")
    logger.info(f"               Filtered | Passed")
    logger.info(f" Problematic     {tp:4d}   |  {fn:4d} = {len(problematic_questions_in_input)}")
    logger.info(f" Normal          {fp:4d}   |  {tn:4d} = {len(normal_questions_in_input)}")
    logger.info(f"                 {tp+fp:4d}   |  {fn+tn:4d} = {len(labelled_questions_in_input)}")
    logger.info("=" * 45)

    # Calculate performance metrics
    if len(problematic_questions_in_input) > 0:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(labelled_questions_in_input)

        logger.info(f"  Precision: {precision:.3f} ({tp}/{tp + fp})")
        logger.info(f"  Recall:    {recall:.3f} ({tp}/{tp + fn})")
        logger.info(f"  F1-Score:  {f1_score:.3f}")
        logger.info(f"  Accuracy:  {accuracy:.3f} ({tp + tn}/{len(labelled_questions_in_input)})")
    else:
        logger.info("No problematic questions in this step for metrics calculation")
        precision = recall = f1_score = None

    # Return metrics
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }
