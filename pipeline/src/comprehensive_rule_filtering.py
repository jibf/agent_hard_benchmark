#!/usr/bin/env python3
"""
Comprehensive rule-based filtering module.
Focuses ONLY on question-level discriminative quality for LLM performance evaluation.
"""

import math
import json
import hashlib
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class ComprehensiveRuleFilter:
    """Comprehensive rule-based filtering focused ONLY on question discriminativeness."""

    def __init__(self):
        pass

    def filter_samples(
        self,
        responses_by_question: Dict[str, List[Dict]],
        all_responses_by_question: Dict[str, List[Dict]],
    ) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
        """
        Filter samples based on question-level discriminativeness with difficulty-based filtering.
        Returns: (passed_responses, dropped_responses)
        """

        # Step 1: Classify questions by difficulty and discriminativeness

        too_easy_dict = defaultdict(list)
        non_discrimination_dict = defaultdict(list)

        original_task_counts = self._calculate_original_task_counts(
            all_responses_by_question
        )
        current_task_counts = self._calculate_original_task_counts(
            responses_by_question
        )

        task_avail_count = {}
        for sub_task, curr_count in current_task_counts.items():
            original_count = original_task_counts[sub_task]
            least_count = math.ceil(0.1 * original_count)
            available_count = max(0, curr_count - least_count)
            task_avail_count[sub_task] = available_count

        for question_id, question_samples in responses_by_question.items():
            mean_score, var_score, is_binary = self._compute_mean_var_score(
                question_samples
            )
            task_name = question_id.task_name
            if mean_score is None or var_score is None:
                continue

            if mean_score > 0.9:
                too_easy_dict[task_name].append((question_id, mean_score))
            if mean_score < 0.1:
                continue

            var_th = 0.1 if is_binary else 0.01
            if var_score < var_th:
                non_discrimination_dict[task_name].append((question_id, var_score))
                continue

        # dropped all non-discrimination questions
        filtered_question_ids = []

        non_discrimination_count = 0
        too_easy_count = 0
        for sub_task, available_count in task_avail_count.items():
            var_score_list = sorted(
                non_discrimination_dict[sub_task], key=lambda x: x[1], reverse=False
            )
            filtered_ids = [qid for qid, _ in var_score_list[:available_count]]
            available_count -= len(filtered_ids)
            non_discrimination_count += len(filtered_ids)
            filtered_question_ids.extend(filtered_ids)
            if available_count <= 0:
                continue
            mean_score_list = sorted(
                too_easy_dict[sub_task], key=lambda x: x[1], reverse=True
            )
            num_too_easy = len(mean_score_list)
            mean_score_list = mean_score_list[:int(0.9*num_too_easy)] # reamin 10% too easy tasks
            filtered_ids = [qid for qid, _ in mean_score_list[:available_count]]
            too_easy_count += len(filtered_ids)
            filtered_question_ids.extend(filtered_ids)

        logger.info(f"Filtered {non_discrimination_count} non-discrimination questions")
        logger.info(f"Filtered {too_easy_count} too-easy questions")

        passed_question_response = {
            id: response
            for id, response in responses_by_question.items()
            if id not in filtered_question_ids
        }

        return passed_question_response

    def _compute_mean_var_score(
        self, question_samples: List[Dict]
    ) -> Tuple[float, float, bool]:
        """
        Classify a question as 'too_easy', 'too_hard', or 'normal' based on model performance.
        Returns: 'too_easy', 'too_hard', or 'normal'
        """
        if len(question_samples) < 2:
            return None, None, False  # Need at least 2 model responses to classify

        # Extract scores for this question
        scores = []
        for sample in question_samples:
            # Try different possible score locations
            if "eval_result" in sample and "score" in sample["eval_result"]:
                scores.append(sample["eval_result"]["score"])
            elif "eval_result" in sample and "scores" in sample["eval_result"]:
                scores.extend(sample["eval_result"]["scores"])
            elif "score" in sample:
                scores.append(sample["score"])
            elif "scores" in sample:
                scores.extend(sample["scores"])

        if not scores:
            return None, None, False

        # Convert to numeric scores
        numeric_scores = []
        for score in scores:
            if isinstance(score, (int, float)):
                numeric_scores.append(float(score))
            elif isinstance(score, dict) and "score" in score:
                try:
                    numeric_scores.append(float(score["score"]))
                except (ValueError, TypeError):
                    continue

        if len(numeric_scores) < 2:
            return None, None, False

        # Calculate statistics
        mean_score = np.mean(numeric_scores)
        variance = np.var(numeric_scores)

        # Question is discriminative if there's sufficient variance in scores
        # This means different models perform differently on this question
        # Check if numeric_scores are binary (all close to 0 or 1) or continuous (0~1)
        is_binary = all(
            np.isclose(score, 0.0) or np.isclose(score, 1.0) for score in numeric_scores
        )
        return float(mean_score), float(variance), is_binary

    def _calculate_original_task_counts(
        self, responses_by_question: Dict
    ) -> Dict[str, int]:
        """Calculate the original count of questions per task type before any filtering."""
        task_counts = defaultdict(int)

        # Get all unique question IDs and their task types
        for qid in responses_by_question:
            task_type = qid.task_name
            task_counts[task_type] += 1

        return dict(task_counts)
