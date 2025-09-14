#!/usr/bin/env python3
"""
Comprehensive rule-based filtering module.
Focuses ONLY on question-level discriminative quality for LLM performance evaluation.
"""

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
        self, responses_by_question: Dict[str, List[Dict]]
    ) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
        """
        Filter samples based on question-level discriminativeness with difficulty-based filtering.
        Returns: (passed_responses, dropped_responses)
        """
        total_samples = sum(
            len(responses) for responses in responses_by_question.values()
        )
        logger.info(
            f"Starting question-level discriminativeness filtering on {total_samples} samples"
        )

        logger.info(f"Found {len(responses_by_question)} unique questions")

        # Step 1: Classify questions by difficulty and discriminativeness
        too_easy_questions = set()
        too_hard_questions = set()
        discriminative_questions = set()
        non_discriminative_questions = set()

        for question_id, question_samples in responses_by_question.items():
            difficulty_classification = self._classify_question_difficulty(
                question_samples
            )

            if difficulty_classification == "too_easy":
                too_easy_questions.add(question_id)
            elif difficulty_classification == "too_hard":
                too_hard_questions.add(question_id)
            elif self._is_question_discriminative(question_samples):
                discriminative_questions.add(question_id)
            else:
                non_discriminative_questions.add(question_id)

        logger.info(f"Too easy questions: {len(too_easy_questions)}")
        logger.info(f"Too hard questions: {len(too_hard_questions)}")
        logger.info(f"Discriminative questions: {len(discriminative_questions)}")
        logger.info(
            f"Non-discriminative questions: {len(non_discriminative_questions)}"
        )

        # Step 3: Calculate original task type counts for retention logic
        all_samples = [
            sample
            for responses in responses_by_question.values()
            for sample in responses
        ]
        original_task_counts = self._calculate_original_task_counts(all_samples)

        # Step 2: Apply filtering logic
        # - Keep ALL samples for discriminative questions
        # - Keep ALL samples for too_hard questions (no filtering)
        # - Keep enough too_easy questions to ensure each task type retains at least 10% of original size
        # - Drop ALL samples for non-discriminative questions
        passed_responses = {}
        dropped_responses = {}

        # Group too_easy questions by task type for retention sampling
        too_easy_by_task = self._group_questions_by_task_type(
            too_easy_questions, all_samples
        )

        for question_id, question_responses in responses_by_question.items():
            if question_id in discriminative_questions:
                # Keep all discriminative questions
                passed_responses[question_id] = question_responses
            elif question_id in too_hard_questions:
                # Keep all too_hard questions (no filtering)
                passed_responses[question_id] = question_responses
            elif question_id in too_easy_questions:
                # Keep enough too_easy questions to ensure task type retention
                # For too_easy questions, we keep all responses for the question if we decide to keep the question
                sample = question_responses[
                    0
                ]  # Use first sample to determine task type
                task_type = self._extract_task_type(sample)
                if self._should_keep_too_easy_sample(
                    question_id, task_type, too_easy_by_task, original_task_counts
                ):
                    passed_responses[question_id] = question_responses
                else:
                    dropped_responses[question_id] = question_responses
            else:
                # Drop non-discriminative questions
                dropped_responses[question_id] = question_responses

        passed_count = sum(len(responses) for responses in passed_responses.values())
        dropped_count = sum(len(responses) for responses in dropped_responses.values())
        logger.info(
            f"Final results: {passed_count} samples passed, {dropped_count} dropped"
        )

        return passed_responses, dropped_responses

    def _group_samples_by_question(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """Group samples by their question/task identifier."""
        question_groups = defaultdict(list)

        for sample in samples:
            question_id = self._extract_question_id(sample)
            question_groups[question_id].append(sample)

        return dict(question_groups)

    def _extract_question_id(self, sample: Dict) -> str:
        """Extract a unique identifier for the question/task."""
        # Try different possible fields for question identification
        if "task_name" in sample and "meta" in sample and "id" in sample["meta"]:
            # For ACEBench, the id field already contains the full question ID
            return sample["meta"]["id"]
        elif "task_name" in sample:
            return sample["task_name"]
        elif "question" in sample:
            return sample["question"]
        elif "prompt" in sample:
            return sample["prompt"]
        elif "messages" in sample and sample["messages"]:
            # Use first message content as question identifier
            first_msg = sample["messages"][0]
            if isinstance(first_msg, dict) and "content" in first_msg:
                return first_msg["content"][:100]  # First 100 chars
        elif "conversation" in sample and sample["conversation"]:
            # Use first turn as question identifier
            first_turn = sample["conversation"][0]
            if isinstance(first_turn, dict) and "content" in first_turn:
                return first_turn["content"][:100]

        # Fallback: use a hash of the entire sample
        return hashlib.md5(json.dumps(sample, sort_keys=True).encode()).hexdigest()

    def _is_question_discriminative(self, question_samples: List[Dict]) -> bool:
        """
        Determine if a question is discriminative based on model performance variation.
        Returns True if the question helps distinguish between different LLM capabilities.
        """
        if len(question_samples) < 2:
            return False  # Need at least 2 model responses to compare

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
            return False

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
            return False

        # Calculate variance to measure discriminativeness
        variance = np.var(numeric_scores)

        # Question is discriminative if there's sufficient variance in scores
        # This means different models perform differently on this question
        return variance > 0.01  # Threshold for meaningful variation

    def _classify_question_difficulty(self, question_samples: List[Dict]) -> str:
        """
        Classify a question as 'too_easy', 'too_hard', or 'normal' based on model performance.
        Returns: 'too_easy', 'too_hard', or 'normal'
        """
        if len(question_samples) < 2:
            return "normal"  # Need at least 2 model responses to classify

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
            return "normal"

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
            return "normal"

        # Calculate statistics
        mean_score = np.mean(numeric_scores)

        # Classification logic:
        # - Too easy: High mean score (>0.8) with low variance (<0.01)
        # - Too hard: Low mean score (<0.2) with low variance (<0.01)
        # - Normal: Everything else (including high variance cases)

        # if mean_score > 0.8 and variance < 0.01:
        #     return 'too_easy'
        # elif mean_score < 0.2 and variance < 0.01:
        #     return 'too_hard'
        # Update to too_easy > 0.9, too_hard < 0.1
        if mean_score > 0.9:
            return "too_easy"
        elif mean_score < 0.1:
            return "too_hard"
        else:
            return "normal"

    def _extract_task_type(self, sample: Dict) -> str:
        """Extract task type from sample for grouping too_easy questions."""
        # Try to extract task type from various possible fieldsge
        if "task_name" in sample:
            # Extract the main task category from task_name
            task_name = sample["task_name"]
            # Split by common separators and take the first part
            parts = task_name.split("_")[0].split("-")[0].split("/")[0]
            return parts.lower()
        elif "benchmark" in sample:
            return sample["benchmark"].lower()
        elif "meta" in sample and "task_type" in sample["meta"]:
            return sample["meta"]["task_type"].lower()
        else:
            # Fallback: use a hash of the first part of the question
            question_id = self._extract_question_id(sample)
            return hashlib.md5(question_id[:50].encode()).hexdigest()[:8]

    def _group_questions_by_task_type(
        self, question_ids: set, samples: List[Dict]
    ) -> Dict[str, List[str]]:
        """Group question IDs by task type for 10% sampling."""
        task_groups = defaultdict(list)

        # Create a mapping from question_id to task_type
        question_to_task = {}
        # Convert question_ids to strings for comparison
        question_id_strings = set()
        for qid in question_ids:
            if hasattr(qid, 'question_id'):
                question_id_strings.add(qid.question_id)
            else:
                question_id_strings.add(str(qid))
        
        for sample in samples:
            question_id = self._extract_question_id(sample)
            if question_id in question_id_strings:
                task_type = self._extract_task_type(sample)
                question_to_task[question_id] = task_type

        # Group questions by task type
        for question_id in question_ids:
            # Handle both string and UniqueQuestionID objects
            if hasattr(question_id, 'question_id'):
                # It's a UniqueQuestionID object, use the question_id field directly
                question_id_str = question_id.question_id
            else:
                # It's already a string
                question_id_str = str(question_id)
            
            task_type = question_to_task.get(question_id_str, "unknown")
            task_groups[task_type].append(question_id_str)

        return dict(task_groups)

    def _calculate_original_task_counts(self, samples: List[Dict]) -> Dict[str, int]:
        """Calculate the original count of questions per task type before any filtering."""
        task_counts = defaultdict(int)

        # Get all unique question IDs and their task types
        seen_questions = set()
        for sample in samples:
            question_id = self._extract_question_id(sample)
            if question_id not in seen_questions:
                task_type = self._extract_task_type(sample)
                task_counts[task_type] += 1
                seen_questions.add(question_id)

        return dict(task_counts)

    def _should_keep_too_easy_sample(
        self,
        question_id: str,
        task_type: str,
        too_easy_by_task: Dict[str, List[str]],
        original_task_counts: Dict[str, int] = None,
    ) -> bool:
        """
        Determine if a too_easy sample should be kept.
        Updated logic: Keep enough too_easy samples to ensure each task type retains at least 10% of original size.
        """
        if task_type not in too_easy_by_task:
            return False

        questions_in_task = too_easy_by_task[task_type]
        if not questions_in_task:
            return False

        # Calculate how many questions we need to keep for this task type
        if original_task_counts and task_type in original_task_counts:
            original_count = original_task_counts[task_type]
            min_retention_count = max(
                1, int(original_count * 0.1)
            )  # At least 10% of original
        else:
            # Fallback to original logic if no original counts provided
            min_retention_count = max(1, int(len(questions_in_task) * 0.1))

        # Use deterministic sampling based on question_id hash
        # This ensures the same questions are selected consistently across runs

        # Sort questions deterministically and select the first min_retention_count
        sorted_questions = sorted(questions_in_task)
        questions_to_keep = sorted_questions[:min_retention_count]

        # Handle both string and UniqueQuestionID objects
        if hasattr(question_id, 'question_id'):
            # It's a UniqueQuestionID object, use the question_id field directly
            question_id_str = question_id.question_id
        else:
            # It's already a string
            question_id_str = str(question_id)

        return question_id_str in questions_to_keep
