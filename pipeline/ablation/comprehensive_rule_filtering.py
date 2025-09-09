#!/usr/bin/env python3
"""
Comprehensive rule-based filtering module.
Focuses ONLY on question-level discriminative quality for LLM performance evaluation.
"""

import json
import hashlib
import re
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class ComprehensiveRuleFilter:
    """Comprehensive rule-based filtering focused ONLY on question discriminativeness."""
    
    def __init__(self):
        pass
        
    def filter_samples(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter samples based on question-level discriminativeness with difficulty-based filtering.
        Returns: (passed_samples, dropped_samples)
        """
        logger.info(f"Starting question-level discriminativeness filtering on {len(samples)} samples")
        
        # Step 1: Group samples by question/task
        question_groups = self._group_samples_by_question(samples)
        logger.info(f"Found {len(question_groups)} unique questions")
        
        # Step 2: Classify questions by difficulty and discriminativeness
        too_easy_questions = set()
        too_hard_questions = set()
        discriminative_questions = set()
        non_discriminative_questions = set()
        
        for question_id, question_samples in question_groups.items():
            difficulty_classification = self._classify_question_difficulty(question_samples)
            
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
        logger.info(f"Non-discriminative questions: {len(non_discriminative_questions)}")
        
        # Print detailed statistics for each category
        self._print_question_statistics(too_easy_questions, question_groups, "TOO EASY")
        self._print_question_statistics(too_hard_questions, question_groups, "TOO HARD")
        # self._print_question_statistics(discriminative_questions, question_groups, "DISCRIMINATIVE")
        self._print_question_statistics(non_discriminative_questions, question_groups, "NON-DISCRIMINATIVE")
        
        # Step 3: Apply filtering logic
        # - Keep ALL samples for discriminative questions
        # - Keep ALL samples for too_hard questions (no filtering)
        # - Keep only 10% of samples for too_easy questions per task type
        # - Drop ALL samples for non-discriminative questions
        passed_samples = []
        dropped_samples = []
        
        # Group too_easy questions by task type for 10% sampling
        too_easy_by_task = self._group_questions_by_task_type(too_easy_questions, samples)
        
        for sample in samples:
            question_id = self._extract_question_id(sample)
            
            if question_id in discriminative_questions:
                # Keep all discriminative questions
                passed_samples.append(sample)
            elif question_id in too_hard_questions:
                # Keep all too_hard questions (no filtering)
                # passed_samples.append(sample)
                dropped_samples.append(sample)
            elif question_id in too_easy_questions:
                # Keep only 10% of too_easy questions per task type
                task_type = self._extract_task_type(sample)
                if self._should_keep_too_easy_sample(question_id, task_type, too_easy_by_task):
                    passed_samples.append(sample)
                else:
                    dropped_samples.append(sample)
            else:
                # Drop non-discriminative questions
                dropped_samples.append(sample)
        
        logger.info(f"Final results: {len(passed_samples)} samples passed, {len(dropped_samples)} dropped")
        
        return passed_samples, dropped_samples
    
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
        if 'task_name' in sample and 'meta' in sample and 'id' in sample['meta']:
            return sample['task_name'] + "_" + sample['meta']['id']
        elif 'task_name' in sample:
            return sample['task_name']
        elif 'question' in sample:
            return sample['question']
        elif 'prompt' in sample:
            return sample['prompt']
        elif 'messages' in sample and sample['messages']:
            # Use first message content as question identifier
            first_msg = sample['messages'][0]
            if isinstance(first_msg, dict) and 'content' in first_msg:
                return first_msg['content'][:100]  # First 100 chars
        elif 'conversation' in sample and sample['conversation']:
            # Use first turn as question identifier
            first_turn = sample['conversation'][0]
            if isinstance(first_turn, dict) and 'content' in first_turn:
                return first_turn['content'][:100]
        
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
            if 'eval_result' in sample and 'score' in sample['eval_result']:
                scores.append(sample['eval_result']['score'])
            elif 'eval_result' in sample and 'scores' in sample['eval_result']:
                scores.extend(sample['eval_result']['scores'])
            elif 'score' in sample:
                scores.append(sample['score'])
            elif 'scores' in sample:
                scores.extend(sample['scores'])
        
        if not scores:
            return False
        
        # Convert to numeric scores
        numeric_scores = []
        for score in scores:
            if isinstance(score, (int, float)):
                numeric_scores.append(float(score))
            elif isinstance(score, dict) and 'score' in score:
                try:
                    numeric_scores.append(float(score['score']))
                except (ValueError, TypeError):
                    continue
        
        if len(numeric_scores) < 2:
            return False
        
        # Calculate variance to measure discriminativeness
        variance = np.var(numeric_scores)
        
        # Question is discriminative if there's sufficient variance in scores
        # This means different models perform differently on this question
        # return variance > 0.01  # Threshold for meaningful variation
        return variance > 0.1
    
    def _classify_question_difficulty(self, question_samples: List[Dict]) -> str:
        """
        Classify a question as 'too_easy', 'too_hard', or 'normal' based on model performance.
        Returns: 'too_easy', 'too_hard', or 'normal'
        """
        if len(question_samples) < 2:
            return 'normal'  # Need at least 2 model responses to classify
        
        # Extract scores for this question
        scores = []
        for sample in question_samples:
            # Try different possible score locations
            if 'eval_result' in sample and 'score' in sample['eval_result']:
                scores.append(sample['eval_result']['score'])
            elif 'eval_result' in sample and 'scores' in sample['eval_result']:
                scores.extend(sample['eval_result']['scores'])
            elif 'score' in sample:
                scores.append(sample['score'])
            elif 'scores' in sample:
                scores.extend(sample['scores'])
        
        if not scores:
            return 'normal'
        
        # Convert to numeric scores
        numeric_scores = []
        for score in scores:
            if isinstance(score, (int, float)):
                numeric_scores.append(float(score))
            elif isinstance(score, dict) and 'score' in score:
                try:
                    numeric_scores.append(float(score['score']))
                except (ValueError, TypeError):
                    continue
        
        if len(numeric_scores) < 2:
            return 'normal'
        
        # Calculate statistics
        mean_score = np.mean(numeric_scores)
        variance = np.var(numeric_scores)
        
        # Classification logic:
        # - Too easy: High mean score (>0.8) with low variance (<0.01)
        # - Too hard: Low mean score (<0.2) with low variance (<0.01)
        # - Normal: Everything else (including high variance cases)
        
        # if mean_score > 0.8 and variance < 0.01:
        if mean_score > 0.9:
            return 'too_easy'
        # elif mean_score < 0.2 and variance < 0.01:
        elif mean_score < 0.1:
            return 'too_hard'
        else:
            return 'normal'
    
    def _extract_task_type(self, sample: Dict) -> str:
        """Extract task type from sample for grouping too_easy questions."""
        # Try to extract task type from various possible fieldsge
        if 'task_name' in sample:
            # Extract the main task category from task_name
            task_name = sample['task_name']
            # Split by common separators and take the first part
            parts = task_name.split('_')[0].split('-')[0].split('/')[0]
            return parts.lower()
        elif 'benchmark' in sample:
            return sample['benchmark'].lower()
        elif 'meta' in sample and 'task_type' in sample['meta']:
            return sample['meta']['task_type'].lower()
        else:
            # Fallback: use a hash of the first part of the question
            question_id = self._extract_question_id(sample)
            return hashlib.md5(question_id[:50].encode()).hexdigest()[:8]
    
    def _group_questions_by_task_type(self, question_ids: set, samples: List[Dict]) -> Dict[str, List[str]]:
        """Group question IDs by task type for 10% sampling."""
        task_groups = defaultdict(list)
        
        # Create a mapping from question_id to task_type
        question_to_task = {}
        for sample in samples:
            question_id = self._extract_question_id(sample)
            if question_id in question_ids:
                task_type = self._extract_task_type(sample)
                question_to_task[question_id] = task_type
        
        # Group questions by task type
        for question_id in question_ids:
            task_type = question_to_task.get(question_id, 'unknown')
            task_groups[task_type].append(question_id)
        
        return dict(task_groups)
    
    def _should_keep_too_easy_sample(self, question_id: str, task_type: str, too_easy_by_task: Dict[str, List[str]]) -> bool:
        """
        Determine if a too_easy sample should be kept (10% per task type).
        Uses deterministic sampling based on question_id hash for consistency.
        """
        if task_type not in too_easy_by_task:
            return False
        
        questions_in_task = too_easy_by_task[task_type]
        if not questions_in_task:
            return False
        
        # Use deterministic sampling based on question_id hash
        # This ensures the same 10% is selected consistently across runs
        hash_value = int(hashlib.md5(question_id.encode()).hexdigest(), 16)
        sample_index = hash_value % len(questions_in_task)
        
        # Keep 10% of questions per task type
        keep_ratio = 0.1
        keep_ratio = 0.0
        keep_count = max(1, int(len(questions_in_task) * keep_ratio))
        
        # Sort questions deterministically and select the first keep_count
        sorted_questions = sorted(questions_in_task)
        questions_to_keep = sorted_questions[:keep_count]
        
        return question_id in questions_to_keep
    
    def _print_question_statistics(self, question_ids: set, question_groups: Dict[str, List[Dict]], category: str):
        """Print detailed statistics for questions in a specific category."""
        if not question_ids:
            logger.info(f"\n{category} QUESTIONS: None found")
            return
            
        logger.info(f"\n{category} QUESTIONS ({len(question_ids)} total):")
        logger.info("=" * 80)
        
        for question_id in sorted(question_ids):
            question_samples = question_groups[question_id]
            scores = self._extract_scores_from_samples(question_samples)
            
            if scores:
                mean_score = np.mean(scores)
                variance = np.var(scores)
                std_dev = np.std(scores)
                min_score = np.min(scores)
                max_score = np.max(scores)
                
                logger.info(f"QID: {question_id[:60]}{'...' if len(question_id) > 60 else ''}")
                logger.info(f"  Mean: {mean_score:.4f}, Variance: {variance:.4f}, StdDev: {std_dev:.4f}")
                logger.info(f"  Range: [{min_score:.4f}, {max_score:.4f}], Samples: {len(scores)}")
                logger.info("")
            else:
                logger.info(f"QID: {question_id[:60]}{'...' if len(question_id) > 60 else ''}")
                logger.info(f"  No valid scores found, Samples: {len(question_samples)}")
                logger.info("")
    
    def _extract_scores_from_samples(self, question_samples: List[Dict]) -> List[float]:
        """Extract numeric scores from question samples."""
        scores = []
        for sample in question_samples:
            # Try different possible score locations
            if 'eval_result' in sample and 'score' in sample['eval_result']:
                scores.append(sample['eval_result']['score'])
            elif 'eval_result' in sample and 'scores' in sample['eval_result']:
                scores.extend(sample['eval_result']['scores'])
            elif 'score' in sample:
                scores.append(sample['score'])
            elif 'scores' in sample:
                scores.extend(sample['scores'])
        
        # Convert to numeric scores
        numeric_scores = []
        for score in scores:
            if isinstance(score, (int, float)):
                numeric_scores.append(float(score))
            elif isinstance(score, dict) and 'score' in score:
                try:
                    numeric_scores.append(float(score['score']))
                except (ValueError, TypeError):
                    continue
        
        return numeric_scores
