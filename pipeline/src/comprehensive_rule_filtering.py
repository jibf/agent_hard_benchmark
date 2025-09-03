#!/usr/bin/env python3
"""
Comprehensive rule-based filtering module.
Focuses ONLY on question-level discriminative quality for LLM performance evaluation.
"""

import json
import hashlib
import re
import numpy as np
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
        Filter samples based ONLY on question-level discriminativeness.
        Returns: (passed_samples, dropped_samples)
        """
        logger.info(f"Starting question-level discriminativeness filtering on {len(samples)} samples")
        
        # Step 1: Group samples by question/task
        question_groups = self._group_samples_by_question(samples)
        logger.info(f"Found {len(question_groups)} unique questions")
        
        # Step 2: Evaluate discriminativeness of each question
        discriminative_questions = set()
        non_discriminative_questions = set()
        
        for question_id, question_samples in question_groups.items():
            if self._is_question_discriminative(question_samples):
                discriminative_questions.add(question_id)
            else:
                non_discriminative_questions.add(question_id)
        
        logger.info(f"Discriminative questions: {len(discriminative_questions)}")
        logger.info(f"Non-discriminative questions: {len(non_discriminative_questions)}")
        
        # Step 3: Keep ALL samples for discriminative questions, drop ALL for non-discriminative
        passed_samples = []
        dropped_samples = []
        
        for sample in samples:
            question_id = self._extract_question_id(sample)
            if question_id in discriminative_questions:
                passed_samples.append(sample)
            else:
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
        return variance > 0.01  # Threshold for meaningful variation
