"""
DrafterBench-specific rule-based filtering.
Implements custom filtering logic for DrafterBench evaluation data.
"""

from typing import Dict, List, Tuple
from .base_filter import BaseBenchmarkFilter
import logging

logger = logging.getLogger(__name__)

class DrafterBenchFilter(BaseBenchmarkFilter):
    """DrafterBench-specific filtering rules."""
    
    def __init__(self):
        super().__init__("DrafterBench")
    
    def get_filter_name(self) -> str:
        return "DrafterBench-Specific Filter"
    
    def is_applicable(self, sample: Dict) -> bool:
        """Check if sample is from DrafterBench."""
        # DrafterBench samples typically have specific structure
        return (
            'task_name' in sample and 
            any(task_type in sample['task_name'] for task_type in [
                'add_', 'delete_', 'map_', 'refresh_', 'revise_'
            ])
        )
    
    def filter_samples(self, samples: Dict) -> Tuple[List[Dict], List[Dict]]:
        """
        Apply DrafterBench-specific filtering rules.
        
        DrafterBench-specific rules:
        1. Must have valid task_name with operation type
        2. Score must be between 0-100 (DrafterBench scale)
        3. Must have sufficient model responses for comparison
        4. Task must show meaningful performance variation
        """
        logger.info(f"Applying DrafterBench-specific filtering to {len(samples)} samples")
        
        # Filter 1: Basic structure validation
        structure_valid = self._filter_by_structure(samples)
        logger.info(f"Structure validation: {len(structure_valid)} samples passed")
        
        # Filter 2: Score sanity check (DrafterBench uses 0-100 scale)
        score_valid = self._filter_by_score_sanity(structure_valid)
        logger.info(f"Score sanity check: {len(score_valid)} samples passed")
        
        # Filter 3: Question-level discriminativeness
        discriminative = self._filter_by_discriminativeness(score_valid)
        logger.info(f"Discriminativeness check: {len(discriminative)} samples passed")
        
        dropped = [s for s in samples if s not in discriminative]
        
        self.log_filtering_stats(len(samples), len(discriminative), len(dropped))
        
        return discriminative, dropped
    
    def _filter_by_structure(self, samples: List[Dict]) -> List[Dict]:
        """Filter by DrafterBench-specific structure requirements."""
        valid_samples = []
        
        for sample in samples:
            if not self.is_applicable(sample):
                continue
                
            # Must have required fields
            if not all(key in sample for key in ['task_name', 'eval_result', 'model_name']):
                continue
                
            # Must have messages
            if 'messages' not in sample or not sample['messages']:
                continue
                
            valid_samples.append(sample)
        
        return valid_samples
    
    def _filter_by_score_sanity(self, samples: List[Dict]) -> List[Dict]:
        """Filter by DrafterBench score sanity (0-100 scale)."""
        valid_samples = []
        
        for sample in samples:
            eval_result = sample.get('eval_result', {})
            score = eval_result.get('score')
            
            if score is None:
                continue
                
            try:
                score_val = float(score)
                # DrafterBench uses 0-100 scale
                if 0 <= score_val <= 100:
                    valid_samples.append(sample)
            except (ValueError, TypeError):
                continue
        
        return valid_samples
    
    def _filter_by_discriminativeness(self, samples: List[Dict]) -> List[Dict]:
        """Filter by question-level discriminativeness."""
        # Group by question (task_name + operation type)
        question_groups = {}
        for sample in samples:
            task_key = sample['task_name']
            if task_key not in question_groups:
                question_groups[task_key] = []
            question_groups[task_key].append(sample)
        
        # Evaluate discriminativeness for each question
        discriminative_samples = []
        
        for question, question_samples in question_groups.items():
            if len(question_samples) < 2:
                continue  # Need at least 2 models for comparison
            
            # Extract scores
            scores = []
            for sample in question_samples:
                score = sample.get('eval_result', {}).get('score')
                if score is not None:
                    try:
                        scores.append(float(score))
                    except (ValueError, TypeError):
                        continue
            
            if len(scores) < 2:
                continue
            
            # Calculate variance (normalize to 0-1 scale)
            normalized_scores = [s / 100.0 for s in scores]
            variance = sum((x - sum(normalized_scores)/len(normalized_scores))**2 for x in normalized_scores) / len(normalized_scores)
            
            # Question is discriminative if variance > threshold
            if variance > 0.01:  # Same threshold as general filter
                discriminative_samples.extend(question_samples)
        
        return discriminative_samples

