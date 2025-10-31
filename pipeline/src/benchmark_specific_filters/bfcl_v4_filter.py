"""
BFCL-specific rule-based filtering.
Implements custom filtering logic for BFCL evaluation data.
"""

import re
from typing import Dict, List, Tuple
from .base_filter import BaseBenchmarkFilter
import logging

logger = logging.getLogger(__name__)

class BFCLV4Filter(BaseBenchmarkFilter):
    """BFCL-V4-specific filtering rules."""
    
    def __init__(self):
        super().__init__("BFCL_V4")
    
    def get_filter_name(self) -> str:
        return "BFCL-V4-Specific Filter"
    
    def is_applicable(self, sample: Dict) -> bool:
        """Check if sample is from BFCL."""
        return (
            'task_name' in sample and 
            any(task_type in sample['task_name'] for task_type in [
                'web_search', 'memory'
            ])
        )

    def _convert_to_dataset_qid(self, question_id: str) -> str:
        # In BFCLv4, question ids in the dataset are transformed according to the question variant type 
        #   e.g., memory_154-notetaker-24 -> memory_vector_154-notetaker-24, memory_kv_154-notetaker-24, memory_rec_sum_151-notetaker-21
        #         web_search_89 -> web_search_no_snippet_89 or web_search_base_89
        # This function converts back to the question ids specified in the dataset
        if question_id.startswith("memory"):
            qid_pattern = re.compile(r"memory_([a-z_]+)_\d+-[a-z]+-\d+")
        elif question_id.startswith("web_search"):
            qid_pattern = re.compile(r"web_search_(no_snippet|base)_\d+")
        else:
            raise ValueError(f"{question_id} is invalid question id for BFCL V4")
        match = qid_pattern.match(question_id)
        if not match:
            return question_id 
        start, end = match.span(1)
        return question_id[:start] + question_id[end+1:]


    
    def filter_samples(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Apply BFCL-specific filtering rules.
        Note: Comprehensive filtering has already been applied before this method is called.
        
        TODO: Implement custom filtering logic for BFCL
        For now, returning samples as-is since comprehensive filtering was already applied
        """
        logger.info(f"Applying BFCL-specific filtering to {len(samples)} samples")
        
        # For now, return samples as-is since comprehensive filtering was already applied
        # TODO: Implement benchmark-specific rules

        passed_samples, dropped_samples = [], []
        for sample in samples:
            if self.is_applicable(sample):
                sample["meta"]["id"] = self._convert_to_dataset_qid(sample["meta"]["id"])
                sample["task_name"] = "web_search" if "web_search" in sample["task_name"] else "memory"
                passed_samples.append(sample)
            else:
                dropped_samples.append(sample)

        logger.info(f"BFCL V4 filtering completed: {len(passed_samples)} passed, {len(dropped_samples)} dropped")
        return passed_samples, dropped_samples

