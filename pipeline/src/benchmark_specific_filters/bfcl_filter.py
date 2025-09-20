"""
BFCL-specific rule-based filtering.
Implements custom filtering logic for BFCL evaluation data.
"""

from typing import Dict, List, Tuple
from .base_filter import BaseBenchmarkFilter
import logging

logger = logging.getLogger(__name__)

class BFCLFilter(BaseBenchmarkFilter):
    """BFCL-specific filtering rules."""
    
    def __init__(self):
        super().__init__("BFCL")
    
    def get_filter_name(self) -> str:
        return "BFCL-Specific Filter"
    
    def is_applicable(self, sample: Dict) -> bool:
        """Check if sample is from BFCL."""
        return (
            'task_name' in sample and 
            any(task_type in sample['task_name'] for task_type in [
                'parallel', 'multiple', 'simple', 'irrelevance', 'live_'
            ])
        )
    
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
            if sample["task_name"].startswith("multi_turn"):
                passed_samples.append(sample)
            else:
                dropped_samples.append(sample)

        logger.info(f"BFCL filtering completed: {len(passed_samples)} passed, {len(dropped_samples)} dropped")
        return passed_samples, dropped_samples

