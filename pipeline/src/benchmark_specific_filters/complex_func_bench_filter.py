"""
ComplexFuncBench-specific rule-based filtering.
Implements custom filtering logic for ComplexFuncBench evaluation data.
"""

from typing import Dict, List, Tuple
from .base_filter import BaseBenchmarkFilter
import logging

logger = logging.getLogger(__name__)

class ComplexFuncBenchFilter(BaseBenchmarkFilter):
    """ComplexFuncBench-specific filtering rules."""
    
    def __init__(self):
        super().__init__("ComplexFuncBench")
    
    def get_filter_name(self) -> str:
        return "ComplexFuncBench-Specific Filter"
    
    def is_applicable(self, sample: Dict) -> bool:
        """Check if sample is from ComplexFuncBench."""
        return (
            'task_name' in sample and 
            any(task_type in sample['task_name'] for task_type in [
                'attraction', 'flights', 'hotels', 'car-rental', 'cross'
            ])
        )
    
    def filter_samples(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Apply ComplexFuncBench-specific filtering rules.
        
        TODO: Implement custom filtering logic for ComplexFuncBench
        For now, using general discriminativeness filtering
        """
        logger.info(f"Applying ComplexFuncBench-specific filtering to {len(samples)} samples")
        
        # For now, use general discriminativeness filtering
        # TODO: Implement benchmark-specific rules
        from ..comprehensive_rule_filtering import ComprehensiveRuleFilter
        general_filter = ComprehensiveRuleFilter()
        return general_filter.filter_samples(samples)

