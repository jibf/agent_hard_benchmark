"""
Base class for benchmark-specific rule-based filtering.
All benchmark filters should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class BaseBenchmarkFilter(ABC):
    """Abstract base class for benchmark-specific filtering."""
    
    def __init__(self, benchmark_name: str):
        self.benchmark_name = benchmark_name
        logger.info(f"Initialized {benchmark_name} filter")
    
    @abstractmethod
    def filter_samples(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter samples using benchmark-specific rules.
        
        Args:
            samples: List of samples to filter
            
        Returns:
            Tuple of (passed_samples, dropped_samples)
        """
        pass
    
    @abstractmethod
    def get_filter_name(self) -> str:
        """Get the name of this filter."""
        pass
    
    def log_filtering_stats(self, total_samples: int, passed_samples: int, dropped_samples: int):
        """Log filtering statistics."""
        logger.info(f"=== {self.get_filter_name()} Filtering Results ===")
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Passed: {passed_samples} ({passed_samples/total_samples*100:.1f}%)")
        logger.info(f"Dropped: {dropped_samples} ({dropped_samples/total_samples*100:.1f}%)")
    
    def is_applicable(self, sample: Dict) -> bool:
        """
        Check if this filter is applicable to a given sample.
        Override in subclasses for benchmark-specific logic.
        """
        return True

