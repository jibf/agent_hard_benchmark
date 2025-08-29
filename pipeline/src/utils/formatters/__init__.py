from typing import Dict, Any
from abc import ABC, abstractmethod
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from benchmark_types import FormattedQuestion


class BaseFormatter(ABC):
    """Base class for dataset formatters"""
    
    @abstractmethod
    def format_sample(self, sample: Dict[str, Any], **kwargs) -> FormattedQuestion:
        """Format a single sample to FormattedQuestion format"""
        pass
    
    @abstractmethod
    def extract_conversation(self, formatted_sample: FormattedQuestion) -> tuple:
        """Extract user_prompt, conversations, and function_list from formatted sample"""
        pass