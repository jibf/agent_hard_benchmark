from typing import Dict, Any, List
from abc import ABC, abstractmethod
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.types import FormattedQuestion







class BaseLoader(ABC):

    @abstractmethod
    def load_questions(self) -> List[FormattedQuestion]:
        """Load questions from the dataset"""
        pass