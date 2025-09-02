import json
import os
import sys
from typing import Dict, Any, List
from . import BaseLoader
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.types import AceBenchQuestion, Benchmark



class AceBenchLoader(BaseLoader):
    def load_questions(self) -> List[AceBenchQuestion]:
        """Load questions from the dataset"""
        raise NotImplementedError