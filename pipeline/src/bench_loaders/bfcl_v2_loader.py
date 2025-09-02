import json
import os
import sys
from typing import Dict, Any, List
from . import BaseLoader
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.types import BfclV2Question, Benchmark



class BfclV2Loader(BaseLoader):
    def load_questions(self) -> List[BfclV2Question]:
        """Load questions from the dataset"""
        raise NotImplementedError