import json
import os
import sys
from typing import Dict, Any, List
from . import BaseLoader
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.types import ToolSandboxQuestion, Benchmark



class ToolSandBoxLoader(BaseLoader):
    def load_questions(self) -> List[ToolSandboxQuestion]:
        """Load questions from the dataset"""
        pass