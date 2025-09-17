import json
import os
import sys
import re
from typing import Dict, Any, List, Tuple
from . import BaseLoader
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.types import ComplexFuncBenchQuestion, Benchmark



class ComplexFuncBenchLoader(BaseLoader):
    def __init__(self):
        self.data_path = "data/ComplexFuncBench.jsonl"
    
    def _parse_question_id(self, question_id: str) -> Tuple[str, str]:
        id_format = re.compile(r"([a-zA-z\-]+)-(\d+)")
        match = id_format.match(question_id)
        return match.group(1), match.group(2)

    def _format_line(self, line: dict) -> ComplexFuncBenchQuestion:
        question_id= line.get('id', 'unknown') # e.g., Cross-36
        task_name, question_id_num = self._parse_question_id(question_id)   # e.g., "Cross" and "36"
        
        conversations = line.get("conversations", [])
        user_prompt = conversations[0].get("content", "") if conversations else ""
        
        return ComplexFuncBenchQuestion(
            question_id=question_id_num,
            task_name=task_name,
            instruction=user_prompt,
            gt_conv_traj=conversations,    
            available_function_list=line.get("functions", []),
            benchmark=Benchmark.COMPLEX_FUNC_BENCH
        )

    def load_questions(self) -> List[ComplexFuncBenchQuestion]:
        questions = []
        with open(self.data_path, 'r') as f:
            for line in f:
                questions.append(self._format_line(json.loads(line)))
        return questions
