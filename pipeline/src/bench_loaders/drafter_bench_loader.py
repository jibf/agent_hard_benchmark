import json
import os
import sys
from typing import Dict, Any, List
from . import BaseLoader
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.types import DrafterBenchQuestion, Benchmark
import re


class DrafterBenchLoader(BaseLoader):
    """Formatter for DrafterBench dataset"""
    
    def __init__(self, data_dir: str = "data/DrafterBench/drafter_tasks", system_prompt_dir: str = "data/DrafterBench/prompts"):
        self.data_dir = data_dir
        self.system_prompt_dir = system_prompt_dir
        self.system_prompts = dict()  # system prompt for each task type
        
        for prompt_filename in os.listdir(self.system_prompt_dir):
            if not prompt_filename.endswith(".txt"):
                continue
            with open(os.path.join(self.system_prompt_dir, prompt_filename), "r") as f:
                file_basename = re.match(r"([a-zA-z\_]+).txt", prompt_filename).group(1)
                self.system_prompts[file_basename] = f.read()
        
        assert len(self.system_prompts) == 12
        
    def load_questions(self) -> List[DrafterBenchQuestion]:
        """Load all DrafterBench questions from JSON files"""
        all_questions = []
        
        # Get all JSON files in the data directory
        json_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        assert set(self.system_prompts.keys()) == set([json_filename.replace(".json", "") for json_filename in json_files])
        
        for json_file in json_files:
            file_path = os.path.join(self.data_dir, json_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    tasks = json.load(f)
                
                for task in tasks:
                    formatted_question = self._format_sample(task)
                    all_questions.append(formatted_question)
                    
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        print(f"Loaded {len(all_questions)} DrafterBench questions")
        return all_questions

    def _format_sample(self, sample: Dict[str, Any]) -> DrafterBenchQuestion:
        """Format DrafterBench task to standard evaluation format"""
        
        # Extract task components
        task_type = sample.get('Tasktype', '')
        task_id = sample.get('Id', '')
        instruction = sample.get('Instruction', '').strip()
        
        # 

        # Construct conversations from ground-truth, which is a single string (code).
        groundtruth = sample.get('Groundtruth', '').strip()
        conversations = [
            {
                "role": "assistant",
                "content": groundtruth
            }
        ] 

        # TODO: No available function call for DrafterBench
        available_function_list = []
        
        return DrafterBenchQuestion(
            question_id=f"{task_type}-{task_id}",
            instruction=instruction,
            gt_conv_traj=conversations,
            available_function_list=available_function_list,
            benchmark=Benchmark.DRAFTER_BENCH,
            meta={
                'drafter_bench_context': {
                    'task_type': task_type,
                    'task_id': task_id,
                    'precise_vague': sample.get('Precise|Vague', ''),
                    'complete_incomplete': sample.get('Complete|Incomplete', ''),
                    'single_multiple_objects': sample.get('Single|Multiple_objects', ''),
                    'single_multiple_operations': sample.get('Single|Multiple_operations', ''),
                    'structured_unstructured': sample.get('Structured/Unstructured', ''),
                    'groundtruth': groundtruth,
                    'system_prompt': self.system_prompts[task_type]
                }
            }
        )