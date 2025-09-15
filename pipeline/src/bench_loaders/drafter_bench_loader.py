import json
import os
import sys
from typing import Dict, Any, List
from . import BaseLoader
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.types import (
    DrafterBenchQuestion,
    Benchmark,
    FormattedQuestion,
    UniqueQuestionID
)
import re


class DrafterBenchLoader(BaseLoader):
    """Formatter for DrafterBench dataset"""
    
    def __init__(self, data_dir: str = "data/DrafterBench/drafter_tasks", system_prompt_dir: str = "data/DrafterBench/prompts"):
        self.data_dir = data_dir
        self.system_prompt_dir = system_prompt_dir
        self.system_prompts = dict()  # system prompt for each task type
        self._load_system_prompts()
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
    
    def _load_system_prompts(self) -> None:
        for prompt_filename in os.listdir(self.system_prompt_dir):
            if not prompt_filename.endswith(".txt"):
                continue
            file_path = os.path.join(self.system_prompt_dir, prompt_filename)
            with open(file_path, "r", encoding="utf-8") as f:
                file_basename = re.match(r"([a-zA-Z_]+)\.txt", prompt_filename).group(1)
                self.system_prompts[file_basename] = f.read()
        
    def _format_sample(self, sample: Dict[str, Any]) -> DrafterBenchQuestion:
        """Format DrafterBench task to standard evaluation format"""
        task_type = sample.get('Tasktype', '')
        task_id = sample.get('Id', '')
        instruction = sample.get('Instruction', '').strip()

        # Construct conversations from ground-truth, which is a single string (code).
        groundtruth = sample.get('Groundtruth', '').strip()
        conversations = [
            { "role": "user", "content": instruction },
            { "role": "assistant", "content": groundtruth }
        ] 

        return DrafterBenchQuestion(
            question_id=f"{task_type}-{task_id}",
            task_name=task_type,
            instruction=instruction,
            gt_conv_traj=conversations,
            available_function_list=[],
            benchmark=Benchmark.DRAFTER_BENCH,
            agent_system_prompt=self.system_prompts[task_type],
            groundtruth=groundtruth,
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
                }
            }
        )

    def load_responses_for_questions(self, questions: List[FormattedQuestion], responses_by_question: Dict[UniqueQuestionID, List[Dict]]):
        """add original contents of jsonl to questions, so each question can get all model's response"""
        for question in questions:
            question_uid = UniqueQuestionID(
                benchmark=question.benchmark,
                task_name=question.task_name,
                question_id=question.question_id
            )
            question.model_responses = responses_by_question[question_uid]

            # use system_prompt saved in jsonl files instead of read it online
            for message in responses_by_question[question_uid][0]["messages"]:
                if message["role"] == "system":
                    agent_system_prompt = message["content"]
                    break
            question.agent_system_prompt = agent_system_prompt