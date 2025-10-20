import copy
import glob
import json
import re
import os
from typing import Any, Dict, List, Optional
from collections import defaultdict
from . import BaseLoader
from src.utils.types import BFCLv4Question, Benchmark, FormattedQuestion
from functools import lru_cache

class BfclV4Loader(BaseLoader):
    """
    Berkeley Function Calling Leaderboard (BFCL) data loader
    
    Supports comprehensive BFCL evaluation including:
    - Single-turn and multi-turn function calling
    - Language-specific processing (Python, Java, JavaScript)
    - Model-specific input formatting (FC vs prompting)
    - Multi-turn state management and missing function handling
    """
    def __init__(self):
        self.data_path = "data/BFCLv4/"
        self.func_doc_path = "data/BFCLv4/multi_turn_func_doc/"
        self.possible_answer_path = "data/BFCLv4/possible_answer/"
        self.memory_prereq_conv_path = "data/BFCLv4/memory_prereq_conversation"

    def load_questions(self) -> List[BFCLv4Question]:
        questions = defaultdict()
        answers = defaultdict() 

        # Add paths of domains newly added in BFCLv4
        relevant_file_paths = []
        for path in os.listdir(self.data_path):
            domain = self.extract_domain_from_file_name(path)
            if domain not in ["web_search", "memory"]:  # TODO: add format_sensitivity
                continue
            with open(os.path.join(self.data_path, path), "r") as f:
                for line in f:
                    sample = json.loads(line)
                    questions[sample['id']] = sample
            with open(os.path.join(self.possible_answer_path, path), "r") as f:
                for line in f:
                    sample = json.loads(line)
                    answers[sample['id']] = sample
        
        for question_id in questions:
            formatted_question = self.format_question_sample(questions[question_id], answers[question_id])


            
        return []

    def _resolve_domain(self, question: dict) -> str:
        domains = ["web_search", "memory", "format_sensitivity"]
        for domain in domains:
            if question["id"].startswith(domain):
                return domain
        raise ValueError

    

    def format_question_sample(self, question: dict, answer: dict) -> BFCLv4Question:
        question_id = question["id"]
        domain = self._resolve_domain(question)
    
        if domain == "memory":
            return self._format_memory_question_sample(question, answer)
        elif domain == "web_search":
            return self._format_web_search_question_sample(question, answer)
        elif domain == "format_sensitivity":
            return self._format_format_sensitivity_question_sample(question, answer)
        else:
            raise ValueError(f"Domain {domain} is not contained in BFCLv4")
    
    @lru_cache(maxsize=1)
    def _get_memory_for_scenario(self, scenario: str) -> List[Dict]:
        memory_list = []
        scenario_memory_path = os.path.join(self.memory_prereq_conv_path, f"memory_{scenario}.json")
        with open(scenario_memory_path, "r") as f:
            for line in f:
                memory_line = json.loads(line)
                del memory_line["id"], memory_line["involved_classes"], memory_line["scenario"]
                memory_list.append(memory_line)
        
        return memory_list


    def _format_memory_question_sample(self, question: dict, answer: dict) -> BFCLv4Question:
        question_id = question["id"]
        domain = self._resolve_domain(question)
        instruction = question["question"][0][0]["content"] # first user message
        scenario = question["scenario"]
        ground_truth = answer["ground_truth"]
        sources = answer["source"]  # web search sources for each hop
        memory_context = self._get_memory_for_scenario(scenario)

        return BFCLv4Question(
            benchmark=Benchmark.BFCL_V4,
            question_id=question_id,
            task_name=domain,
            instruction=instruction,
            gt_conv_traj=ground_truth,
            sources=sources,
            memory_context=memory_context,
            available_function_list=[]
        )


    def _format_web_search_question_sample(self, question: dict, answer: dict) -> BFCLv4Question:
        question_id = question["id"]
        domain = self._resolve_domain(question)
        instruction = question["question"][0][0]["content"] # first user message
        ground_truth = answer["ground_truth"]
        sources = answer["source"]  # web search sources for each hop

        return BFCLv4Question(
            benchmark=Benchmark.BFCL_V4,
            question_id=question_id,
            task_name=domain,
            instruction=instruction,
            gt_conv_traj=ground_truth,
            sources=sources,
            available_function_list=[]
        )


    def _format_format_sensitivity_question_sample(self, question: dict, answer: dict) -> BFCLv4Question:
        question_id = question["id"]
        domain = self._resolve_domain(question)
        instruction = question["question"][0][0]["content"] # first user message


    def extract_domain_from_file_name(self, file_name: str) -> Optional[str]:
        file_pattern = re.compile(r"BFCL_v4_([a-z_]+).json")
        match = file_pattern.match(file_name)
        if match is None:
            return None
        domain = match.group(1)
        return domain


if __name__=="__main__":
    loader = BfclV4Loader()
    questions = loader.load_questions()