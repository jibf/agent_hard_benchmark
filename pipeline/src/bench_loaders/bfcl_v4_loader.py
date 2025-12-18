import copy
import json
import re
import os
import sys
from typing import Any, Dict, List, Optional
from collections import defaultdict
from functools import lru_cache


from . import BaseLoader
from src.utils.types import BFCLv4Question, Benchmark
from src.utils.bfcl_v4 import normalize_bfcl_v4_task_info


class BfclV4Loader(BaseLoader):
    def __init__(self):
        self.data_path = "data/BFCLv4/"
        self.func_doc_path = "data/BFCLv4/multi_turn_func_doc/"
        self.possible_answer_path = "data/BFCLv4/possible_answer/"
        self.memory_prereq_conv_path = "data/BFCLv4/memory_prereq_conversation"

    def load_questions(self) -> List[BFCLv4Question]:
        questions = defaultdict()
        answers = defaultdict() 

        # Add paths of domains newly added in BFCLv4
        relevant_domains = {"web_search", "memory"}
        formatted_questions: List[BFCLv4Question] = []

        for path in os.listdir(self.data_path):
            domain = self.extract_domain_from_file_name(path)
            if domain not in relevant_domains:
                continue

            data_path = os.path.join(self.data_path, path)
            answer_path = os.path.join(self.possible_answer_path, path)

            with open(data_path, "r") as f:
                for line in f:
                    sample = json.loads(line)
                    questions[sample["id"]] = sample

            with open(answer_path, "r") as f:
                for line in f:
                    sample = json.loads(line)
                    answers[sample["id"]] = sample

        for question_id, question in questions.items():
            answer = answers.get(question_id)
            if not answer:
                continue

            formatted_question = self.format_question_sample(question, answer)
            formatted_questions.append(formatted_question)

        # formatted_questions.extend(self._load_format_sensitivity_questions()) # TODO for my self, not codex

        return formatted_questions

    def _resolve_domain(self, question: dict) -> str:
        domains = ["web_search", "memory"] #, "format_sensitivity"]
        for domain in domains:
            if question["id"].startswith(domain):
                return domain
        raise ValueError

    

    def format_question_sample(self, question: dict, answer: dict) -> BFCLv4Question:
        raw_question_id = question["id"]
        domain = self._resolve_domain(question)
        task_name, question_id = normalize_bfcl_v4_task_info(domain, raw_question_id)
    
        if domain == "memory":
            return self._format_memory_question_sample(question, answer, question_id, raw_question_id, task_name)
        elif domain == "web_search":
            return self._format_web_search_question_sample(question, answer, question_id, raw_question_id, task_name)
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


    def _format_memory_question_sample(self, question: dict, answer: dict, question_id: str, raw_question_id: str, task_name: str) -> BFCLv4Question:
        domain = self._resolve_domain(question)
        instruction = question["question"][0][0]["content"] # first user message
        scenario = question["scenario"]
        ground_truth = answer["ground_truth"]
        sources = answer["source"]  # web search sources for each hop
        memory_context = self._get_memory_for_scenario(scenario)

        return BFCLv4Question(
            benchmark=Benchmark.BFCL_V4,
            question_id=question_id,
            task_name=task_name,
            instruction=instruction,
            ground_truth=ground_truth,
            gt_conv_traj=[],
            sources=[sources],
            memory_context=memory_context,
            meta={"scenario": scenario, "original_id": raw_question_id},
            available_function_list=[]
        )


    def _format_web_search_question_sample(self, question: dict, answer: dict, question_id: str, raw_question_id: str, task_name: str) -> BFCLv4Question:
        domain = self._resolve_domain(question)
        instruction = question["question"][0][0]["content"] # first user message
        ground_truth = answer["ground_truth"]
        sources = answer["source"]  # web search sources for each hop
        num_hops = answer.get("num_hops")

        return BFCLv4Question(
            benchmark=Benchmark.BFCL_V4,
            question_id=question_id,
            task_name=task_name,
            instruction=instruction,
            ground_truth=ground_truth,
            gt_conv_traj=[],
            sources=sources,
            available_function_list=[],
            meta={"original_id": raw_question_id}
        )

    def _extract_format_config(self, question_id: str) -> str:
        parts = question_id.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid format sensitivity question id: {question_id}")
        return parts[1]

    def _extract_base_question_id(self, question_id: str) -> str:
        parts = question_id.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid format sensitivity question id: {question_id}")
        return parts[2]

    def _extract_first_user_message(self, conversation: List[List[Dict]]) -> str:
        for turn in conversation or []:
            for message in turn:
                if isinstance(message, dict) and message.get("role") == "user":
                    content = message.get("content", "")
                    if content:
                        return content
        if conversation and conversation[0]:
            return conversation[0][0].get("content", "")
        return ""


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
