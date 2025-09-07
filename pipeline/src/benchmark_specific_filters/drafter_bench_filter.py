"""
DrafterBench-specific rule-based filtering.
Implements custom filtering logic for DrafterBench evaluation data.
"""

from typing import Dict, List, Tuple
from .base_filter import BaseBenchmarkFilter
import json
import logging

logger = logging.getLogger(__name__)

class DrafterBenchFilter(BaseBenchmarkFilter):
    """DrafterBench-specific filtering rules."""
    
    def __init__(self):
        super().__init__("DrafterBench")
    
    def get_filter_name(self) -> str:
        return "DrafterBench-Specific Filter"
    
    def is_structure_applicable(self, sample: Dict) -> Tuple[bool, str]:
        """Check if sample is from DrafterBench."""
        # DrafterBench samples typically have specific structure

        if not (
            'task_name' in sample and 
            any(task_type in sample['task_name'] for task_type in [
                'add_', 'delete_', 'map_', 'refresh_', 'revise_'
            ])
        ):
            return False, "task name missing"

        if not sample.get('model_name', None):
            return False, "model name missing"

        score = sample["eval_result"].get("score", -1)
        if not (score >= 0 and score <= 1):
            return False, "invalid eval score"

        if 'messages' not in sample or not sample['messages']:
            return False, "meaasge missing"

        if 'id' not in sample["meta"]:
            return False, "id missing"

        if not all (key in sample["meta"] for key in ["precise_vague", "complete_incomplete", "single_multiple_objects", "single_multiple_operations", "structured_unstructured"]):
            return False, "question category missing"

        return True, "normal"
    
    def filter_samples(self, samples: Dict) -> Tuple[List[Dict], List[Dict]]:
        """
        Apply DrafterBench-specific filtering rules.
        
        DrafterBench-specific rules:
        1. Must have valid task_name with operation type
        2. Score must be between 0-1 (DrafterBench scale)
        3. Must have sufficient model responses for comparison
        4. Task must show meaningful performance variation
        """
        logger.info(f"Applying DrafterBench-specific filtering to {len(samples)} samples")

        filtered_samples = self._filter_by_structure(samples)
        filtered_samples = self._filter_duplicate_id(filtered_samples)

        logger.info("========================= Question Level Filtering =========================")
        format_samples = self._format_samples(filtered_samples)
        num_questions = len(format_samples)
        format_samples = self._filter_duplicate_question(format_samples)
        format_samples = self._filter_missing_results_questions(format_samples)
        format_samples = self._filter_by_prompt(format_samples)

        self.log_filtering_stats(num_questions, len(format_samples), num_questions - len(format_samples))

        passed_samples = self._unformat_samples(format_samples)
        dropped_samples = [] # placeholder, duplicated cases over 50% for now.
        
        return passed_samples, dropped_samples
    
    def _filter_by_structure(self, samples: List[Dict]) -> List[Dict]:
        """Filter by DrafterBench-specific structure requirements."""
        valid_samples = []
        
        for sample in samples:
            if not self.is_structure_applicable(sample):
                continue
                
            valid_samples.append(sample)
        
        logger.info(f"Filter invalid structures: {len(samples) - len(valid_samples)}/{len(samples)} filtered")

        return valid_samples

    def _filter_duplicate_id(self, samples: List[Dict]) -> List[Dict]:
        known_question_ids = set()
        valid_samples = []
        for sample in samples:
            question_id = sample["model_name"] + "_" + sample["task_name"] + "_" + sample["meta"]["id"]
            if question_id not in known_question_ids:
                known_question_ids.add(question_id)
                valid_samples.append(sample)

        logger.info(f"Filter duplicate id: {len(samples) - len(valid_samples)}/{len(samples)} filtered")
        return valid_samples

    def _format_samples(self, samples: List[Dict]) -> Dict[str, Dict]:
        
        sample_dict = {}
        for sample in samples:
            question_id = sample["task_name"] + "_" + sample["meta"]["id"]
            if question_id not in sample_dict:
                sample_dict[question_id] = []
            
            sample_dict[question_id].append(sample)
        
        logger.info(f"Find {len(sample_dict)} unique questions in total.")

        return sample_dict
    
    def _unformat_samples(self, passed_format_samples: Dict[str, Dict]) -> List[Dict]:
        passed_samples = []
        for sample in passed_format_samples.values():
            passed_samples.extend(sample)
        
        return passed_samples


    def _filter_duplicate_question(self, question_dict: Dict[str, Dict]) -> Dict[str, Dict]:
        known_questions = set()
        valid_samples = {}
        for question_id, sample_list in question_dict.items():
            task_type = question_id.rsplit("_", 1)[0]
            question = task_type + "_" + sample_list[0]["messages"][0]["content"].strip()
            if question not in known_questions:
                known_questions.add(question)
                valid_samples[question_id] = sample_list
        
        logger.info(f"Filter duplicate questions: {len(question_dict) - len(valid_samples)}/{len(question_dict)} filtered")
        return valid_samples

    def _filter_missing_results_questions(self, question_dict: Dict[str, Dict]) -> Dict[str, Dict]:

        num_model_list = [len(question_dict[question]) for question in question_dict]
        num_model = max(num_model_list)

        valid_questions = {}
        for question_id in question_dict:
            if len(question_dict[question_id]) == num_model:
                valid_questions[question_id] = question_dict[question_id]

        logger.info(f"Filter questions missing results: {len(question_dict) - len(valid_questions)}/{len(question_dict)} filtered")

        return valid_questions

    def _filter_by_prompt(self, question_dict: Dict[str, Dict]) -> Dict[str, Dict]:
        valid_questions = {}
        reason_count = {
            "vague_question": 0,
            "low_discriminativeness": 0
        }
        for question, question_list in question_dict.items():
            is_valid = True

            # filter all vague questions
            if question_list[0]["meta"]["precise_vague"] == "Vague":
                is_valid = False
                reason_count["vague_question"] += 1
                continue

            scores = [sample["eval_result"]["score"] for sample in question_list]
            if len(scores) > 2:
                variance = sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores)
                if variance < 0.01:
                    is_valid = False
                    reason_count["low_discriminativeness"] += 1
                    continue

            if is_valid:
                valid_questions[question] = question_list

        logger.info(f"Filter questions by rules: \n{json.dumps(reason_count, indent=2)}\n Total: {sum(reason_count.values())}/{len(question_dict)}")
        return valid_questions


