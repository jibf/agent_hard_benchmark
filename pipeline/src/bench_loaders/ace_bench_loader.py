import json
import os
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add the src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.types import AceBenchQuestion, Benchmark
from src.bench_loaders.base_loader import BaseLoader


class AceBenchLoader(BaseLoader):
    """Loader for ACEBench benchmark data."""
    
    def __init__(self, data_dir: str = "data/ACEBench"):
        self.data_dir = data_dir
        
    def _get_ground_truth(self, question_id: str, task_name: str) -> List[Dict[str, Any]]:
        """Load ground truth from possible_answer files if available."""
        ground_truth_path = os.path.join(self.possible_answers_dir, task_name + '.json')
        ground_truths_within_task = []
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            for line in f:
                ground_truths_within_task.append(json.loads(line))
        
        for answer in ground_truths_within_task:
            if answer.get('id') == question_id:
                ground_truth = answer.get('ground_truth')
                if isinstance(ground_truth, dict):
                    ground_truth = [ground_truth]
                elif isinstance(ground_truth, str):
                    assert "I cannot solve this problem" in ground_truth  # ground_truth is str only if the sample is erronous
                    ground_truth = [{"error": ground_truth}]
                return ground_truth
        return None

    
    def _get_system_prompts(self, question_id: str, question_text: str, functions: list, time_info: str, profile: str, lang: str) -> tuple[Optional[str], Optional[str]]:
        """Get system prompts based on data type and language."""
        prompt_file = os.path.join(self.data_dir, "model_inference", f"prompt_{lang}.py")
        if not os.path.exists(prompt_file):
            return None, None
            
        sys.path.insert(0, os.path.dirname(prompt_file))
        
        # Extract category from question_id (similar to original implementation)
        category = question_id.rsplit("_", 1)[0] if question_id else ""
        
        if lang == 'en':
            from prompt_en import (SYSTEM_PROMPT_FOR_NORMAL_DATA_EN, SYSTEM_PROMPT_FOR_PREFERENCE_DATA_EN,
                                   SYSTEM_PROMPT_FOR_SPECIAL_DATA_EN, USER_PROMPT_EN)
            
            if "special" in category:
                agent_prompt = SYSTEM_PROMPT_FOR_SPECIAL_DATA_EN.format(time=time_info, function=functions)
            elif "preference" in category:
                agent_prompt = SYSTEM_PROMPT_FOR_PREFERENCE_DATA_EN.format(profile=profile, function=functions)
            else:
                agent_prompt = SYSTEM_PROMPT_FOR_NORMAL_DATA_EN.format(time=time_info, function=functions)
            
            user_prompt = USER_PROMPT_EN.format(question=question_text)
            
        else:
            from prompt_zh import (SYSTEM_PROMPT_FOR_NORMAL_DATA_ZH, SYSTEM_PROMPT_FOR_PREFERENCE_DATA_ZH,
                                   SYSTEM_PROMPT_FOR_SPECIAL_DATA_ZH, USER_PROMPT_ZH)
            
            if "special" in category:
                agent_prompt = SYSTEM_PROMPT_FOR_SPECIAL_DATA_ZH.format(time=time_info or "", function=functions)
            elif "preference" in category:
                agent_prompt = SYSTEM_PROMPT_FOR_PREFERENCE_DATA_ZH.format(profile=profile or "", function=functions)
            else:
                agent_prompt = SYSTEM_PROMPT_FOR_NORMAL_DATA_ZH.format(time=time_info or "", function=functions)
            
            user_prompt = USER_PROMPT_ZH.format(question=question_text)
        
        return agent_prompt, user_prompt
    
    def _load_questions_from_file(self, question_file_path: str, lang: str) -> List[AceBenchQuestion]:
        """Load questions from a single JSON file."""
        results = []
        task_file_name = Path(question_file_path).stem
        task_name = task_file_name.replace("data_", "").replace(".json", "")
        
        raw_questions = []
        with open(question_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                raw_questions.append(json.loads(line))
            
        # We'll get prompts per question since they depend on question-specific data
            
        for raw_question in raw_questions:
            question_id = raw_question.get('id', '')
            question_text = raw_question.get('question', '')
            functions = raw_question.get('function', [])
            time_info = raw_question.get('time', '')
            profile = raw_question.get('profile', '')  # Extract profile for preference data
            
            # Multi-turn specific fields
            initial_config = raw_question.get('initial_config')
            path = raw_question.get('path')
            involved_classes = raw_question.get('involved_classes')
            
            ground_truth = self._get_ground_truth(question_id, task_file_name)
            
            # Get prompts for this specific question
            agent_prompt, user_prompt = self._get_system_prompts(question_id, question_text, functions, time_info, profile, lang)
            
            # Create the question object
            question = AceBenchQuestion(
                benchmark=Benchmark.ACE_BENCH,
                task_name=task_name,
                question_id=question_id,
                instruction=question_text,
                available_function_list=functions,
                gt_conv_traj=ground_truth, 
                time=time_info if time_info else None,
                initial_config=initial_config,
                path=path,
                involved_classes=involved_classes,
                agent_system_prompt=agent_prompt,
                user_system_prompt=user_prompt,
                meta={
                    'data_type': task_file_name,
                    'file_path': question_file_path
                }
            )
            
            results.append(question)
        
        return results
    
    def load_questions(self) -> List[AceBenchQuestion]:
        """Load all questions from ACEBench evaluation files."""
        questions = []
        
        for lang in ['data_en']: # not using 'data_zh'
            lang_dir = os.path.join(self.data_dir, 'data_all', lang)
            self.possible_answers_dir = os.path.join(lang_dir, 'possible_answer')
            
            if not os.path.exists(lang_dir):
                continue
                
            # Load all JSON files in the language directory
            for file_name in os.listdir(lang_dir):
                if file_name.endswith('.json'):
                    file_path = os.path.join(lang_dir, file_name)
                    lang_code = lang.split('_')[1]  # 'data_en' -> 'en', 'data_zh' -> 'zh'
                    file_questions = self._load_questions_from_file(file_path, lang_code)
                    questions.extend(file_questions)
        
        return questions