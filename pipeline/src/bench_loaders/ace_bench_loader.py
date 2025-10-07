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
                ground_truth = self._normalize_ground_truth(answer.get('ground_truth'))
                if isinstance(ground_truth, str):
                    assert "I cannot solve this problem" in ground_truth  # ground_truth is str only if the sample is erronous
                    ground_truth = [{"response": ground_truth}]
                return ground_truth
        return None

    def _normalize_ground_truth(self, raw_ground_truth: Any) -> Optional[List[Any]]:
        if raw_ground_truth is None:
            return None

        if isinstance(raw_ground_truth, dict):
            return self._expand_ground_truth_dict(raw_ground_truth)

        if isinstance(raw_ground_truth, list):
            normalized_list: List[Any] = []
            for item in raw_ground_truth:
                if isinstance(item, dict) and len(item) == 1:
                    normalized_list.extend(self._expand_ground_truth_dict(item))
                else:
                    normalized_list.append(item)
            return normalized_list

        return raw_ground_truth

    def _expand_ground_truth_dict(self, raw_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []

        for key, value in raw_dict.items():
            base_key = key
            if isinstance(key, str) and '_' in key:
                base, suffix = key.rsplit('_', 1)
                if suffix.isdigit():
                    base_key = base

            if isinstance(value, list):
                for item in value:
                    calls.append({base_key: item})
            else:
                calls.append({base_key: value})

        return calls

    
    def _get_system_prompts(
        self,
        question_id: str,
        question_text: str,
        functions: list,
        time_info: str,
        profile: str,
        involved_classes: Optional[List[str]],
        lang: str,
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Get system prompts based on data type and language.

        Returns a tuple of (agent_system_prompt, conversation_history_prompt, user_system_prompt).
        The user_system_prompt is only populated for agent-style categories that simulate a user LLM.
        """
        prompt_file = os.path.join(self.data_dir, "model_inference", f"prompt_{lang}.py")
        if not os.path.exists(prompt_file):
            return None, None, None
            
        sys.path.insert(0, os.path.dirname(prompt_file))
        
        # Extract category from question_id (similar to original implementation)
        category = question_id.rsplit("_", 1)[0] if question_id else ""

        user_system_prompt = None
        
        if lang == 'en':
            from prompt_en import (
                SYSTEM_PROMPT_FOR_NORMAL_DATA_EN,
                SYSTEM_PROMPT_FOR_PREFERENCE_DATA_EN,
                SYSTEM_PROMPT_FOR_SPECIAL_DATA_EN,
                USER_PROMPT_EN,
            )

            if category.startswith('agent'):
                agent_prompt = self._compose_agent_system_prompt_en(functions, involved_classes)
                user_prompt = self._compose_agent_user_prompt_en(question_text, functions)
                user_system_prompt = self._compose_agent_user_system_prompt_en(question_text, involved_classes)
            elif "special" in category:
                agent_prompt = SYSTEM_PROMPT_FOR_SPECIAL_DATA_EN.format(time=time_info, function=functions)
                user_prompt = USER_PROMPT_EN.format(question=question_text)
            elif "preference" in category:
                agent_prompt = SYSTEM_PROMPT_FOR_PREFERENCE_DATA_EN.format(profile=profile, function=functions)
                user_prompt = USER_PROMPT_EN.format(question=question_text)
            else:
                agent_prompt = SYSTEM_PROMPT_FOR_NORMAL_DATA_EN.format(time=time_info, function=functions)
                user_prompt = USER_PROMPT_EN.format(question=question_text)

        else:
            from prompt_zh import (
                SYSTEM_PROMPT_FOR_NORMAL_DATA_ZH,
                SYSTEM_PROMPT_FOR_PREFERENCE_DATA_ZH,
                SYSTEM_PROMPT_FOR_SPECIAL_DATA_ZH,
                USER_PROMPT_ZH,
            )

            if category.startswith('agent'):
                agent_prompt = self._compose_agent_system_prompt_zh(functions, involved_classes)
                user_prompt = self._compose_agent_user_prompt_zh(question_text, functions)
                user_system_prompt = self._compose_agent_user_system_prompt_zh(question_text, involved_classes)
            elif "special" in category:
                agent_prompt = SYSTEM_PROMPT_FOR_SPECIAL_DATA_ZH.format(time=time_info or "", function=functions)
                user_prompt = USER_PROMPT_ZH.format(question=question_text)
            elif "preference" in category:
                agent_prompt = SYSTEM_PROMPT_FOR_PREFERENCE_DATA_ZH.format(profile=profile or "", function=functions)
                user_prompt = USER_PROMPT_ZH.format(question=question_text)
            else:
                agent_prompt = SYSTEM_PROMPT_FOR_NORMAL_DATA_ZH.format(time=time_info or "", function=functions)
                user_prompt = USER_PROMPT_ZH.format(question=question_text)
        
        return agent_prompt, user_prompt, user_system_prompt

    def _compose_agent_system_prompt_en(self, functions: list, involved_classes: Optional[List[str]]) -> str:
        from multi_step.common_agent_step import MULTI_TURN_AGENT_PROMPT_SYSTEM_EN
        from prompt_en import BASE_PROMPT_EN, TRAVEL_PROMPT_EN

        prompt = MULTI_TURN_AGENT_PROMPT_SYSTEM_EN.strip()
        classes = involved_classes or []
        if any("Travel" in cls for cls in classes):
            prompt = f"{prompt}\n\n{TRAVEL_PROMPT_EN.strip()}"
        if any("BaseApi" in cls for cls in classes):
            prompt = f"{prompt}\n\n{BASE_PROMPT_EN.strip()}"
        return prompt

    def _compose_agent_user_prompt_en(self, question_text: str, functions: list) -> str:
        from multi_step.common_agent_step import MULTI_TURN_AGENT_PROMPT_USER_EN

        functions_str = json.dumps(functions, indent=2)
        history = question_text or ""
        return MULTI_TURN_AGENT_PROMPT_USER_EN.format(functions=functions_str, history=history)

    def _compose_agent_user_system_prompt_en(self, instruction: str, involved_classes: Optional[List[str]]) -> str:
        from multi_turn.APIModel_user import SYSTEM_PROMPT_BASE_EN, SYSTEM_PROMPT_TRAVEL_EN

        classes = involved_classes or []
        if any("Travel" in cls for cls in classes):
            template = SYSTEM_PROMPT_TRAVEL_EN
        else:
            template = SYSTEM_PROMPT_BASE_EN
        return template.format(instruction=instruction)

    def _compose_agent_system_prompt_zh(self, functions: list, involved_classes: Optional[List[str]]) -> str:
        from multi_step.common_agent_step import MULTI_TURN_AGENT_PROMPT_SYSTEM_ZH
        from prompt_zh import BASE_PROMPT_ZH, TRAVEL_PROMPT_ZH

        prompt = MULTI_TURN_AGENT_PROMPT_SYSTEM_ZH.strip()
        classes = involved_classes or []
        if any("Travel" in cls for cls in classes):
            prompt = f"{prompt}\n\n{TRAVEL_PROMPT_ZH.strip()}"
        if any("BaseApi" in cls for cls in classes):
            prompt = f"{prompt}\n\n{BASE_PROMPT_ZH.strip()}"
        return prompt

    def _compose_agent_user_prompt_zh(self, question_text: str, functions: list) -> str:
        from multi_step.common_agent_step import MULTI_TURN_AGENT_PROMPT_USER_ZH

        functions_str = json.dumps(functions, indent=2, ensure_ascii=False)
        history = question_text or ""
        return MULTI_TURN_AGENT_PROMPT_USER_ZH.format(functions=functions_str, history=history)

    def _compose_agent_user_system_prompt_zh(self, instruction: str, involved_classes: Optional[List[str]]) -> str:
        from multi_turn.APIModel_user import SYSTEM_PROMPT_BASE_ZH, SYSTEM_PROMPT_TRAVEL_ZH

        classes = involved_classes or []
        if any("Travel" in cls for cls in classes):
            template = SYSTEM_PROMPT_TRAVEL_ZH
        else:
            template = SYSTEM_PROMPT_BASE_ZH
        return template.format(instruction=instruction or "")

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

            category = question_id.rsplit("_", 1)[0] if question_id else ""
            skip_llm_judge = "special" in category.lower()

            # Get prompts for this specific question
            agent_prompt, previous_conversation_history, user_system_prompt = self._get_system_prompts(
                question_id,
                question_text,
                functions,
                time_info,
                profile,
                involved_classes,
                lang
            )
            
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
                user_system_prompt=user_system_prompt,
                previous_conversation_history=previous_conversation_history,
                skip_llm_judge=skip_llm_judge,
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
