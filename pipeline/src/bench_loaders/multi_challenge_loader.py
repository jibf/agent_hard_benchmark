import json
import os
import sys
from typing import Dict, Any, List, Optional

# Add the src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.types import MultiChallengeQuestion, Benchmark
from .base_loader import BaseLoader


class MultiChallengeLoader(BaseLoader):
    """Loader for MultiChallenge benchmark data."""
    
    def __init__(self):
        self.benchmark_name = "multi_challenge"
        self.evaluation_dir = "benchmark/multi_challenge-evaluation"
    
    def load_questions(self) -> List[MultiChallengeQuestion]:
        """Load all questions from MultiChallenge evaluation files."""
        questions = []
        
        if not os.path.exists(self.evaluation_dir):
            print(f"Warning: MultiChallenge evaluation directory {self.evaluation_dir} not found")
            return questions
        
        # Look for .jsonl files in the evaluation directory
        for filename in os.listdir(self.evaluation_dir):
            if filename.endswith('.jsonl'):
                file_path = os.path.join(self.evaluation_dir, filename)
                file_questions = self._load_questions_from_file(file_path)
                questions.extend(file_questions)
        
        print(f"Loaded {len(questions)} questions from MultiChallenge")
        return questions
    
    def _load_questions_from_file(self, file_path: str) -> List[MultiChallengeQuestion]:
        """Load questions from a single .jsonl file."""
        questions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        question = self._format_sample(sample, line_num)
                        if question:
                            questions.append(question)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON at line {line_num}: {e}")
                        continue
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
        
        return questions
    
    def _format_sample(self, sample: Dict[str, Any], line_num: int) -> MultiChallengeQuestion:
        """Format a raw sample into a MultiChallengeQuestion."""
        try:
            question_id = sample.get('meta', {}).get('id', f"multichallenge_{line_num}")
            benchmark_name = sample.get('benchmark_name', 'multi_challenge')
            task_name = sample.get('task_name', 'unknown')
            axis = sample.get('meta', {}).get('axis', task_name)
            
            instruction = self._extract_instruction(sample)
            available_function_list = self._extract_functions(sample)
            gt_conv_traj = self._extract_ground_truth(sample)
            
            meta = {
                'multi_challenge_context': {
                    'benchmark_name': benchmark_name,
                    'task_name': task_name,
                    'axis': axis,
                    'model_path': sample.get('model_path', ''),
                    'sampling_params': sample.get('sampling_params', {}),
                    'eval_result': sample.get('eval_result', {}),
                    'source_file': sample.get('meta', {}).get('source_file', ''),
                    'multi_challenge_result': sample.get('meta', {}).get('multi_challenge_result', ''),
                    'is_correct': sample.get('meta', {}).get('is_correct', False),
                    'error_type': sample.get('meta', {}).get('error_type', ''),
                    'possible_answer': sample.get('meta', {}).get('possible_answer', ''),
                    'target_question': sample.get('meta', {}).get('target_question', ''),
                    'pass_criteria': sample.get('meta', {}).get('pass_criteria', ''),
                    'judge_verdict': sample.get('meta', {}).get('judge_verdict', ''),
                    'passed': sample.get('meta', {}).get('passed', ''),
                    'reasoning': sample.get('meta', {}).get('reasoning', ''),
                    'final_result': sample.get('meta', {}).get('final_result', ''),
                    'original_conversation': sample.get('meta', {}).get('original_conversation', '')
                }
            }
            
            return MultiChallengeQuestion(
                question_id=question_id,
                instruction=instruction,
                available_function_list=available_function_list,
                gt_conv_traj=gt_conv_traj,
                benchmark=Benchmark.MULTI_CHALLENGE,
                meta=meta
            )
        except Exception as e:
            print(f"Error formatting sample {line_num}: {e}")
            return None
    
    def _extract_instruction(self, sample: Dict[str, Any]) -> str:
        """Extract instruction from sample messages."""
        messages = sample.get('messages', [])
        if messages:
            user_messages = [msg for msg in messages if msg.get('role') == 'user']
            if user_messages:
                instruction_parts = []
                for msg in user_messages:
                    content = msg.get('content', '').strip()
                    if content:
                        instruction_parts.append(content)
                if instruction_parts:
                    return '\n\n'.join(instruction_parts)
        
        task_name = sample.get('task_name', 'unknown')
        axis = sample.get('meta', {}).get('axis', task_name)
        return f"Complete the MultiChallenge task in the {axis} category: {task_name}"
    
    def _extract_functions(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract available functions from sample."""
        # MultiChallenge doesn't have explicit function definitions
        # Return empty list as this is conversation-based
        return []
    
    def _extract_ground_truth(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract ground truth conversation trajectory."""
        messages = sample.get('messages', [])
        if messages:
            conversation = []
            for msg in messages:
                if msg.get('role') == 'user':
                    conversation.append({
                        'role': 'user',
                        'content': msg.get('content', '')
                    })
                elif msg.get('role') == 'assistant':
                    conversation.append({
                        'role': 'assistant', 
                        'content': msg.get('content', '')
                    })
            return conversation
        return []