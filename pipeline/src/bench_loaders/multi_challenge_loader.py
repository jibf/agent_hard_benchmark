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
        self.data_file = "data/multi_challenge.jsonl"
    
    def load_questions(self) -> List[MultiChallengeQuestion]:
        """Load all questions from MultiChallenge evaluation files."""
        all_questions = []
        
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"MultiChallenge data file not found: {self.data_file}")
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                data = json.loads(line)
                question = self._format_multi_challenge_question(data, line_num)
                if question:
                    all_questions.append(question)
    
        print(f"Loaded {len(all_questions)} MultiChallenge questions")
        return all_questions
    
    def _format_multi_challenge_question(self, data: Dict[str, Any], line_num: int) -> Optional[MultiChallengeQuestion]:
        """Format a MultiChallenge data entry into a MultiChallengeQuestion"""
        try:
            question_id = data.get('QUESTION_ID', f'multi_challenge_{line_num}')
            axis = data.get('AXIS', 'UNKNOWN')
            conversation = data.get('CONVERSATION', [])
            target_question = data.get('TARGET_QUESTION', '')
            pass_criteria = data.get('PASS_CRITERIA', 'YES')
            
            # The agent answers to the final user prompt
            assert conversation[-1].get('role') == 'user', "Last message in conversation should be from user"
            user_instruction = conversation[-1].get('content', '')

            evaluation_criteria = {
                'question': target_question,
                'pass_criteria': pass_criteria
            }
            
            return MultiChallengeQuestion(
                question_id=question_id,
                task_name=axis,
                instruction=user_instruction,
                gt_conv_traj=[],             # No ground-truth trajectory provided
                available_function_list=[],  # MultiChallenge doesn't use function calling
                benchmark=Benchmark.MULTI_CHALLENGE,
                evaluation_criteria=evaluation_criteria,
                original_conversation=conversation,
                meta={
                    'multi_challenge_context': {
                        'axis': axis,
                    }
                }
            )
            
        except Exception as e:
            print(f"Error formatting MultiChallenge question {question_id}: {e}")
            return None
    
