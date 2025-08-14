import json
import copy
from utils.utils import *
from models.gpt import GPTModel

from prompts.response import (
    complete_system_prompt, 
    complete_user_prompt, 
    correct_system_prompt, 
    correct_user_prompt
)

class RespEvalRunner:
    def __init__(self, args, logger):
        self.logger = logger
        self.model = GPTModel("gpt-4o-2024-08-06")

    @retry(max_attempts=10)
    def completeness_eval(self, **kwargs):
        complete_result = self.model(complete_system_prompt, complete_user_prompt, **kwargs)
        decoded_complete_result = decode_json(complete_result)

        self.logger.info(f"Complete Result: {decoded_complete_result}")

        if not isinstance(decoded_complete_result, dict) or "score" not in decoded_complete_result:
            return None
        if decoded_complete_result['score'] not in [0, 1, 2]:
            return None
        return decoded_complete_result

    @retry(max_attempts=10)
    def correctness_eval(self, **kwargs):
        correct_result = self.model(correct_system_prompt, correct_user_prompt, **kwargs)
        decoded_correct_result = decode_json(correct_result)
        self.logger.info(f"Correct Result: {decoded_correct_result}")
        if not isinstance(decoded_correct_result, dict) or "score" not in decoded_correct_result:
            return None
        if decoded_correct_result['score'] not in [0, 1, 2]:
            return None
        return decoded_correct_result

    def run(self, data, gen_response):
        if gen_response == "":
            return {
                "complete": {"score": -2, "reason": "Do not generate response successfully."}, 
                "correct": {"score": -2, "reason": "Do not generate response successfully."}
            }
        
        convs = data['conversations']

        kwargs = {
            "query": convs[0]['content'],
            "gen_response": gen_response,
        }

        complete_result = self.completeness_eval(**kwargs)

        kwargs = {
            "history": json.dumps(convs[:-1], ensure_ascii=False),
            "gen_response": gen_response
        }

        correct_result = self.correctness_eval(**kwargs)

        if complete_result and correct_result:
            return {
                "complete": {"score": complete_result['score'], "reason": complete_result.get("reason", None)}, 
                "correct": {"score": correct_result['score'], "reason": correct_result.get("reason", None)}
            }
        elif complete_result:
            return {
                "complete": {"score": complete_result['score'], "reason": complete_result.get("reason", None)}, 
                "correct": {"score": -1, "reason": "Correctness Eval failed."}
            } 
        elif correct_result:
            return {
                "complete": {"score": -1, "reason": "Completeness Eval failed."},
                "correct": {"score": correct_result['score'], "reason": correct_result.get("reason", None)}
            }
        else:
            return {
                "complete": {"score": -1, "reason": "Completeness Eval failed."},
                "correct": {"score": -1, "reason": "Correctness Eval failed."}
            }