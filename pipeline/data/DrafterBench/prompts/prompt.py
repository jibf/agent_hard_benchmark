import sys
import os

sys.path.append(os.path.abspath("../.."))

from DrafterBench.prompts import Backend_prompt


class Prompt:
    def __init__(
        self,
        task_index: str,
        user_instruction: str,
    ):
        self.system = Backend_prompt[task_index]
        self.user_instruction = user_instruction

    def message(self):
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user_instruction},
        ]
