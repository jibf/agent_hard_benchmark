from typing import Dict, Any, List
from abc import ABC, abstractmethod
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.types import FormattedQuestion, UniqueQuestionID


class BaseLoader(ABC):

    @abstractmethod
    def load_questions(self) -> List[FormattedQuestion]:
        """Load questions from the dataset"""
        pass

    def load_responses_for_questions(self, questions: List[FormattedQuestion], responses_by_question: Dict[UniqueQuestionID, List[Dict]]):
        """add original contents of jsonl to questions, so each question can get all model's response"""
        for question in questions:
            question_uid = UniqueQuestionID(
                benchmark=question.benchmark,
                task_name=question.task_name,
                question_id=question.question_id
            )
            question.model_responses = responses_by_question[question_uid]