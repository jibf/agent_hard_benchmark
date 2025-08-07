from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Conversation:
    question_id: int              # Unique ID for tracking conversations
    axis: str                     # Axis for evaluation (REFINEMENT, EXPLICIT IF, COHERENCE, RECOLLECTION)
    conversation: List[Dict]      # messages alternating between user and assistant
    target_question: str          # The key question being evaluated by the judge
    pass_criteria: str            # Criteria for passing this conversation
