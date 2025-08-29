from enum import Enum
from pydantic import BaseModel
from typing import Optional

class Benchmark(Enum):
    COMPLEX_FUNC_BENCH = "complex_func_bench"
    TAU_BENCH = "tau_bench"

class FormattedQuestion(BaseModel):
    """Data model for a benchmark question."""
    benchmark: Benchmark
    question_id: str
    user_prompt: str
    conversations: list
    available_function_list: list
    meta: Optional[dict] = None 