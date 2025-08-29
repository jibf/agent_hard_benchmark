from enum import Enum
from pydantic import BaseModel

class BenchmarkType(Enum):
    COMPLEX_FUNC_BENCH = "complex_func_bench"
    TAU_BENCH = "tau_bench"

class FormattedQuestion(BaseModel):
    """Data model for a benchmark question."""
    conversations: list
    available_function_list: list
    meta: dict = {}