from enum import Enum
from pydantic import BaseModel
from typing import Optional, Dict, List

class Benchmark(Enum):
    TAU_BENCH = "tau-bench"
    TAU2_BENCH = "tau2-bench"
    ACE_BENCH = "ACEBench"
    NEXUS_BENCH = "NexusBench"
    TOOL_SANDBOX = "ToolSandbox"
    COMPLEX_FUNC_BENCH = "complex-func-bench"
    DRAFTER_BENCH = "DrafterBench"
    BFCL = "BFCL"
    MULTI_CHALLENGE = "multi_challenge"


# Base class
class FormattedQuestion(BaseModel):
    benchmark: Benchmark                    # Benchmark that the question belongs to. example: Benchmark.TAU_BENCH
    question_id: str                        # A *unique* ID of the question. example: "Flight-83" (NOT just "83", to make this serve as a unique identifier within the benchmark!)
    instruction: str                        # The user's instruction (request) to the agent. example: "I want to rent a car for a self-driving trip starting tomorrow. Could you provide me with the ratings of the vehicle suppliers?"
                                            # For benchmarks that uses LLMs to simulate user (e.g., Tau-Bench), the user's instruction might be multi-turn would be different in every run. 
                                            # This case, use this field as a place for the prompt that initializes the user model. example: "Your user id is mia_li_3668. You want to fly from New York to Seattle on May 20..."
    available_function_list: list           # List of functions schemas that are available to the agent. This corresponds to the `tools` property in the OpenAI API endpoint. You can refer to https://platform.openai.com/docs/guides/function-calling.
    gt_conv_traj: list                      # Ground-truth conversation trajectory (if provided). If the benchmark does not provide one, leave this as an empty list.
    meta: Optional[dict] = None             # Other information of the question in dictionary type.



# TODO: Add fields so that all the necessary information for assessing each question is contained. Please try to keep variable names consistent to other benchmarks.

class ComplexFuncBenchQuestion(FormattedQuestion):  # CFB requires no additional field
    pass

class TauBenchQuestion(FormattedQuestion):
    agent_system_prompt: str                # the system prompt used to initialize the agent. e.g., "You are a specialized retail agent. Your task is to..."
    user_context: str                       # information of the user and order/reservation details


class Tau2BenchQuestion(FormattedQuestion):
    pass


class AceBenchQuestion(FormattedQuestion):
    task_name: str                              # The task name for ACEBench evaluation (e.g., "normal_weather", "special_math")
    benchmark_name: str                         # The benchmark name (e.g., "acebench")
    model_path: str                             # The model path used for evaluation
    sampling_params: dict                       # Sampling parameters for the model
    eval_result: dict                           # Evaluation result from ACEBench
    source_file: str                            # Source file where the question originated
    acebench_result: str                        # ACEBench specific result (can be complex data, stored as JSON string)
    is_correct: bool                            # Whether the model response was correct
    error_type: str                             # Type of error if any
    possible_answer: str                        # Possible answer for the question (can be complex data, stored as JSON string)
    finish_reason: str                          # Reason why the model finished
    turn_idx: int                               # Turn index in the conversation


class NexusBenchQuestion(FormattedQuestion):
    pass


class ToolSandboxQuestion(FormattedQuestion):
    pass





class DrafterBenchQuestion(FormattedQuestion):
    pass


class BfclV2Question(FormattedQuestion):
    pass


class BfclV3Question(FormattedQuestion):
    pass


class MultiChallengeQuestion(FormattedQuestion):
    pass


###

class LLMJudgeOutput(BaseModel):
    benchmark: Benchmark
    question_id: str
    # filtering
    is_flawed: bool
    error_category: Optional[str]
    reasoning: Optional[str]
    reasoning_summary: Optional[str]
    # scoring
    scores: Optional[Dict] = None
    # meta
    meta: Optional[Dict] = None