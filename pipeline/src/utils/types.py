from enum import Enum
import re
import json
from typing import Optional, Dict, List, Any, Union

try:
    from pydantic import BaseModel
except ModuleNotFoundError:
    class BaseModel:  # pragma: no cover - lightweight fallback
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

        def dict(self) -> Dict[str, Any]:
            return self.__dict__

class Benchmark(Enum):
    TAU_BENCH = "tau-bench"
    TAU2_BENCH = "tau2-bench"
    ACE_BENCH = "ACEBench"
    NEXUS_BENCH = "NexusBench"
    TOOL_SANDBOX = "ToolSandbox"
    COMPLEX_FUNC_BENCH = "complex-func-bench"
    DRAFTER_BENCH = "DrafterBench"
    BFCL = "BFCL"
    BFCL_V4 = "BFCLv4"
    MULTI_CHALLENGE = "multi_challenge"
    
    def __str__(self):
        return self.value


class UniqueQuestionID(BaseModel):
    benchmark: Benchmark
    task_name: Optional[str] = None
    question_id: str
    
    def _normalize_question_id(self, question_id: str) -> str:
        if not self.task_name:
            return question_id
        match = re.match(f'^{re.escape(self.task_name)}[-_](\\d+)$', question_id)
        return match.group(1) if match else question_id
    
    def __hash__(self):
        return hash((self.benchmark, self.task_name, self._normalize_question_id(self.question_id)))
    
    def __eq__(self, other):
        return (
            self.benchmark == other.benchmark and 
            self.task_name == other.task_name and 
            self._normalize_question_id(self.question_id) == other._normalize_question_id(other.question_id)
        )


# Base class
class FormattedQuestion(UniqueQuestionID):
    instruction: str                        # The user's instruction (request) to the agent. example: "I want to rent a car for a self-driving trip starting tomorrow. Could you provide me with the ratings of the vehicle suppliers?"
                                            # For benchmarks that uses LLMs to simulate user (e.g., Tau-Bench), the user's instruction might be multi-turn would be different in every run. 
                                            # This case, use this field as a place for the prompt that initializes the user model. example: "Your user id is mia_li_3668. You want to fly from New York to Seattle on May 20..."
    available_function_list: list           # List of functions schemas that are available to the agent. This corresponds to the `tools` property in the OpenAI API endpoint. You can refer to https://platform.openai.com/docs/guides/function-calling.
    gt_conv_traj: list                      # Ground-truth conversation trajectory (if provided). If the benchmark does not provide one, leave this as an empty list.
    meta: Optional[dict] = None             # Other information of the question in dictionary type.
    model_responses: Optional[list] = None
    skip_llm_judge: bool = False            # When True, bypass LLM-as-judge evaluation and auto-pass filters.



# TODO: Add fields so that all the necessary information for assessing each question is contained. Please try to keep variable names consistent to other benchmarks.

class ComplexFuncBenchQuestion(FormattedQuestion):  # CFB requires no additional field
    pass

class TauBenchQuestion(FormattedQuestion):
    agent_system_prompt: str                # the system prompt used to initialize the agent. e.g., "You are a specialized retail agent. Your task is to..."
    user_context: str                       # information of the user and order/reservation details


class Tau2BenchQuestion(FormattedQuestion):
    agent_system_prompt: str
    user_context: str
    available_user_function_list: list      # available functions that the user model can call (if any). This is only for telecom domain.
    initial_state: Optional[Dict[str, Any]] = None  # Initial state setup for the test environment
    evaluation_criteria: Optional[Dict[str, Any]] = None  # Complete evaluation criteria including nl_assertions and env_assertions 


class AceBenchQuestion(FormattedQuestion):
    time: Optional[str] = None                       # Time context provided with the question
    initial_config: Optional[Dict[str, Any]] = None  # For multi-turn scenarios, initial device state
    path: Optional[List] = None                      # For multi-turn scenarios, execution path
    involved_classes: Optional[List[str]] = None     # Classes involved in multi-turn scenarios
    agent_system_prompt: Optional[str] = None        # System prompt for the assistant agent
    user_system_prompt: Optional[str] = None         # System prompt for user simulation (if applicable)
    previous_conversation_history: str = None        # Conversation history between the user and the agent in natural language
    
class NexusBenchQuestion(FormattedQuestion):
    subbench_name: str
    subbench_description: Optional[str] = None
    reference: str
    system_prompts: Optional[Any] = None  # Captured system prompts/instructions sent to model
    tool_definitions: Optional[str] = None  # JSON of tool schemas for prompt rendering
    tool_implementations: Optional[str] = None  # actual implementation of tools (only used in LangChainMath)


class ToolSandboxQuestion(FormattedQuestion):
    user_system_prompt: str
    starting_messages: Optional[List[Dict[str, Any]]] = None
    initial_databases: Optional[Dict[str, List[Dict[str, Any]]]] = None
    milestones: Optional[List[Dict[str, Any]]] = None
    minefields: Optional[List[Dict[str, Any]]] = None

class DrafterBenchQuestion(FormattedQuestion):
    agent_system_prompt: str    # system prompt for the agent
    groundtruth: str

class BFCLQuestion(FormattedQuestion):
    initial_config: Optional[Dict] = None       # Initial state configuration for multi-turn scenarios
    initial_pwd_description: Optional[str] = None       # Initial state configuration for multi-turn scenarios
    expected_path: Optional[List[str]] = None   # Expected function call path for multi-turn scenarios
    involved_classes: Optional[List[str]] = None # Classes involved in multi-turn scenarios
    is_multi_turn: bool = False                 # Whether this question is multi-turn
    ground_truth: Optional[Any] = None          # Expected function call answers (can be various formats)
    category: Optional[str] = None              # Category (simple, multiple, parallel, etc.)
    subcategory: Optional[str] = None           # Subcategory for more detailed classification
    is_live_data: bool = False                  # Whether this is live data with different format
    
    # BFCL evaluation specific fields
    missed_function: Optional[str] = None # Functions that should be missing for specific turns
    system_prompt: Optional[str] = None         # System prompt for this question type
    max_turn_limit: int = 20                    # Maximum number of turns allowed
    exclude_state_log: bool = False            # Whether to exclude state logging for multi-turn
    default_states: Optional[str] = None        # Human-readable snapshot of tool default states


class BFCLv4Question(BFCLQuestion):
    sources: Any = []
    memory_context: List = []


class MultiChallengeQuestion(FormattedQuestion):
    original_conversation: List[Dict[str, Any]]
    evaluation_criteria: Dict


###

class RuleBasedOutput(BaseModel):
    passed: bool
    reason: Optional[str] = None

class FilterResult(BaseModel):
    """Individual filter result (universal or specific)"""
    is_flawed: bool
    error_category: Optional[str]
    reasoning: Optional[str]
    reasoning_summary: Optional[str]

class LLMJudgeOutput(BaseModel):
    # filtering results (separated by filter type)
    universal_filter: Optional[FilterResult] = None
    specific_filter: Optional[FilterResult] = None
    # scoring
    scores: Optional[Dict] = None
    # meta
    meta: Optional[Dict] = None

class PipelineOutput(BaseModel):
    rule_based_output: Optional[RuleBasedOutput] = None
    llm_judge_output: Optional[LLMJudgeOutput] = None

class LLMJudgeStep(Enum):
    UNIVERSAL_FILTER = "universal_filter"
    SPECIFIC_FILTER = "specific_filter"
    SCORE = "score"
