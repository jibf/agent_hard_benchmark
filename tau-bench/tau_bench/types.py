# Copyright Sierra

from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

RESPOND_ACTION_NAME = "respond"
RESPOND_ACTION_FIELD_NAME = "content"


class Action(BaseModel):
    name: str
    kwargs: Dict[str, Any]


class Task(BaseModel):
    user_id: str
    actions: List[Action]
    instruction: str
    outputs: List[str]


class RewardOutputInfo(BaseModel):
    r_outputs: float
    outputs: Dict[str, bool]


class RewardActionInfo(BaseModel):
    r_actions: float
    gt_data_hash: str


class RewardResult(BaseModel):
    reward: float
    info: Union[RewardOutputInfo, RewardActionInfo]
    actions: List[Action]


class SolveResult(BaseModel):
    reward: float
    messages: List[Dict[str, Any]]
    info: Dict[str, Any]
    total_cost: Optional[float] = None


class EnvInfo(BaseModel):
    task: Task
    source: Optional[str] = None
    user_cost: Optional[float] = None
    reward_info: Optional[RewardResult] = None


class EnvResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    info: EnvInfo


class EnvResetResponse(BaseModel):
    observation: str
    info: EnvInfo


class RunMetadata(BaseModel):
    """Comprehensive metadata about the benchmark run"""
    # Benchmark information
    benchmark: str  # e.g., "tau-bench"
    environment: str  # e.g., "retail", "airline"
    task_split: str  # e.g., "test", "train", "dev"
    
    # Model configuration
    agent_model: str  # e.g., "gpt-4o", "claude-3-5-sonnet-20240620"
    agent_model_provider: str  # e.g., "openai", "anthropic"
    agent_strategy: str  # e.g., "tool-calling", "act", "react"
    user_model: str  # e.g., "gpt-4o"
    user_model_provider: str  # e.g., "openai", "anthropic"
    user_strategy: str  # e.g., "llm", "react", "verify"
    
    # Sampling parameters
    temperature: float
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    
    # Execution configuration
    num_trials: int
    max_concurrency: int
    seed: int
    shuffle: bool
    
    # Task range
    start_index: int
    end_index: int
    task_ids: Optional[List[int]] = None
    
    # Timing information
    timestamp_start: datetime
    timestamp_end: Optional[datetime] = None
    total_duration_seconds: Optional[float] = None
    
    # Additional configuration
    few_shot_displays_path: Optional[str] = None
    log_dir: str = "results"


class EnvRunResult(BaseModel):
    task_id: int
    reward: float
    info: Dict[str, Any]
    traj: List[Dict[str, Any]]
    trial: int
    metadata: Optional[RunMetadata] = None


class RunConfig(BaseModel):
    model_provider: str
    user_model_provider: str
    model: str
    user_model: str = "gpt-4o"
    num_trials: int = 1
    env: str = "retail"
    agent_strategy: str = "tool-calling"
    temperature: float = 0.0
    task_split: str = "test"
    start_index: int = 0
    end_index: int = -1
    task_ids: Optional[List[int]] = None
    log_dir: str = "results"
    max_concurrency: int = 1
    seed: int = 10
    shuffle: int = 0
    user_strategy: str = "llm"
    few_shot_displays_path: Optional[str] = None
    base_url: Optional[str] = None
