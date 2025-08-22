"""
Unified Data Models for Benchmark Framework

This module defines the core data structures used across all benchmarks
in the unified framework.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum





class BenchmarkType(Enum):
    """Benchmark type enumeration"""
    DRAFTERBENCH = "drafterbench"
    ACEBENCH = "acebench"
    TAUBENCH = "taubench"
    TOOLSANDBOX = "toolsandbox"
    NEXUSBENCH = "nexusbench"
    BFCL = "bfcl"


@dataclass
class BenchmarkTask:
    """Unified task representation for all benchmarks"""
    
    # Core task information
    task_id: str
    benchmark_type: BenchmarkType
    category_type: str  # Simplified from category
    
    # Data
    original_data: Dict[str, Any]  # Original benchmark data
    ground_truth: Optional[Dict[str, Any]] = None  # Ground truth/reference data
    
    # Metadata for flexibility
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamp
    created_at: datetime = field(default_factory=datetime.now)





@dataclass
class EvaluationMetrics:
    """Unified evaluation metrics"""
    
    # Core metrics
    accuracy: Optional[float] = None
    score: Optional[float] = None
    success_rate: Optional[float] = None
    
    # Detailed metrics
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # Task-specific metrics
    task_specific_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    evaluation_method: str = "default"
    evaluator: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedBenchmarkResult:
    """Unified benchmark result format for all benchmarks"""
    
    # Model information
    model_path: str  # mandatory
    benchmark_name: str  # mandatory
    
    # Sampling parameters
    sampling_params: Dict[str, Any]  # mandatory
    
    # Full conversation trajectory
    messages: List[Dict[str, Any]]  # mandatory
    
    # Evaluation result
    eval_result: Dict[str, Any]  # mandatory, must contain 'score' (0-1)
    
    # Optional fields
    user_model_path: Optional[str] = None  # optional
    task_name: Optional[str] = None  # optional, for subtasks
    user_sampling_params: Optional[Dict[str, Any]] = None  # optional
    meta: Optional[Dict[str, Any]] = None  # optional
    
    def __post_init__(self):
        """Validate the result format"""
        # Ensure score is present and between 0 and 1
        if "score" not in self.eval_result:
            raise ValueError("eval_result must contain 'score' field")
        
        score = self.eval_result["score"]
        if not isinstance(score, (int, float)) or score < 0 or score > 1:
            raise ValueError("score must be a number between 0 and 1")
        
        # Ensure required fields are present
        if not self.model_path:
            raise ValueError("model_path is mandatory")
        if not self.benchmark_name:
            raise ValueError("benchmark_name is mandatory")
        if not self.sampling_params:
            raise ValueError("sampling_params is mandatory")
        if not self.messages:
            raise ValueError("messages is mandatory")





@dataclass
class BenchmarkDataset:
    """Unified dataset representation"""
    
    # Dataset information
    name: str
    benchmark_type: BenchmarkType
    version: str = "1.0"
    
    # Tasks
    tasks: List[BenchmarkTask] = field(default_factory=list)
    
    # Metadata
    description: Optional[str] = None
    source: Optional[str] = None
    license: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    total_tasks: int = 0
    categories: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate statistics after initialization"""
        self.total_tasks = len(self.tasks)
        
        # Count categories
        for task in self.tasks:
            if task.category_type:
                self.categories[task.category_type] = self.categories.get(task.category_type, 0) + 1





@dataclass
class BenchmarkConfig:
    """Unified configuration for all benchmarks"""
    
    # API Configuration
    api_key: str
    base_url: str
    model_name: str
    model_provider: str = "openai"
    
    # Generation parameters
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    
    # Execution parameters
    max_workers: int = 4
    batch_size: int = 1
    timeout: int = 300  # seconds
    
    # Output configuration
    output_dir: str = "./unified_results"
    cache_dir: str = "./unified_cache"
    save_detailed: bool = True
    save_intermediate: bool = False
    
    # Benchmark-specific parameters
    benchmark_params: Dict[str, Any] = field(default_factory=dict)
    
    # Logging
    verbose: bool = False
    log_level: str = "INFO"
