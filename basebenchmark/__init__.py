"""
BaseBenchmark - Unified Benchmark Framework

A simplified and unified framework for benchmark evaluation.
"""

from .core import (
    BenchmarkTask, BenchmarkDataset, BenchmarkConfig,
    BenchmarkType, EvaluationMetrics, UnifiedBenchmarkResult
)

from .adapters import BFCLAdapter

__all__ = [
    # Core models
    "BenchmarkTask",
    "BenchmarkDataset", 
    "BenchmarkConfig",
    "BenchmarkType",
    "EvaluationMetrics",
    "UnifiedBenchmarkResult",
    
    # Adapters
    "BFCLAdapter",
]

__version__ = "1.0.0"
