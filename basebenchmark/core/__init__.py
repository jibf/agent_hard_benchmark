"""
BaseBenchmark Core - Data Models

This package provides the core data models for the BaseBenchmark framework.
"""

from .models import (
    BenchmarkTask, BenchmarkDataset, BenchmarkConfig,
    BenchmarkType, EvaluationMetrics, UnifiedBenchmarkResult
)

__all__ = [
    "BenchmarkTask",
    "BenchmarkDataset", 
    "BenchmarkConfig",
    "BenchmarkType",
    "EvaluationMetrics",
    "UnifiedBenchmarkResult",
]

__version__ = "1.0.0"

