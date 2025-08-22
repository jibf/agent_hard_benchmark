# BaseBenchmark

A simplified and unified framework for benchmark evaluation.

## Structure

```
basebenchmark/
├── __init__.py          # Main package exports
├── core/                # Core data models
│   ├── __init__.py
│   └── models.py        # Data models (BenchmarkTask, BenchmarkConfig, etc.)
└── adapters/            # Benchmark adapters
    ├── __init__.py
    └── bfcl_adapter.py  # BFCL benchmark adapter
```

## Quick Start

```python
from basebenchmark import BenchmarkConfig, BFCLAdapter

# Create configuration
config = BenchmarkConfig(
    model_name="openai/gpt-4o-20240806",
    api_key="your-api-key",
    base_url="your-base-url",
    temperature=0.5,
    max_tokens=4096
)

# Create adapter
adapter = BFCLAdapter(config=config)

# Load dataset
dataset = adapter.load_dataset(config)

# Run evaluation
for task in dataset.tasks:
    response = adapter.get_response_results(task)
    metrics = adapter.evaluate(response, task)
    # Process results...
```

## Core Models

### BenchmarkConfig
Configuration for benchmark execution.

### BenchmarkTask
Individual task data with ground truth.

### BenchmarkDataset
Collection of tasks for evaluation.

### UnifiedBenchmarkResult
Standardized result format for all benchmarks.

## Adapter Interface

Each benchmark adapter implements 4 core methods:

1. **`load_dataset(config)`** - Load and filter tasks
2. **`get_response_results(task)`** - Generate model response
3. **`evaluate(response, task)`** - Evaluate single task
4. **`get_final_kpi(results_file)`** - Calculate final metrics

## Universal Runner

The `run_benchmark.py` script provides a universal interface for running any benchmark:

- **Benchmark Selection**: Use `--benchmark` to specify which benchmark to run
- **Category Support**: Use `--category` to specify benchmark-specific categories
- **Multi-threading**: Use `--max-workers` to control concurrency
- **Progress Tracking**: Real-time progress bar and statistics

## Usage Example

```bash
# List available benchmarks
python run_benchmark.py --list

# Run BFCL benchmark (default: simple category)
python run_benchmark.py \
    --benchmark bfcl \
    --model-name "openai/gpt-4o-20240806" \
    --api-key "your-key" \
    --base-url "your-url" \
    --max-tasks 10 \
    --max-workers 4

# Run BFCL benchmark with specific category
python run_benchmark.py \
    --benchmark bfcl \
    --category java \
    --model-name "openai/gpt-4o-20240806" \
    --api-key "your-key" \
    --base-url "your-url" \
    --max-tasks 10

# Run multi-turn function calling
python run_benchmark.py \
    --benchmark bfcl \
    --category multi_turn_base \
    --model-name "openai/gpt-4o-20240806" \
    --api-key "your-key" \
    --base-url "your-url" \
    --max-tasks 5
```

## Features

- **Simplified**: Only essential components
- **Unified**: Standard interface for all benchmarks
- **Extensible**: Easy to add new benchmarks
- **Robust**: Direct error handling, no hidden defaults
- **Multi-threaded**: Concurrent task processing with progress tracking
