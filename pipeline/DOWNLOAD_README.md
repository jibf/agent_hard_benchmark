# AgentHard Dataset Downloader

This repository contains a comprehensive pipeline for downloading, filtering, and evaluating AI agent benchmarks from the AgentHard dataset.

## Overview

The pipeline consists of two main stages:
1. **Rule-based Filtering (Step 1)**: Filters out problematic samples using rule-based checks and benchmark-specific filtering rules
2. **LLM-as-Judge Filtering (Step 2)**: Uses LLM evaluation to further refine the dataset quality

## Download Process

### 1. Download AgentHard Datasets

Run the download script to fetch all benchmark datasets:

```bash
python3 download_agenthard.py
```

This will download the following benchmarks to the `benchmark/` directory:
- **ACEBench-evaluation**: Tool usage and function calling evaluation
- **BFCL-evaluation**: Benchmark for function calling and language understanding
- **DrafterBench-evaluation**: Code generation and editing tasks
- **NexusBench-evaluation**: Multi-modal reasoning and tool usage
- **ToolSandbox-evaluation**: Sandboxed tool execution environment
- **complex-func-bench-evaluation**: Complex function composition tasks
- **multi_challenge-evaluation**: Multi-turn conversation challenges
- **tau-bench-evaluation**: Task understanding and execution
- **tau2-bench-evaluation**: Enhanced task understanding tasks

### 2. Dataset Structure

Each benchmark directory contains:
- `.jsonl` files with evaluation data
- Model responses and scoring information
- Metadata for filtering and analysis

## Pipeline Usage

### Step 1: Rule-based Filtering

Run rule-based filtering to prune problematic samples:

```bash
# Use comprehensive filtering (default)
python3 main.py --target_benchmark [BENCHMARK_NAME]

# Use benchmark-specific filtering rules
python3 main.py --target_benchmark [BENCHMARK_NAME] --specific-step1

# Skip rule-based filtering (go directly to step 2)
python3 main.py --target_benchmark [BENCHMARK_NAME] --skip-rule-based
```

**Output**: 
- `filtered_datasets/unified_pruned_[filter_name].jsonl` - Unified pruned dataset
- `filtered_datasets/[benchmark_name]_pruned_[filter_name].jsonl` - Benchmark-specific files

### Step 2: LLM-as-Judge Filtering

Use LLM evaluation for further refinement:

```bash
python3 main.py --target_benchmark [BENCHMARK_NAME] --skip-rule-based --num-proc 32
```

## Benchmark-Specific Features

### MultiChallenge
- **Categories**: Instruction Retention, Inference Memory, Reliable Versioned Editing, Self-Coherence
- **Filtering**: Memory failure detection, instruction violation checks, self-contradiction analysis
- **Prompts**: Specialized filtering and scoring prompts for conversation quality

### ACEBench
- **Categories**: Normal, Special, Agent splits
- **Filtering**: Parameter value validation, function call correctness, error handling
- **Prompts**: Tool usage evaluation prompts with parameter accuracy focus

### Other Benchmarks
- **DrafterBench**: Code generation quality filtering
- **BFCL**: Function calling and language understanding
- **NexusBench**: Multi-modal reasoning tasks
- **TauBench**: Task understanding and execution

## File Structure

```
pipeline/
├── benchmark/                    # Downloaded datasets
├── filtered_datasets/           # Output from step 1
├── src/
│   ├── bench_loaders/          # Benchmark-specific data loaders
│   ├── benchmark_specific_filters/  # Custom filtering rules
│   ├── prompts/                # LLM-as-judge prompts
│   └── utils/                  # Data types and utilities
├── download_agenthard.py       # Download script
├── main.py                     # Main pipeline script
└── README.md                   # This file
```

## Requirements

- Python 3.8+
- Required packages: See `requirements.txt`
- Sufficient disk space for datasets (~10GB+)

## Notes

- The download process may take several minutes depending on network speed
- Each benchmark has its own specialized filtering rules and prompts
- The pipeline automatically handles both unified and benchmark-specific outputs
- All filtered datasets are saved in JSONL format for easy processing

