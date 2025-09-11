# AgentHard Benchmark Filtering Pipeline

A comprehensive pipeline for filtering and quality assessment of AgentHard benchmark data using rule-based and LLM-as-Judge approaches.

## Overview

This pipeline implements a two-stage filtering process for AgentHard evaluation datasets to improve benchmark quality:

1. **Step 1: Rule-Based Filtering** - Removes problematic samples using comprehensive rules and benchmark-specific filtering
2. **Step 2: LLM-as-Judge Filtering** - Evaluates benchmark quality and detects flaws using LLM assessment

## Quick Start

1. **Download Data**: `python3 download_agenthard.py`
2. **Run Pipeline**: `python main.py`
3. **View Results**: Check generated JSONL files and logs

## Repository Structure

```
pipeline/
├── main.py                           # Main pipeline orchestrator
├── download_agenthard.py             # Download AgentHard datasets
├── DOWNLOAD_README.md                # Download script documentation
├── src/
│   ├── comprehensive_rule_filtering.py  # General rule-based filtering
│   ├── rule_filtering_orchestrator.py   # Orchestrates filtering strategy
│   ├── benchmark_specific_filters/      # Custom filtering rules per benchmark
│   │   ├── multi_challenge_filter.py    # MultiChallenge-specific rules
│   │   ├── ace_bench_filter.py          # ACEBench-specific rules
│   │   └── ...                          # Other benchmark filters
│   ├── bench_loaders/                   # Benchmark-specific data loaders
│   │   ├── multi_challenge_loader.py    # MultiChallenge data loader
│   │   ├── ace_bench_loader.py          # ACEBench data loader
│   │   └── ...                          # Other benchmark loaders
│   ├── prompts/                         # LLM-as-judge prompts
│   │   ├── multi_challenge_prompt.py    # MultiChallenge prompts
│   │   ├── ace_bench_prompt.py          # ACEBench prompts
│   │   └── ...                          # Other benchmark prompts
│   ├── llm_judge_filtering.py          # Step 2: LLM-as-Judge assessment
│   ├── data_loader.py                   # Universal data loading utility
│   └── utils/                           # Data types and utilities
├── benchmark/                          # Downloaded AgentHard datasets
│   ├── ACEBench-evaluation/            # Tool usage evaluation
│   ├── BFCL-evaluation/                # Function calling evaluation
│   ├── DrafterBench-evaluation/        # Code generation evaluation
│   ├── NexusBench-evaluation/          # Multi-domain agent evaluation
│   ├── ToolSandbox-evaluation/         # Tool usage evaluation
│   ├── complex-func-bench-evaluation/  # Complex function calling evaluation
│   ├── multi_challenge-evaluation/     # Multi-challenge evaluation
│   ├── tau-bench-evaluation/           # Task understanding evaluation
│   └── tau2-bench-evaluation/          # Enhanced task understanding
├── filtered_datasets/                  # Output from step 1 filtering
└── README.md                           # This file
```

## Quick Start

### Step 0: Download Benchmark Data (Required First)

Before running the filtering pipeline, you need to download the AgentHard evaluation datasets:

```bash
# Download all AgentHard datasets to benchmark/ folder
python3 download_agenthard.py

# Expected download size: ~10GB+ for all datasets
```

**Prerequisites for downloading:**
```bash
# Install required package
pip install -U huggingface_hub

# For gated datasets, authenticate with Hugging Face
huggingface-cli login
```

### Step 1: Rule-Based Filtering

Run rule-based filtering to prune problematic samples:

```bash
# Use comprehensive filtering (default)
python3 main.py --target_benchmark [BENCHMARK_NAME]

# Use benchmark-specific filtering rules
python3 main.py --target_benchmark [BENCHMARK_NAME] --specific-step1

# Examples:
python3 main.py --target_benchmark multi_challenge --specific-step1
python3 main.py --target_benchmark ace_bench --specific-step1
```

**Output from Step 1**:
- `filtered_datasets/unified_pruned_[filter_name].jsonl` - Unified pruned dataset
- `filtered_datasets/[benchmark_name]_pruned_[filter_name].jsonl` - Benchmark-specific files

### Step 2: LLM-as-Judge Filtering

Use LLM evaluation for further refinement:

```bash
# Run both steps
python3 main.py --target_benchmark [BENCHMARK_NAME]

# Skip rule-based filtering (go directly to step 2)
python3 main.py --target_benchmark [BENCHMARK_NAME] --skip-rule-based --num-proc 32
```

## Command Line Options

```bash
python3 main.py [OPTIONS]

Options:
  --target_benchmark BENCHMARK    Target benchmark name (required)
  --specific-step1               Use benchmark-specific filtering rules
  --skip-rule-based              Skip Step 1 (rule-based filtering)
  --skip-llm-judge               Skip Step 2 (LLM-as-Judge filtering)
  --num-proc N                   Number of processes for LLM evaluation (default: 1)
  --llm-model MODEL              LLM model to use (default: gpt-4o)
  --llm-max-samples N            Maximum samples for Step 2 (default: all)
  --llm-batch-size N             Batch size for LLM processing (default: 10)
  -h, --help                     Show help message
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

## Pipeline Architecture

### Step 1: Rule-Based Filtering

**Comprehensive Filtering** (`src/comprehensive_rule_filtering.py`):
- Question-level discriminativeness (variance > 0.01)
- Score sanity checks
- Basic structure validation

**Benchmark-Specific Filtering** (`src/benchmark_specific_filters/`):
- Custom rules for each benchmark
- Domain-specific quality checks
- Always includes variance-based discriminativeness as final step

**Orchestrator** (`src/rule_filtering_orchestrator.py`):
- Selects appropriate filtering strategy
- Saves both unified and benchmark-specific outputs
- Manages file organization

### Step 2: LLM-as-Judge Filtering

**File**: `src/llm_judge_filtering.py`

**Quality Evaluation** (1-5 scale):
- Task-specific evaluation criteria
- Benchmark-appropriate scoring dimensions
- Flaw detection and quality assessment

## Output Structure

### Step 1 Results
```
filtered_datasets/
├── unified_pruned_comprehensive.jsonl          # All benchmarks combined
├── multi_challenge_pruned_multi_challenge.jsonl # MultiChallenge specific
├── ace_bench_pruned_ace_bench.jsonl            # ACEBench specific
└── ...                                         # Other benchmarks
```

### Step 2 Results
- Final filtered samples for LLM evaluation
- Quality assessment reports
- Benchmark-specific scoring

## Examples

### MultiChallenge Filtering
```bash
# Use MultiChallenge-specific rules
python3 main.py --target_benchmark multi_challenge --specific-step1

# Output: filtered_datasets/multi_challenge_pruned_multi_challenge.jsonl
```

### ACEBench Filtering
```bash
# Use ACEBench-specific rules
python3 main.py --target_benchmark ace_bench --specific-step1

# Output: filtered_datasets/ace_bench_pruned_ace_bench.jsonl
```

### Complete Pipeline
```bash
# Run both steps with MultiChallenge
python3 main.py --target_benchmark multi_challenge --specific-step1

# Run both steps with ACEBench
python3 main.py --target_benchmark ace_bench --specific-step1
```

## Prerequisites

### For Data Download
- Python 3.6+
- `huggingface_hub` package: `pip install -U huggingface_hub`

### For Pipeline Execution
- Python 3.8+
- Required packages: `numpy`, `json`, `openai`
- OpenAI API key for LLM-as-judge (set `OPENAI_API_KEY` environment variable)

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Expected Results

### Step 1 Results (Rule-Based Filtering)
- **Comprehensive Filtering**: Removes non-discriminative questions (variance ≤ 0.01)
- **Benchmark-Specific Filtering**: Additional domain-specific quality checks
- **Output**: Both unified and benchmark-specific pruned datasets

### Step 2 Results (LLM-as-Judge Filtering)
- Further quality refinement using LLM evaluation
- Benchmark-specific scoring and assessment
- Final high-quality dataset for evaluation

## Cost Management

### LLM Usage
- **Model**: Use `gpt-4o-mini` for cost-effectiveness
- **Batch Size**: Larger batches reduce API overhead
- **Sample Limits**: Use `--llm-max-samples` for testing

## Troubleshooting

### Common Issues

1. **Benchmark Directory Not Found**
   ```bash
   # Check if datasets were downloaded
   ls benchmark/
   
   # Re-download if needed
   python3 download_agenthard.py
   ```

2. **Import Errors**
   ```bash
   # Ensure you're in the pipeline directory
   cd /path/to/pipeline
   
   # Run from root directory
   python3 main.py --target_benchmark [BENCHMARK_NAME]
   ```

3. **OpenAI API Issues**
   ```bash
   # Check API key
   echo $OPENAI_API_KEY
   
   # Set API key
   export OPENAI_API_KEY="your-key-here"
   ```

## Testing

```bash
# Test benchmark loaders
python3 -c "from src.bench_loaders.multi_challenge_loader import MultiChallengeLoader; loader = MultiChallengeLoader(); print(f'Loaded {len(loader.load_questions())} questions')"

# Test specific filtering
python3 test_specific_filtering.py
```

## Notes

- Each benchmark has specialized filtering rules and prompts
- The pipeline automatically handles both unified and benchmark-specific outputs
- All filtered datasets are saved in JSONL format for easy processing
- Benchmark-specific filtering always includes variance-based discriminativeness
- The orchestrator manages file organization
