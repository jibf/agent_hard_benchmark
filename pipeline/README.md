# AgentHard Benchmark Filtering Pipeline

A comprehensive pipeline for filtering and quality assessment of AgentHard benchmark data using rule-based and LLM-as-Judge approaches.

## Overview

This pipeline implements a two-stage filtering process for AgentHard evaluation datasets to improve benchmark quality:

1. **Step 1: Comprehensive Rule-Based Filtering** - Removes clearly problematic samples and analyzes multi-model performance
2. **Step 2: LLM-as-Judge Filtering** - Evaluates benchmark quality and detects flaws using LLM assessment

## 🚀 Quick Start

1. **Download Data**: `python3 download_agenthard.py --out ./benchmark`
2. **Run Pipeline**: `python main.py`
3. **View Results**: Check generated JSONL files and logs

## Repository Structure

```
Archive (1)/
├── main.py                           # Main pipeline orchestrator
├── download_agenthard.py             # Download AgentHard datasets
├── DOWNLOAD_README.md                # Download script documentation
├── src/
│   ├── comprehensive_rule_filtering.py  # Step 1: Rule-based filtering
│   ├── llm_judge_filtering.py          # Step 2: LLM-as-Judge assessment
│   ├── data_loader.py                  # Data loading utility
│   └── __init__.py                     # Package marker
├── benchmark/                         # Downloaded AgentHard datasets
│   ├── NexusBench-evaluation/         # Multi-domain agent evaluation
│   ├── DrafterBench-evaluation/       # Code generation evaluation
│   ├── complex-func-bench-evaluation/ # Complex function calling evaluation
│   ├── multi_challenge-evaluation/    # Multi-challenge evaluation
│   ├── BFCL-evaluation/               # Benchmark for Function Calling Language
│   └── ToolSandbox-evaluation/        # Tool usage evaluation
└── README.md                          # This file
```

## Quick Start

### Step 0: Download Benchmark Data (Required First)

Before running the filtering pipeline, you need to download the AgentHard evaluation datasets:

```bash
# Download all AgentHard datasets to benchmark/ folder
python3 download_agenthard.py --out ./benchmark

# Or download only specific datasets
python3 download_agenthard.py --repo AgentHard/NexusBench-evaluation --repo AgentHard/DrafterBench-evaluation --out ./benchmark

# Download only JSONL files (recommended for faster processing)
python3 download_agenthard.py --patterns "*.jsonl" --out ./benchmark
```

**Prerequisites for downloading:**
```bash
# Install required package
pip install -U huggingface_hub

# For gated datasets, authenticate with Hugging Face
huggingface-cli login
```

**Expected download size:** ~4.3 GB for all datasets

### Step 2: Run Complete Pipeline (Step 1 + Step 2)
```bash
# Run both steps (requires OpenAI API key)
python main.py

# Run with custom LLM configuration
python main.py --llm-model gpt-4o-mini --llm-max-samples 100 --llm-batch-size 5
```

### Run Only Step 1 (Rule-Based Filtering)
```bash
# Skip LLM-as-Judge step (no API costs)
python main.py --skip-llm-judge
```

## Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --skip-llm-judge          Skip Step 2 (LLM-as-Judge filtering)
  --llm-model MODEL         LLM model to use (default: gpt-4o-mini)
  --llm-max-samples N       Maximum samples for Step 2 (default: all)
  --llm-batch-size N        Batch size for LLM processing (default: 10)
  --llm-max-retries N       Max retries for LLM calls (default: 3)
  --llm-retry-delay SEC     Delay between retries (default: 1.0)
  -h, --help               Show help message
```

## Examples

### Test Run (Small Sample)
```bash
# Process only 50 samples for testing
python main.py --llm-max-samples 50 --llm-batch-size 5
```

### Production Run
```bash
# Process all samples with cost-effective settings
python main.py --llm-model gpt-4o-mini --llm-batch-size 20
```

### Rule-Based Only
```bash
# Fast filtering without LLM costs
python main.py --skip-llm-judge
```

## Prerequisites

### For Data Download (Step 0)
- Python 3.6+
- `huggingface_hub` package: `pip install -U huggingface_hub`
- Hugging Face authentication (for gated datasets): `huggingface-cli login`

### For Step 1 (Rule-Based)
- Python 3.8+
- Required packages: `numpy`, `hashlib`, `json`

### For Step 2 (LLM-as-Judge)
- OpenAI API key: Set `OPENAI_API_KEY` environment variable
- Required packages: `openai`

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Output Files

### Step 1 Results
- `step1_rule_based_passed_samples.jsonl` - Samples that passed rule-based filtering
- `step1_rule_based_dropped_samples.jsonl` - Samples dropped by rule-based filtering

### Step 2 Results
- `step2_llm_judge_passed_samples.jsonl` - Final filtered samples
- `step2_llm_judge_dropped_samples.jsonl` - Samples dropped by LLM-as-Judge

### Logs
- `pipeline.log` - Detailed execution log

## Pipeline Architecture

### Step 1: Comprehensive Rule-Based Filtering
**File**: `src/comprehensive_rule_filtering.py`

**Sample-Level Rules**:
- Structure sanity (valid JSON, required fields)
- Conversation length (2-80 messages, ≤8k tokens)
- Scoring sanity (numeric scores within ranges)
- Obvious broken samples (degenerate output, refusals)
- Duplicate detection

**Question-Level Rules**:
- No variation → Drop question
- Binary variation → Keep question
- Continuous variation → Apply variance thresholds

### Step 2: LLM-as-Judge Filtering
**File**: `src/llm_judge_filtering.py`

**Quality Evaluation** (1-5 scale):
1. Tool Necessity
2. Planning and Context Depth
3. Parameter Generation
4. Tool Selection Difficulty
5. Real-World Applicability

**Flaw Detection**:
- Argument Value Mismatch
- Argument Type Mismatch
- Unjustified Assumption
- Misspelling
- Dataset Integrity Issue

## 📊 Expected Results

### Step 1 Results (Rule-Based Filtering)
```
Total samples: ~260,246
├── Sample-level dropped: ~189,550 (72.8%)
├── Question-level dropped: ~25,203 (9.7%)
└── Final passed: ~45,493 (17.5%)
    └── ~2,548 unique questions
```

### Step 2 Results (LLM-as-Judge Filtering)
The LLM-as-Judge stage will further filter the samples from Step 1, typically dropping:
- Low-quality benchmarks (poor tool necessity, simple tasks)
- Flawed ground truth (logical errors, incorrect parameters)
- Unrealistic scenarios (synthetic or academic examples)
- Ambiguous or poorly defined tasks

## Cost Management

### LLM Usage
- **Model**: Use `gpt-4o-mini` for cost-effectiveness
- **Batch Size**: Larger batches reduce API overhead
- **Sample Limits**: Use `--llm-max-samples` for testing

## Dataset Information

### AgentHard Datasets

The pipeline processes six AgentHard evaluation datasets:

1. **NexusBench-evaluation** (137 files) - Multi-domain agent evaluation with tool usage
2. **DrafterBench-evaluation** (208 files) - Code generation and editing tasks
3. **complex-func-bench-evaluation** (90 files) - Complex function calling scenarios
4. **multi_challenge-evaluation** (80 files) - Multi-step reasoning challenges
5. **BFCL-evaluation** (282 files) - Benchmark for Function Calling Language
6. **ToolSandbox-evaluation** (4 files) - Tool usage and sandbox testing

### Data Format

Each dataset contains JSONL files with:
- Model responses and tool calls
- Evaluation scores and metadata
- Ground truth information
- Task descriptions and parameters

## Troubleshooting

### Download Issues

1. **Hugging Face Authentication Error**
   ```bash
   # Login to Hugging Face
   huggingface-cli login
   
   # Check if huggingface_hub is installed
   pip install -U huggingface_hub
   ```

2. **Network/Download Errors**
   ```bash
   # Resume interrupted download
   python3 download_agenthard.py --out ./benchmark
   
   # Download only specific files to save bandwidth
   python3 download_agenthard.py --patterns "*.jsonl" --out ./benchmark
   ```

3. **Disk Space Issues**
   ```bash
   # Check available space (need ~4.3 GB)
   df -h
   
   # Download to different location
   python3 download_agenthard.py --out /path/to/external/drive/benchmark
   ```

### Pipeline Issues

1. **OpenAI API Error**
   ```bash
   # Check API key
   echo $OPENAI_API_KEY
   
   # Set API key
   export OPENAI_API_KEY="your-key-here"
   ```

2. **Memory Issues**
   ```bash
   # Use smaller batch size
   python main.py --llm-batch-size 5
   ```

3. **Rate Limiting**
   ```bash
   # Increase retry delay
   python main.py --llm-retry-delay 2.0
   ```

### Testing

```bash
# Test Step 1 only
python test_comprehensive_rules.py

# Test Step 2 with small sample
python test_llm_judge.py
```
