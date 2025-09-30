# 🚀 AgentHard: LLM Tool Use Evaluation Benchmarking Suite

This repository serves as a comprehensive collection and evaluation suite for various Large Language Model (LLM) tool-use benchmarks. It provides a standardized environment and scripts to facilitate the evaluation of LLMs' capabilities in using external tools and APIs for complex task automation.

**🔧 Model Compatibility:** This suite exclusively supports **OpenAI API-compatible models**. All benchmarks use a unified API interface that requires OpenAI-compatible endpoints (OpenAI, vLLM, Ollama, custom servers, etc.).

## 🚀 Quick Start

### 1. Environment Setup

**⚠️ Note: The installation process may take 15-30 minutes depending on your internet connection and system.**

```bash
# Clone the repository
git clone <repository-url>
cd agent_hard_benchmark

# Run the setup script (this will take some time)
bash setup.sh
```

### 2. Configuration Setup

Create your environment configuration file:

```bash
# Copy the example configuration
cp .env.example .env

# Edit the .env file with your API credentials
nano .env  # or use your preferred editor
```

**Required Configuration in `.env`:**
- `API_KEY`: Your API key for the OpenAI-compatible endpoint
- `BASE_URL`: Your OpenAI-compatible API endpoint URL
- `RAPID_API_KEY`: Required for CFBench and ToolSandbox (get from RapidAPI)
- `USER_API_KEY` / `USER_BASE_URL`: Optional, for benchmarks needing separate user simulation models

**Example `.env` configuration:**
```bash
# Main API Configuration (Required)
API_KEY=your_api_key_here
BASE_URL=your_openai_compatible_endpoint_url

# User Model Configuration (Optional - for benchmarks with a llm user)
USER_API_KEY=your_user_model_key_here
USER_BASE_URL=your_user_model_endpoint_url

# RAPID_API Configuration (Required for some benchmarks)
RAPID_API_KEY=your_rapid_api_key_here
```

### 3. Verify Available Models

Before running benchmarks, check which models are available from your configured API endpoints:

```bash
# Check available models from your configured endpoints
python list_available_models.py
```

This script will:
- Load your `.env` configuration
- Test connections to both agent and user model endpoints
- Display all available model IDs that you can use
- Show your current configuration status

**Example output:**
```
Agent Hard Benchmark - Model Availability Checker
============================================================
Loaded environment from .env

AGENT/CLIENT MODELS:
------------------------------
Agent Endpoint: Testing connection to https://api.openai.com/v1
Agent Endpoint: Connected successfully
Agent Endpoint: Found 15 models
Available models for agents:
   1. gpt-4o-2024-08-06
   2. gpt-4o-mini-2024-07-18
   3. gpt-4-turbo-2024-04-09
   ...

USER MODELS (for benchmarks with a llm user):
---------------------------------------------
USER_API_KEY and USER_BASE_URL not configured
This is optional and only needed for benchmarks with a llm user like ToolSandbox
```

### 4. Running Benchmarks

#### Basic Usage

```bash
# Run all benchmarks with a specific model
python run_benchmarks.py "openai/gpt-4o-20240806"

# Run a specific benchmark
python run_benchmarks.py "openai/gpt-4o-20240806" --benchmark drafterbench

# List all available benchmarks
python run_benchmarks.py --list-benchmarks
```

#### Advanced Usage

```bash
# Run with custom parameters
python run_benchmarks.py "anthropic/claude-4-sonnet" \
    --benchmark toolsandbox \
    --temperature 0.1 \
    --proc-num 8 \
    --user-model "openai/gpt-4o-mini"

# Run multiple benchmarks concurrently (use with caution for API limits)
python run_benchmarks.py "your-model-name" \
    --benchmark "drafterbench,toolsandbox,nexusbench" \
    --concurrent
```

#### Parameter Reference

- **`model_name`** (required): Name of the model to evaluate (e.g., "openai/gpt-4o-20240806")
- **`--benchmark`**: Specific benchmark(s) to run. Options:
  - `all` (default): Run all available benchmarks
  - Single benchmark: `drafterbench`, `toolsandbox`, `nexusbench`, `cfbench`, `multichallenge`, `acebench`, `taubench`, `tau2bench`, `bfcl`
  - Multiple benchmarks: `"drafterbench,toolsandbox,nexusbench"`
- **`--temperature`**: Temperature for model generation (default: 0.0)
- **`--proc-num`**: Number of parallel processes (default: 4)
- **`--user-model`**: User model for benchmarks requiring user simulation (default: from .env or "openai/gpt-4o-20240806")
- **`--output-dir`**: Custom output directory for results
- **`--concurrent`**: Run benchmarks in parallel (⚠️ may hit API limits)
- **`--list-benchmarks`**: Show all available benchmarks

#### Available Benchmarks

| Benchmark | Description | Special Requirements |
|-----------|-------------|---------------------|
| **drafterbench** | Technical drawing revision evaluation | None |
| **toolsandbox** | Stateful tool use evaluation | RAPID_API_KEY |
| **nexusbench** | Function calling capabilities | None |
| **cfbench** | Complex function benchmarking | RAPID_API_KEY |
| **multichallenge** | Multi-domain task evaluation | None |
| **acebench** | Agent capability evaluation | Multiple provider keys |
| **taubench** | User simulation (retail+airline) | USER_MODEL |
| **tau2bench** | User simulation (retail+airline+telecom) | USER_MODEL |
| **bfcl** | Berkeley Function Call Leaderboard | None |

### 4. Results

Results will be saved in the `results/` directory with timestamps. Each benchmark creates its own subdirectory with detailed logs and evaluation metrics.

---

## Benchmarks

This repository includes the following benchmarks:

### DrafterBench

DrafterBench is designed for the comprehensive evaluation of LLM agents in the context of technical drawing revision in civil engineering.

*   **Paper/Resource:** [DrafterBench: Benchmarking Large Language Models for Tasks Automation in Civil Engineering](https://arxiv.org/abs/2507.11527)

**How to Run DrafterBench:**
1.  Follow the [README](./DrafterBench/README.md) to set up the environment.
2.  Run the following commands to get evaluation results:
    ```bash
    OPENAI_API_KEY=<YOUR_OPENAI_API_KEY> python evaluation.py --model openai/o4-mini-high  --model-provider openai --temperature 0.0 --vllm_url <YOUR_BASE_URL>

    OPENAI_API_KEY=<YOUR_OPENAI_API_KEY> python evaluation.py --model togetherai/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8  --model-provider together_ai --temperature 0.0 --vllm_url <YOUR_BASE_URL>
    ```
3.  Run script `./DrafterBench/cal_avg_metric.py` to get the average score.

### ToolSandbox

ToolSandbox is a stateful, conversational, and interactive evaluation benchmark for LLM tool use capabilities. It focuses on evaluating models over stateful tool execution and implicit state dependencies between tools.

*   **Paper/Resource:** [ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities](https://arxiv.org/abs/2408.04682)

**How to Run ToolSandbox:**
1.  Follow the [README](./ToolSandbox/README.md) to set up the environment.
2.  Run the following command to get evaluation results:
    ```bash
    env OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>  OPENAI_BASE_URL=<YOUR_BASE_URL> tool_sandbox --user GPT_4_o_2024_08_06 --agent GPT_4_o_2024_08_06
    ```
3.  Run script `./ToolSandbox/cal_avg_benchmark.py` to get the average score.

### NexusBench

NexusBench is a benchmarking suite for function call, tool use, and agent capabilities of LLMs.

*   **GitHub Repository:** [NexusflowAI/NexusBench](https://github.com/nexusflowai/NexusBench)

**How to Run NexusBench:**
1.  Follow the [README](./NexusBench/README.md) to set up the environment.
2.  Set up `.env` files with your API key and base URL:
    ```
    API_KEY=<API_KEY>
    BASE_URL=<URL>
    ```
3.  Run the following command to get evaluation results:
    ```bash
    nexusbench --client OpenAI --model openai/gpt-4o-mini --benchmarks all --output_dir ./results/
    ```

### CFBench

This section outlines how to run the CFBench evaluation.

**How to Run CFBench:**
1.  Follow the [README](./CFBenchmark/README.md) to set up the environment.
2.  Navigate to the source directory and run the evaluation command:
    ```bash
    cd CFBenchmark/CFBenchmark-basic/src

    OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>  OPENAI_BASE_URL=<YOUR_BASE_URL> python run.py --model_name=openai/gpt-4o-20240806
    ```
3.  Calculate the average score using F1 score and cos_similarity.

### multi_challenge

This section provides instructions for running the multi_challenge benchmark.

**How to Run multi_challenge:**
1.  Follow the [README](./multi_challenge/README.md) to set up the environment.
2.  Navigate to the `multi_challenge` directory and execute the evaluation command:
    ```bash
    cd multi_challenge

    OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>  OPENAI_BASE_URL=<YOUR_BASE_URL> python main.py --model-provider openai --provider-args model=openai/gpt-4o-20240806 temp=0  --attempts 3  --output-file results/gpt4o_20240806_evaluation_results.txt --raw results/gpt4o_20240806_detailed_results.csv
    ```

### ACEBench

ACEBench is a benchmark for evaluating LLM agents. 

*   **Paper/Resource:** [ACEBench: A Comprehensive Evaluation Benchmark for LLM Agents](https://arxiv.org/abs/2501.12851)

**How to Run ACEBench:**
1.  Follow the [README](./ACEBench/README.md) to set up the environment (Qwen3 support may require higher vLLM versions).
2.  Set up `.env` files with the necessary API keys and base URLs:
    ```
    OPENAI_API_BASE=<URL>
    OPENAI_API_KEY=<API>
    GPT_AGENT_API_KEY=<API>
    GPT_BASE_URL=<URL>
    GPT_API_KEY=<API>
    GPT_BASE_URL=<URL>
    DEEPSEEK_API_KEY=<API>
    DEEPSEEK_BASE_URL=<URL>
    QWEN_API_KEY=<API>
    QWEN_BASE_URL=<URL>
    ```
3.  Run the generation command (update `category normal` to `test_all` if needed):
    ```bash
    # API model
    python generate.py --model openai/gpt-4o-20240806 --category normal --language en

    # Local model
    # Download Huggingface snapshot model first
    python generate.py --model Qwen3-32B-local --model-path /nethome/hsuh45/.cache/huggingface/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137/ --category normal --language en --num-gpus 4 
    ```
4.  After generation, run the evaluation command:
    ```bash
    python eval_main.py --model <model_name> --category <category> --language <language>
    ```

### TauBench

TauBench is a benchmark for evaluating LLM agents in retail and airline environments. 

*   **Paper/Resource:** [TauBench: A Tool-Use Benchmark for LLM Agents in Retail Environments](https://arxiv.org/abs/2406.12045)

**How to Run TauBench:**
1.  Follow the [README](./tau-bench/README.md) to set up the environment (Qwen3 support may require higher vLLM versions).
2.  Set up `.env` files with your API keys and base URLs:
    ```
    OPENAI_API_KEY=<API>
    OPENAI_API_BASE=<URL>
    ANTHROPIC_API_KEY=<API>
    ANTHROPIC_API_BASE=<URL>
    VLLM_API_BASE=<API>
    ``` 
3.  Run the following command:
    ```bash
    # API model
    python run.py --agent-strategy tool-calling --env retail --model togetherai/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 --model-provider together_ai --user-model openai/gpt-4o-20240806 --user-model-provider openai --user-strategy llm --max-concurrency 10
    ```

### BFCL-v3

BFCL-v3 (Berkeley Function Call Leaderboard v3) is a benchmark focusing on multi-turn function calling capabilities of LLMs.

*   **Resource:** [BFCL-v3: Multi-turn Function Call Leaderboard](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)

**How to Run BFCL-v3:**
1.  Follow the [README](./gorilla/berkeley-function-call-leaderboard/README.md) to set up the environment.
2.  Navigate to the benchmark directory and run the generation and evaluation commands:
    ```bash
    cd gorilla/berkeley-function-call-leaderboard/

    bfcl generate --model openai/gpt-4o-20240806 --num-threads 4
    bfcl evaluate --model openai/gpt-4o-mini 
    ```
