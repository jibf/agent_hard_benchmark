# 🚀 AgentHard: LLM Tool Use Evaluation Benchmarking Suite

This repository serves as a comprehensive collection and evaluation suite for various Large Language Model (LLM) tool-use benchmarks. It provides a standardized environment and scripts to facilitate the evaluation of LLMs' capabilities in using external tools and APIs for complex task automation.

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
