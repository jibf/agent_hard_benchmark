# TODO: need to finalize a detailed readme ASAP!


### run DrafterBench
1. Follow the [ReadMe](./DrafterBench/README.md) to setup environment.
2. Run the following commend to get evaluation results.
```
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY> python evaluation.py --model openai/o4-mini-high  --model-provider openai --temperature 0.0 --vllm_url <YOUR_BASE_URL>

OPENAI_API_KEY=<YOUR_OPENAI_API_KEY> python evaluation.py --model togetherai/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8  --model-provider together_ai --
temperature 0.0 --vllm_url <YOUR_BASE_URL>
```
3. Run script `./DrafterBench/cal_avg_metric.py` to get the average score.

### run ToolSandbox
1. Follow the [ReadMe](./ToolSandbox/README.md) to setup environment.
2. Run the following commend to get evaluation results.
```
env OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>  OPENAI_BASE_URL=<YOUR_BASE_URL> tool_sandbox --user GPT_4_o_2024_05_13 --agent GPT_4_o_2024_05_13
```
3. Run script `./ToolSandbox/cal_avg_benchmark.py` to get the average score.