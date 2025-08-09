# TODO: need to finalize a detailed readme ASAP!


### run DrafterBench
1. Follow the [ReadMe](./DrafterBench/README.md) to setup environment.
2. Run the following commend to get evaluation results.
```
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY> python evaluation.py --model openai/o4-mini-high  --model-provider openai --temperature 0.0 --vllm_url <YOUR_BASE_URL>

OPENAI_API_KEY=<YOUR_OPENAI_API_KEY> python evaluation.py --model togetherai/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8  --model-provider together_ai --temperature 0.0 --vllm_url <YOUR_BASE_URL>
```
3. Run script `./DrafterBench/cal_avg_metric.py` to get the average score.

### run ToolSandbox
1. Follow the [ReadMe](./ToolSandbox/README.md) to setup environment.
2. Run the following commend to get evaluation results.
```
env OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>  OPENAI_BASE_URL=<YOUR_BASE_URL> tool_sandbox --user GPT_4_o_2024_08_06 --agent GPT_4_o_2024_08_06
```
3. Run script `./ToolSandbox/cal_avg_benchmark.py` to get the average score.


### run NexusBench
1. Follow the [ReadMe](./NexusBench/README.md) to setup environment.
2. Setup .env files 
```
API_KEY=<API_KEY>
BASE_URL=<URL>
```
3. Run the following command to get evaluation results.
```
nexusbench     --client OpenAI     --model openai/gpt-4o-mini     --benchmarks all --output_dir ./results/
```

### run CFBench
1. Follow the [ReadMe](./CFBenchmark/README.md) to setup environment.
2. Run the following commend to get evaluation results.
```
cd CFBenchmark/CFBenchmark-basic/src

OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>  OPENAI_BASE_URL=<YOUR_BASE_URL> python run.py --model_name=openai/gpt-4o-20240806
```
3. Calculate the average score by F1 score and cos_similarity.

### run multi_challenge
1. Follow the [ReadMe](./multi_challenge/README.md) to setup environment.
2. Run the following commend to get evaluation results.
```
cd multi_challenge

OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>  OPENAI_BASE_URL=<YOUR_BASE_URL> python main.py --model-provider openai --provider-args model=openai/gpt-4o-20240806 temp=0  --attempts 3  --output-file results/gpt4o_20240806_evaluation_results.txt --raw results/gpt4o_20240806_detailed_results.csv

```

### run ACEBench
1. Follow the [ReadMe](./ACEBench/README.md) to setup enviroment (Qwen3 support may need higher vllm versions)
2. Setup .env files
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
3.  Run the following command (category `normal` -> should update to `test_all`)
```
# API model
# 1. Run
python generate.py --model openai/gpt-4o-20240806 --category normal --language en

# Local model
# 1. Download Huggingface snapshot model
# 2. Run 
python generate.py --model Qwen3-32B-local --model-path /nethome/hsuh45/.cache/huggingface/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137/ --category normal --language en --num-gpus 4 

# After generate -> run eval
python eval_main.py --model <model_name> --category <category> --language <language>
```

### run Taubench
1. Follow the [ReadMe](./tau-bench/README.md) to setup enviroment (Qwen3 support may need higher vllm versions)
2. Setup .env files 
```
OPENAI_API_KEY=<API>
OPENAI_API_BASE=<URL>
ANTHROPIC_API_KEY=<API>
ANTHROPIC_API_BASE=<URL>
VLLM_API_BASE=<API>
``` 
3.  Run the following command
```
# API model
python run.py --agent-strategy tool-calling --env retail --model togetherai/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 --model-provider together_ai --user-model openai/gpt-4o-20240806 --user-model-provider openai --user-strategy llm --max-concurrency 10
```

### run BFCL-v3
1. Follow the [ReadMe](./gorilla/berkeley-function-call-leaderboard/README.md) to setup environment.
2. Run the following commend to get evaluation results.
```
cd gorilla/berkeley-function-call-leaderboard/

bfcl generate --model openai/gpt-4o-20240806   --num-threads 4
bfcl evaluate --model openai/gpt-4o-mini 

```