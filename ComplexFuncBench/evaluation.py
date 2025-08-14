# -*- coding: utf-8 -*- 
import json
import random
import argparse
import os
import logging
import datetime
import sys
from collections import defaultdict
import multiprocessing
from multiprocessing import Pool, Manager
from functools import partial
from tqdm import tqdm
from dotenv import load_dotenv

from utils.logger import Logger
from utils.utils import *

from runner.gpt_runner import GPTRunner
from runner.glm_runner import GLMRunner, GLMAPIRunner
from runner.claude_runner import ClaudeRunner
from runner.qwen_runner import QwenRunner
from runner.llama_runner import LlamaRunner
from runner.mistral_runner import MistralRunner
from runner.response_runner import RespEvalRunner


load_dotenv()

MODEL_MAPPING = {
    "openai/gpt-4.1": GPTRunner,
    "openai/gpt-4o-mini": GPTRunner,
    "openai/o4-mini-high": GPTRunner,
    "openai/o3-high": GPTRunner,
    "gpt-4o-2024-08-06": GPTRunner,
    "gpt-4-turbo-2024-04-09": GPTRunner,
    "anthropic/claude-4-sonnet-thinking-off": GPTRunner,
    "anthropic/claude-4-sonnet-thinking-on": GPTRunner,
    "glm-4-9b-chat": GPTRunner,
    "glm-4-long": GPTRunner,
    "Llama-3.1-70B": GPTRunner,
    "Llama-3.1-8B": GPTRunner,
    "Meta-Llama-3.1-405B-Instruct-FP8": GPTRunner,
    "qwen2.5-7b-instruct": GPTRunner,
    "qwen2.5-72b-instruct": GPTRunner,
    "qwen2.5-7b-instruct": GPTRunner,
    "togetherai/Qwen/Qwen3-235B-A22B-FP8": GPTRunner,
    "togetherai/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8": GPTRunner,
    "togetherai/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8": GPTRunner,
    "deepseek-ai/DeepSeek-V3-0324": GPTRunner, 
    "deepseek-ai/DeepSeek-R1-0528": GPTRunner,
    "mistral-large-2407": GPTRunner,
    "google/gemini-2.5-flash-thinking-off": GPTRunner,
    "google/gemini-2.5-flash-thinking-on": GPTRunner,
    "google/gemini-2.5-pro-thinking-off": GPTRunner,
    "google/gemini-2.5-pro-thinking-on": GPTRunner,
    "xai/grok-4": GPTRunner,
    "togetherai/moonshotai/Kimi-K2-Instruct": GPTRunner
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs/test.log")
    parser.add_argument("--input_file", type=str, default="data/ComplexFuncBench.jsonl")
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_MAPPING.keys()), help="The name of the model to be evaluated.")
    parser.add_argument('--exp_name', type=str, default='full-1000')
    parser.add_argument("--vllm_url", type=str, default=os.environ['BASE_URL'])
    parser.add_argument("--proc_num", type=int, default=1)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    os.makedirs(f"logs/{datetime.date.today().strftime('%Y-%m-%d')}/{args.model_name}", exist_ok=True)
    os.makedirs(f"result/{args.model_name}/{args.exp_name}/logs", exist_ok=True)

    args.log_dir = f"logs/{datetime.date.today().strftime('%Y-%m-%d')}/{args.model_name}/{args.exp_name}.log"
    args.output_dir = f"result/{args.model_name}/{args.exp_name}.jsonl"
    args.log_dir = f"result/{args.model_name}/{args.exp_name}/logs"
    return args


def process_example(data, args):
    log_dir = f"{args.log_dir}/{data['id']}.log"
    logger = Logger(f"evaluation_logger_{data['id']}", log_dir, logging.DEBUG)

    model = MODEL_MAPPING[args.model_name](args=args, logger=logger)
    resp_eval_model = RespEvalRunner(args=args, logger=logger)

    logger.info(f"Test Example {data['id']}")
    logger.info(f"Query: {data['conversations'][0]['content']}")
    
    turn_count, call_count = 0, 0
    for turn in data['conversations']:
        if turn['role'] == "assistant" and "function_call" in turn:
            turn_count += 1
            call_count += len(turn["function_call"])

    convs, message, turn_id, correct_count = model.run(data)

    # API Error
    if isinstance(message, dict) and message["error_type"] == "unknown_error":
        print(f"\nError in sample {data['id']}: {message['content']}", file=sys.stderr)
        return None
    
    real_turn_count = 0
    for turn in convs:
        if turn['role'] == "assistant" and "function_call" in turn:
            real_turn_count += 1
    
    if convs[-1]['role'] == "assistant" and "content" in convs[-1]:
        gen_response = convs[-1]['content']
        resp_eval_result = resp_eval_model.run(data, gen_response)
    else:
        resp_eval_result = None

    logger.info(f"Message: {message}")
    logger.info(f"Success turn num = {turn_id}")
    logger.info("-" * 100)

    result = {
        "id": data['id'],
        "gen_convs": convs,
        "message": message,
        "count_dict": {
            "success_turn_num": turn_id,
            "total_turn_num": turn_count,
            "correct_call_num": correct_count,
            "total_call_num": call_count,
            "real_turn_num": real_turn_count
        },
        "resp_eval": resp_eval_result
    }

    with open(args.output_dir, 'a+') as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()

    return result


def main():
    args = get_args()
    test_data = load_json(args.input_file)
    if args.debug:
        test_data = random.sample(test_data, 10)
    
    if os.path.exists(args.output_dir):
        finished_data = load_json(args.output_dir)
        finised_ids = [d["id"] for d in finished_data]
    else:
        finised_ids = []
    test_data = [d for d in test_data if d['id'] not in finised_ids]
            
    with Manager() as manager:
        pool = Pool(processes=args.proc_num)
        process_example_partial = partial(process_example)
        
        with tqdm(total=len(test_data), desc="Processing samples", unit="sample") as pbar:
            results = []
            for data in test_data:
                result = pool.apply_async(process_example_partial, (data, args))
                results.append(result)
            
            # Wait for completion and update progress bar
            for result in results:
                result.get()
                pbar.update(1)
        
    pool.close()
    pool.join()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
