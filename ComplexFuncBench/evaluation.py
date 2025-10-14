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
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from dotenv import load_dotenv

from utils.logger import Logger
from utils.utils import *

from runner.gpt_runner import GPTRunner
from runner.response_runner import RespEvalRunner


load_dotenv()



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs/test.log")
    parser.add_argument("--input_file", type=str, default="data/ComplexFuncBench.jsonl")
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to be evaluated.")
    parser.add_argument('--exp_name', type=str, default='full-1000')
    parser.add_argument("--vllm_url", type=str, default=os.environ['BASE_URL'])
    parser.add_argument("--proc_num", type=int, default=1)
    parser.add_argument("--eval_model", type=str, default="openai/gpt-4o-20240806", help="The evaluation model for response evaluation (default: openai/gpt-4o-20240806)")
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

    model = GPTRunner(args=args, logger=logger)
    resp_eval_model = RespEvalRunner(args=args, logger=logger)

    logger.info(f"Test Example {data['id']}")
    logger.info(f"Query: {data['conversations'][0]['content']}")
    
    turn_count, call_count = 0, 0
    for turn in data['conversations']:
        if turn['role'] == "assistant" and "function_call" in turn:
            turn_count += 1
            call_count += len(turn["function_call"])

    convs, message, turn_id, correct_count = model.run(data)

    # Handle API errors
    if isinstance(message, dict):
        if message["error_type"] == "unknown_error":
            print(f"\nError in sample {data['id']}: {message['content']}", file=sys.stderr)
            return None
        elif message["error_type"] == "context_length_exceeded":
            print(f"\nContext length exceeded in sample {data['id']}: {message['content']}", file=sys.stderr)
            # Continue processing to save partial results instead of returning None
    
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

    # Check and warn about eval_model
    eval_model = args.eval_model
    normalized_model = eval_model.split('/')[-1].replace('-', '').lower()
    expected_normalized = "gpt4o20240806"

    if normalized_model != expected_normalized:
        warning_msg = [
            "=" * 80,
            "WARNING: UNSUPPORTED EVALUATION MODEL DETECTED!",
            f"Current eval_model: '{eval_model}'",
            f"Official supported model: 'gpt-4o-2024-08-06'",
            "Using a different model may lead to UNRELIABLE evaluation results!",
            "=" * 80
        ]
        print("\n" + "\n".join(warning_msg) + "\n", flush=True)

    test_data = load_json(args.input_file)
    if args.debug:
        test_data = random.sample(test_data, 10)

    if os.path.exists(args.output_dir):
        finished_data = load_json(args.output_dir)
        finised_ids = [d["id"] for d in finished_data]
    else:
        finised_ids = []
    test_data = [d for d in test_data if d['id'] not in finised_ids]
    
    process_example_partial = partial(process_example, args=args)
    with Pool(processes=args.proc_num) as pool:
        with tqdm(total=len(test_data), desc="Processing samples", unit="sample") as pbar:
            for _ in pool.imap_unordered(process_example_partial, test_data):
                pbar.update(1)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
