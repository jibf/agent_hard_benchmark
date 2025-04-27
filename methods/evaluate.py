# -*- coding: utf-8 -*-
import multiprocessing
import os
import numpy as np
import datetime
import sys
import os
sys.path.append(os.path.abspath('../..'))

from tqdm import tqdm
from functools import partial
from datasets import DatasetDict, Dataset, load_dataset
from multiprocessing import Manager
from . import task_sets
from .evaluator import evaluator





def average_award(results: list):
    award_list = [x["task_score"]["task_score"] for x in results]
    return np.average(award_list)


def process_data(data: list, style: str):
    organized_data = []

    def calculate_score(success_count, total_count):
        return success_count / total_count if total_count else 1


    def score_format(style, da):
        format = {
            "language_style": style,
            "task_set": da["task_set"],
            "task_id": da["task_id"],
            "define_arguments": calculate_score(da["task_score"]["success_arguments_define"], da["task_score"]["total_arguments_define"]),
            "transfer_variable": calculate_score(da["task_score"]["success_variable_transfer"], da["task_score"]["total_variable_transfer"]),
            "call_function": calculate_score(da["task_score"]["success_function_calling"], da["task_score"]["total_function_calling"]),
            "Single_tool": calculate_score(da["task_score"]["success_single_tool_selection"], da["task_score"]["total_single_tool_selection"]),
            "Multi-tool": calculate_score(da["task_score"]["success_multi_tool_selection"], da["task_score"]["total_multi_tool_selection"]),
            "Execute_plan": calculate_score(da["task_score"]["intersected_plan_execution"], da["task_score"]["total_plans_appeared"]),
            "average_score": da["task_score"]["task_score"],
        }
        return format

    if style in ["structured", "unstructured"]:
        for da in data:
            organized_data.append(score_format(style, da))
    else:
        for da in data[0]:
            organized_data.append(score_format("Structured", da))
        for da in data[1]:
            organized_data.append(score_format("Unstructured", da))
    return organized_data


def evaluate(args):
    print(f"The benchmark is runing with following arguments:")
    print(f"{args}")
    print(f"Running tasks in {args.task_group} language style")
    os.makedirs(f"{args.log_dir}", exist_ok=True)
    os.makedirs(f"{args.result_dir}", exist_ok=True)
    log_path = f"{args.log_dir}/{datetime.date.today().strftime('%Y-%m-%d-%H:%M:%S')}_{args.model.replace('/','_')}_{args.exp_name}.log"
    result_path = f"{args.result_dir}{datetime.datetime.now().strftime('%Y-%m-%d-%H')}_{args.model.replace('/', '_')}_{args.exp_name}.json"
    structured_results = Manager().list()
    unstructured_results = Manager().list()
    results = [structured_results, unstructured_results]
    ctx = multiprocessing.get_context('spawn')
    pool = ctx.Pool(processes=args.proc_num)
    ground_codes = load_dataset("Eason666/DrafterBenchmark", "ground_code")
    structured_instructions = load_dataset(
        "Eason666/DrafterBenchmark", "structured_instructions"
    )
    unstructured_instructions = load_dataset(
        "Eason666/DrafterBenchmark", "unstructured_instructions"
    )
    evaluator_partial = partial(
        evaluator,
        args,
        results,
        ground_codes,
        structured_instructions,
        unstructured_instructions,
    )
    r = list(tqdm(pool.imap(evaluator_partial, task_sets), total=len(task_sets)))
    pool.close()
    pool.join()
    if args.task_group == "structured":
        averagy_score = average_award(structured_results)
        organized_data = process_data(structured_results, args.task_group)
    elif args.task_group == "unstructured":
        averagy_score = average_award(unstructured_results)
        organized_data = process_data(unstructured_results, args.task_group)
    else:
        averagy_score = np.average(
            [average_award(structured_results), average_award(unstructured_results)]
        )
        organized_data = process_data(
            [structured_results, unstructured_results], args.task_group
        )
    print(f"Average reward: {averagy_score}\n")
    print(f"Results saved to {result_path}")
    if args.push_result_to:
        dataset = Dataset.from_list(organized_data)
        dataet_dict = DatasetDict({"result": dataset})
        dataet_dict.push_to_hub(args.push_result_to, token=os.getenv('HUGGINGFACE_TOKEN'))
        print(
            f"Dataset successfully pushed to the hub at https://huggingface.co/{args.push_result_to}"
        )
