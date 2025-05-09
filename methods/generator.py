# -*- coding: utf-8 -*-
import json
import copy
import sys
import os

sys.path.append(os.path.abspath('..'))
from DrafterBench.methods import task_sets

from DrafterBench.methods.agent import Drafter_agent
from DrafterBench.prompts.prompt import Prommt
from DrafterBench.methods.collect_result import process_code, execute_code


def openfile(file):
    f = open(file, "r", encoding="utf-8")
    content = json.load(f)
    return content


def savedate(data, jsonpath):
    with open(jsonpath, "w", encoding="utf-8") as w:
        json.dump(data, w, indent=4)


def generator(
    args, max_length, response_results, data, task_messages, result_path, task_set
):
    if len(response_results) <= max_length:
        agent = Drafter_agent(
            args.model, args.model_provider, args.temperature, args.vllm_url
        )
        task_message = task_messages[task_set]
        indx = task_sets.index(task_set) + 1
        rang = range(1, 3) if args.debug else range(1, len(task_message) + 1)
        for i in rang:
            task = task_message[i - 1]
            task_parameters = [
                task["Precise|Vague"],
                task["Completed|Error"],
                task["Single|Multiple_objects"],
                task["Single|Multiple_operations"],
                task["Structured/Unstructured"],
            ]
            if args.task_group == 'All' or any(
                [True if x == args.task_group else False for x in task_parameters]
            ):
                prompt = Prommt(str(indx), task["Instruction"])
                pre_code = agent.get_response(messages=prompt.message())
                test_code = process_code(pre_code)
                test_info = execute_code(test_code)
                ground_code = task["Groundtruth"]
                test_ground_code = process_code(ground_code)
                ground_info = execute_code(test_ground_code)
                response = copy.deepcopy(task)
                response.update({'Response_code': pre_code})
                data.append(copy.deepcopy(response))
                savedate(list(data), result_path)
                response.update({'Groundpath': ground_info, 'Testpath': test_info})
                response_results.append(response)
    return response_results
