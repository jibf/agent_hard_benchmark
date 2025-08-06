# -*- coding: utf-8 -*-
import json
import copy
import sys
import os

sys.path.append(os.path.abspath(".."))

from DrafterBench.methods import task_sets
from DrafterBench.methods.agent import Drafter_agent
from DrafterBench.prompts.prompt import Prompt
from DrafterBench.methods.collect_result import process_code, execute_code


def openfile(file):
    f = open(file, "r", encoding="utf-8")
    content = json.load(f)
    return content


def savedate(data, jsonpath):
    with open(jsonpath, "w", encoding="utf-8") as w:
        json.dump(data, w, indent=4)


def generator(model, model_provider, temperature, vllm_url, max_length, response_results, data, result_path, task):
    if len(response_results) <= max_length:
        agent = Drafter_agent(
            model, model_provider, temperature, vllm_url
        )
        indx = task_sets.index(task["Tasktype"]) + 1
        prompt = Prompt(str(indx), task["Instruction"])
        pre_code = agent.get_response(messages=prompt.message())
        test_code = process_code(pre_code)
        test_info = execute_code(test_code)
        ground_code = task["Groundtruth"]
        test_ground_code = process_code(ground_code)
        ground_info = execute_code(test_ground_code)
        response = copy.deepcopy(task)
        response.update({"Response_code": pre_code})
        data.append(copy.deepcopy(response))
        savedate(list(data), result_path)
        response.update({"Groundpath": ground_info, "Testpath": test_info})
        response_results.append(response)
    return response_results
