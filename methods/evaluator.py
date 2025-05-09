# -*- coding: utf-8 -*-
import json
import sys
import os

sys.path.append(os.path.abspath('..'))
from DrafterBench.methods import metric
from DrafterBench.utils.types import Score_builder
from DrafterBench.methods.collect_result import collect_result


def openfile(file):
    f = open(file, "r", encoding="utf-8")
    content = json.load(f)
    return content


def savedate(data, jsonpath):
    with open(jsonpath, "w", encoding="utf-8") as w:
        json.dump(data, w, ensure_ascii=False, indent=4)


def evaluator(result_path, eval_result, response_result):
    ground_details = metric.groundcheck(response_result['Groundpath'])
    prompt_score = (
        Score_builder()
        .ground_fill(ground_details)
        .result(
            *metric.cross_check(
                response_result['Groundpath'],
                response_result['Testpath'],
                (
                    "precise"
                    if response_result["Precise|Vague"] == "Precise"
                    else "vaguely"
                ),
            )
        )
        if response_result['Testpath']
        else Score_builder().ground_fill(ground_details).fail()
    )
    response_result.update(collect_result(prompt_score))
    del response_result['Groundpath']
    del response_result['Testpath']
    eval_result.append(response_result)
    eval_data = list(eval_result)
    savedate(eval_data, result_path)
    return eval_result
