import json
import re
import signal
import copy
import sys
import os
from datetime import datetime
from . import task_sets

sys.path.append(os.path.abspath('..'))

from DrafterBench import testf
from DrafterBench.methods.agent import Drafter_agent
from DrafterBench.prompts.prompt import Prommt
from DrafterBench.utils.types import Score_builder
from DrafterBench.methods.collect_result import collect_result


def timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out.")


def execute_code(code_string):
    try:
        variables = {}
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)
        testf.functions.taskinformation = []
        exec(code_string, variables)
        code_information = copy.deepcopy(testf.functions.taskinformation)
    except Exception as e:
        code_information = []
    return code_information


def process_code(pre_code):
    test_code = re.sub(
        "import PDFbf|import fitz",
        "from DrafterBench import testf",
        pre_code,
    )
    test_code = re.sub("PDFbf|fitz", "testf", test_code)
    return test_code


def openfile(file):
    f = open(file, "r", encoding="utf-8")
    content = json.load(f)
    return content


def savedate(data, jsonpath):
    with open(jsonpath, "w", encoding="utf-8") as w:
        json.dump(data, w, ensure_ascii=False, indent=4)


def evaluator(args, results, ground, structured, unstructured, task_set):
    result_path = f"{args.result_dir}/{datetime.now().strftime('%Y-%m-%d-%H')}_{args.model.replace('/', '_')}_{args.exp_name}.json"
    agent = Drafter_agent(
        args.model, args.model_provider, args.temperature, args.vllm_url
    )
    indx = task_sets.index(task_set) + 1
    ground_codes = ground[task_set]
    structured_instructions = structured[task_set]
    unstructured_instructions = unstructured[task_set]
    rang = range(1, 3) if args.debug else range(1, 81)

    def get_response(instruction):
        prompt = Prommt(str(indx), instruction)
        pre_code = agent.get_response(messages=prompt.message())
        test_code = process_code(pre_code)
        prompt_info = execute_code(test_code)
        prompt_score = (
            Score_builder()
            .ground_fill(ground_details)
            .result(
                *testf.cross_check(
                    ground_info,
                    prompt_info,
                    "precise" if i <= 40 else "vaguely",
                )
            )
            if prompt_info
            else Score_builder().ground_fill(ground_details).fail()
        )
        prompt_result = collect_result(
            task_set,
            i,
            instruction,
            ground_code,
            pre_code,
            prompt_score,
        )
        return prompt_result

    if args.task_group in ["structured", "unstructured"]:
        target_task = [args.task_group]
    else:
        target_task = ["structured", "unstructured"]
    for i in rang:
        ground_code = ground_codes[i - 1][f"Code{i}"]
        test_ground_code = process_code(ground_code)
        ground_info = execute_code(test_ground_code)
        ground_details = testf.groundcheck(ground_info)
        if "structured" in target_task:
            str_prompt_result = get_response(
                structured_instructions[i - 1][f"Instruction{i}"]
            )
            results[0].append(str_prompt_result)
        if "unstructured" in target_task:
            ustr_prompt_result = get_response(
                unstructured_instructions[i - 1][f"Instruction{i}"]
            )
            results[1].append(ustr_prompt_result)
        if args.task_group == "structured":
            data = list(results[0])
        elif args.task_group == "unstructured":
            data = list(results[1])
        else:
            data = [list(results[0]), list(results[1])]
        savedate(data, result_path)
    return results
