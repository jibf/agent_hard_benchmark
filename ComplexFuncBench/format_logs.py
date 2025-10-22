import os
import json
from dataclasses import dataclass, asdict
from tqdm import tqdm

BENCHMARK_NAME = "complex-func-bench"
INPUT_BASE_DIR = "result/"
OUTPUT_BASE_DIR = f"formatted_logs/{BENCHMARK_NAME}-evaluation/"

@dataclass
class FormattedLog:
    # Mandatory fields
    model_path: str
    benchmark_name: str
    sampling_params: dict
    messages: list
    eval_result: dict

    # Optional fields
    user_model_path: str = None 
    task_name: str = ""
    user_sampling_params: dict = None
    meta: dict = None


def process_log_entry(model_path: str, log_entry: dict) -> dict:
    """
    Convert logs into following format ###
    {
        "model_path": "openai/gpt-4o-20240806", #mandatory
        "user_model_path": "openai/gpt-4o-20240806", #optional
        "benchmark_name": "tau2-bench", #mandatory
        "task_name": "airline", #optional, add if there're mutliple subtasks
        "sampling_params": { #mandatory
            "max_tokens": 16384,
            "temperature": 0.0
        },
        "user_sampling_params": { #optional
            "temperature": 1.0
        },
        "messages": [ #mandatory (full trajectory)
            {
                "role": "user",
                "content": ...,
            },
            {
                "role": "assistant",
                "content": null,
                "turn_idx": 1,
                "tool_calls": [...]
            },
        ],
        "eval_result": { #mandatory
            "score": 1, #mandatory - We always use score to represent the final evaluation result for this individual prompt, it needs to be in between 0 and 1.
            # --- Any other data that might be related to the evaluation.
            "bleu": 0.71
        },
        "meta": { # optional - anything else from the benchmark's results
            "id": "airline-task-atl-jfk-001",
            "is_correct": true,
            "finish_reason": "success",
            "run_timestamp": "2025-08-19T09:15:00Z",
            "task_description": {
                "purpose": "Book a one-way flight...",
                "user_scenario": {
                    "instructions": {
                        "domain": "airline",
                        "reason_for_call": "Book a flight",
                        "known_info": "User knows departure city, arrival city, and the desired day of travel.",
                        "task_instructions": "Find the cheapest flight that matches the user's criteria and confirm the booking."
                    }
                }
            }
        }
    }
    """
    sample_id = log_entry['id']
    task_name = '-'.join(sample_id.split("-")[:-1])  # e.g., Car-Rental, Cross

    # format messages
    messages = log_entry['gen_convs']
    for i in range(len(messages)):
        messages[i]['turn_idx'] = i
        if messages[i]['role'] == 'assistant':
            if "content" not in messages[i].keys():
                messages[i]['content'] = None
            if 'function_call' in messages[i].keys():
                messages[i]['tool_calls'] = messages[i].pop('function_call')

    # construct `eval_result`
    eval_result = {
        "score": 1.0 if log_entry['message'] == "Success." else 0.0,
        "message": log_entry.get('message', ""),
        "resp_eval": log_entry.get('resp_eval', {}),
    }

    is_claude_thinking = ("thinking-on" in model_path) and ("claude" in model_path)

    sampling_params = {
        'max_tokens': 2048,
        'temperature': 1.0 if is_claude_thinking else 0.0,
    }

    if is_claude_thinking:
        sampling_params['thinking'] = {
            "type": "enabled",
            "budget_tokens": 10000
        }

    meta = {
        "id": sample_id,
        "count_dict": log_entry.get('count_dict', {}),
    }

    formatted_log = FormattedLog(
        model_path=model_path, benchmark_name=BENCHMARK_NAME, messages=messages, sampling_params=sampling_params, eval_result=eval_result,
        meta=meta, task_name=task_name
    )

    return asdict(formatted_log)


model_paths = []

for path, dir, files in os.walk(INPUT_BASE_DIR):
    for file in files:
        if file == "full-1000.jsonl":
            model_path = '/'.join(path.split("/")[1:])  # e.g., openai/gpt-4.1
            model_paths.append(model_path)

for model_path in model_paths:
    print("Processing model:", model_path)
    formatted_logs_by_category = {}
    with open(os.path.join(INPUT_BASE_DIR, model_path, "full-1000.jsonl"), "r") as f:
        for line in tqdm(f):
            formatted_log_entry = process_log_entry(model_path, json.loads(line))
            if formatted_log_entry['task_name'] not in formatted_logs_by_category:
                formatted_logs_by_category[formatted_log_entry['task_name']] = []
            formatted_logs_by_category[formatted_log_entry['task_name']].append(formatted_log_entry)

    for task_name, formatted_logs in formatted_logs_by_category.items():
        formatted_logs.sort(key=lambda x: int(x['meta']['id'].split("-")[-1]))  # Sort by sample ID
        output_path = os.path.join(OUTPUT_BASE_DIR, f"{model_path.split('/')[-1]}_{task_name.lower()}.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as out_file:
            for log in formatted_logs:
                out_file.write(json.dumps(log) + "\n")
