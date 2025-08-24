#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert BFCL results to the specified format.
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any
import glob
import argparse

# Mapping dictionary from model_path to model_name
model_path_to_name = {
    "xai/grok-4": "grok-4",
    "togetherai/moonshotai/Kimi-K2-Instruct": "Kimi-K2-Instruct",
    "togetherai/Qwen/Qwen3-8B": "Qwen3-8B",
    "togetherai/Qwen/Qwen3-32B": "Qwen3-32B",
    "togetherai/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8": "Qwen3-235B-A22B-Thinking-2507-FP8",
    "togetherai/Qwen/Qwen3-235B-A22B-FP8": "Qwen3-235B-A22B-FP8",
    "togetherai/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8": "Qwen3-235B-A22B-Instruct-2507-FP8",
    "openai/o4-mini-high": "o4-mini-high",
    "openai/o3-high": "o3-high",
    "openai/gpt-4o-20240806": "gpt-4o-20240806",
    "openai/gpt-4o-mini": "gpt-4o-mini",
    "openai/gpt-4.1": "gpt-4.1",
    "deepseek-ai/DeepSeek-V3-0324": "DeepSeek-V3-0324",
    "deepseek-ai/DeepSeek-R1-0528": "DeepSeek-R1-0528",
    "anthropic/claude-4-sonnet-thinking-on-10k": "claude-4-sonnet-thinking-on-10k",
    "anthropic/claude-4-sonnet-thinking-off": "claude-4-sonnet-thinking-off"
}


def extract_model_path_from_filename(filename: str) -> str:
    """Extract model path from the filename."""
    # Example: "openai_gpt-4o-20240806" -> "openai/gpt-4o-20240806"
    if  "qwen" in filename.lower():
        if "32b" in filename.lower():
            return "togetherai/Qwen/Qwen3-32B"
        elif "8b" in filename.lower():
            return "togetherai/Qwen/Qwen3-8B"
    return filename.replace('_', '/')


def extract_task_type_from_filename(filename: str) -> str:
    """Extract task type from the filename."""
    # Example: "BFCL_v3_simple_result.json" -> "simple"
    # Example: "BFCL_v3_multi_turn_base_result.json" -> "multi_turn_base"
    if filename.startswith("BFCL_v3_"):
        task_part = filename[8:]  # Remove "BFCL_v3_"
        if task_part.endswith("_result.json"):
            return task_part[:-12]  # Remove "_result.json"
    return "unknown"


def convert_bfcl_file_with_results(input_file: str, output_dir: str, model_path: str = None, benchmark_name: str = "bfcl", score_data: List[Dict[str, Any]] = None, prompt_data: Dict[str, Dict[str, Any]] = None):
    """Convert a BFCL results file to the target format and save by task type, return converted results."""
    
    # Read input file (BFCL files are JSONL format)
    results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    result_data = json.loads(line)
                    results.append(result_data)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON line: {line[:100]}...")
                    continue
    
    # Extract model path from filename if not provided
    if model_path is None:
        dir_name = os.path.basename(os.path.dirname(input_file))
        model_path = extract_model_path_from_filename(dir_name)
    
    # Get model_name from mapping
    model_name = model_path_to_name[model_path]
    
    # Extract task type from filename
    filename = os.path.basename(input_file)
    task_type = extract_task_type_from_filename(filename)
    
    # Create output directory structure
    output_base_dir = os.path.join(output_dir, benchmark_name, model_path.replace('/', '_'))
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Convert results
    converted_results = []
    for result_data in results:
        # Find corresponding score data
        result_id = result_data.get("id", "")
        score_info = None
        if score_data:
            for score_item in score_data:
                if score_item.get("id") == result_id:
                    score_info = score_item
                    break
        
        # If no score info found, assume it's correct (score = 1.0)
        # This happens because score files only contain failed samples
        if score_info is None:
            # Create a default score info with valid=True for successful samples
            score_info = {
                "valid": True,
                "error": [],
                "error_type": ""
            }
        
        converted_result = convert_bfcl_result(result_data, model_path, task_type, benchmark_name, score_info, prompt_data)
        converted_results.append(converted_result)
    
    # Save as jsonl file
    output_file = os.path.join(output_base_dir, f"{model_name}_{task_type}.jsonl")
    
    # Write as jsonl (one JSON object per line)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in converted_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(converted_results)} results from {input_file} to {output_file}")
    
    return converted_results


def convert_bfcl_result(result_data: Dict[str, Any], model_path: str, task_type: str, benchmark_name: str = "bfcl", score_info: Dict[str, Any] = None, prompt_data: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convert a single BFCL result to the target format."""
    
    # Construct the converted result
    converted_result = {
        "model_path": model_path,
        "user_model_path": None,
        "benchmark_name": benchmark_name,
        "task_name": task_type,
        "sampling_params": {
            "max_tokens": 4096,
            "temperature": 0.0 if model_path != "anthropic/claude-4-sonnet-thinking-on-10k" else 1.0,
        },
        "user_sampling_params": {},
        "messages": construct_messages_from_bfcl_result(result_data, task_type, score_info, prompt_data),
        "eval_result": {
            "score": 1.0 if score_info and score_info.get("valid", False) else 0.0,  # Use score info if available
        },
        "meta": {
            "id": result_data.get("id", ""),
            "task_type": task_type,
            "valid": score_info.get("valid", False) if score_info else result_data.get("valid", False),
            "error": score_info.get("error", []) if score_info else result_data.get("error", []),
            "error_type": score_info.get("error_type", "") if score_info else result_data.get("error_type", ""),
            "result": result_data.get("result", ""),
                            "model_result_raw": score_info.get("model_result_raw", "") if score_info else "",
            "model_result_decoded": result_data.get("model_result_decoded", ""),
            "possible_answer": result_data.get("possible_answer", ""),
            "input_token_count": result_data.get("input_token_count", 0),
            "output_token_count": result_data.get("output_token_count", 0),
            "latency": result_data.get("latency", 0),
            "inference_log": result_data.get("inference_log", []),
        }
    }
    
    return converted_result


def construct_messages_from_bfcl_result(result_data: Dict[str, Any], task_type: str, score_info: Dict[str, Any] = None, prompt_data: Dict[str, Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Construct messages array from BFCL result data."""
    messages = []
    
    # Add system message - use BFCL's default system prompt
    system_prompt = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task."""
    
    messages.append({
        "role": "system",
        "content": system_prompt
    })
    
    # Add user message from the prompt - try to get from prompt_data first, then from score_info, then from result_data
    user_content = ""
    result_id = result_data.get("id", "")
    
    # Try to get from prompt_data first (most reliable)
    if prompt_data and result_id in prompt_data:
        prompt_item = prompt_data[result_id]
        if "question" in prompt_item and prompt_item["question"]:
            # Extract user message from the question
            user_content = prompt_item["question"][0][0]["content"]
            # Add function information if available
            if "function" in prompt_item:
                user_content += "\n\nAvailable functions:\n"
                for func in prompt_item["function"]:
                    user_content += f"- {func['name']}: {func['description']}\n"
                    if "parameters" in func and "properties" in func["parameters"]:
                        user_content += "  Parameters:\n"
                        for param_name, param_info in func["parameters"]["properties"].items():
                            user_content += f"    - {param_name}: {param_info.get('description', 'No description')}\n"
    elif score_info and "prompt" in score_info:
        prompt_data = score_info["prompt"]
        if "question" in prompt_data and prompt_data["question"]:
            # Extract user message from the question
            user_content = prompt_data["question"][0][0]["content"]
            # Add function information if available
            if "function" in prompt_data:
                user_content += "\n\nAvailable functions:\n"
                for func in prompt_data["function"]:
                    user_content += f"- {func['name']}: {func['description']}\n"
                    if "parameters" in func and "properties" in func["parameters"]:
                        user_content += "  Parameters:\n"
                        for param_name, param_info in func["parameters"]["properties"].items():
                            user_content += f"    - {param_name}: {param_info.get('description', 'No description')}\n"
    elif "prompt" in result_data:
        prompt_data = result_data["prompt"]
        if "question" in prompt_data and prompt_data["question"]:
            # Extract user message from the question
            user_content = prompt_data["question"][0][0]["content"]
            # Add function information if available
            if "function" in prompt_data:
                user_content += "\n\nAvailable functions:\n"
                for func in prompt_data["function"]:
                    user_content += f"- {func['name']}: {func['description']}\n"
                    if "parameters" in func and "properties" in func["parameters"]:
                        user_content += "  Parameters:\n"
                        for param_name, param_info in func["parameters"]["properties"].items():
                            user_content += f"    - {param_name}: {param_info.get('description', 'No description')}\n"
    
    if user_content:
        messages.append({
            "role": "user",
            "content": user_content
        })
    
    # Add assistant response - use the original raw result from score_info if available, otherwise from result_data
    assistant_content = ""
    if score_info and "model_result_raw" in score_info:
        assistant_content = score_info["model_result_raw"]
    elif "model_result_raw" in result_data:
        assistant_content = result_data["model_result_raw"]
    elif "result" in result_data:
        assistant_content = result_data["result"]
    else:
        assistant_content = ""
    
    messages.append({
        "role": "assistant",
        "content": assistant_content,
    })
    
    return messages


def load_score_data(score_dir: str, model_name: str, task_type: str) -> Dict[str, Any]:
    """Load score data for a specific model and task type."""
    score_file = os.path.join(score_dir, model_name, f"BFCL_v3_{task_type}_score.json")
    if os.path.exists(score_file):
        with open(score_file, 'r', encoding='utf-8') as f:
            score_data = []
            for line in f:
                try:
                    score_data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
            return score_data
    return []


def load_prompt_data(task_type: str) -> Dict[str, Dict[str, Any]]:
    """Load prompt data for a specific task type."""
    prompt_file = os.path.join("gorilla/berkeley-function-call-leaderboard/bfcl_eval/data", f"BFCL_v3_{task_type}.json")
    if os.path.exists(prompt_file):
        prompt_data = {}
        with open(prompt_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                prompt_data[data["id"]] = data
        return prompt_data
    return {}


def convert_bfcl_results(bfcl_dir: str, output_dir: str, score_dir: str = None):
    """Convert BFCL results."""
    print("Converting BFCL results...")
    
    # Find all model directories
    model_dirs = [d for d in os.listdir(bfcl_dir) if os.path.isdir(os.path.join(bfcl_dir, d))]
    
    for model_dir in model_dirs:
        model_path = extract_model_path_from_filename(model_dir)
        model_name = model_path_to_name.get(model_path, model_path.replace('/', '_'))
        model_result_dir = os.path.join(bfcl_dir, model_dir)
        
        # Find all result files in this model directory
        result_files = glob.glob(os.path.join(model_result_dir, "BFCL_v3_*_result.json"))
        
        # Collect all results for this model
        all_model_results = []
        
        for result_file in result_files:
            try:
                # Extract task type from filename
                filename = os.path.basename(result_file)
                task_type = extract_task_type_from_filename(filename)
                
                # Load score data if available
                score_data = {}
                if score_dir:
                    score_data = load_score_data(score_dir, model_dir, task_type)
                
                # Load prompt data
                prompt_data = load_prompt_data(task_type)
                
                # Convert and collect results
                converted_results = convert_bfcl_file_with_results(result_file, output_dir, model_path, "bfcl", score_data, prompt_data)
                all_model_results.extend(converted_results)
            except Exception as e:
                print(f"Error converting {result_file}: {e}")
        
        # Save combined file with all results for this model
        if all_model_results:
            output_base_dir = os.path.join(output_dir, "bfcl", model_path.replace('/', '_'))
            combined_output_file = os.path.join(output_base_dir, f"{model_name}.jsonl")
            with open(combined_output_file, 'w', encoding='utf-8') as f:
                for result in all_model_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            print(f"Saved {len(all_model_results)} total results to combined file {combined_output_file}")


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Convert BFCL results to specified format')
    parser.add_argument('input_dir', help='BFCL results directory (e.g., result)')
    parser.add_argument('output_dir', help='Output directory path')
    parser.add_argument('--score-dir', help='BFCL score directory (e.g., score)')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert the results
    convert_bfcl_results(args.input_dir, args.output_dir, args.score_dir)
    
    print("Conversion completed successfully!")


if __name__ == "__main__":
    main()
