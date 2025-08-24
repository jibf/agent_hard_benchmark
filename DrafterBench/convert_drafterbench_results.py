#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert DrafterBench results to the specified format.
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any

# Import modules directly since we're now in the DrafterBench directory
from methods import task_sets

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


def load_backend_prompts():
    """Load backend prompts manually to avoid path issues."""
    prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')
    
    def open_file(filepath):
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()
    
    backend_prompts = {}
    prompt_files = [
        "add_table.txt", "revise_table.txt", "map_table.txt", "refresh_table.txt",
        "add_text.txt", "revise_text.txt", "map_text.txt", "refresh_text.txt",
        "add_vector.txt", "delete_vector.txt", "map_vector.txt", "refresh_vector.txt"
    ]
    
    for i, filename in enumerate(prompt_files, 1):
        filepath = os.path.join(prompts_dir, filename)
        if os.path.exists(filepath):
            backend_prompts[str(i)] = open_file(filepath)
        else:
            backend_prompts[str(i)] = f"System prompt for task type {filename[:-4]}"
    
    return backend_prompts


# Load backend prompts manually to avoid path issues
Backend_prompt = load_backend_prompts()


def get_system_message(task_type: str) -> str:
    """Get the system message for a given task type."""
    try:
        task_index = str(task_sets.index(task_type) + 1)
        return Backend_prompt[task_index]
    except (ValueError, KeyError):
        return "You are a helpful assistant for PDF editing tasks."


def construct_messages(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Construct messages array based on the task."""
    system_message = get_system_message(task.get("Tasktype", ""))
    user_instruction = task.get("Instruction", "")
    
    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user", 
            "content": user_instruction
        },
        {
            "role": "assistant",
            "content": task.get("Response_code", ""),
        }
    ]
    
    return messages

def extract_model_path_from_filename(filename: str) -> str:
    """Extract model path from the filename."""
    # Example: "openai_gpt-4o-20240806" -> "openai/gpt-4o-20240806"
    parts = filename.split('_')
    if len(parts) >= 2:
        provider = parts[0]
        model_name = '_'.join(parts[1:])
        return f"{provider}/{model_name}"
    return filename


def convert_drafterbench_result(task: Dict[str, Any], model_path: str, benchmark_name: str = "drafterbench") -> Dict[str, Any]:
    """Convert a single DrafterBench result to the target format."""
    
    # Construct the converted result
    converted_result = {
        "model_path": model_path,
        "user_model_path": None,
        "benchmark_name": benchmark_name,
        "task_name": task.get("Tasktype", ""),  # Use Tasktype as task_name
        "sampling_params": {
            "max_tokens": 4096,  # Default value
            "temperature": 0.0 if model_path != "anthropic/claude-4-sonnet-thinking-on-10k" else 1.0,
        },
        "user_sampling_params": {},
        "messages": construct_messages(task),
        "eval_result": {
            "score": task["Task_score"]["Task_score"],
        },
        "meta": {
            "id": str(task.get("Id", "")),
            "task_type": task.get("Tasktype", ""),
            "precise_vague": task.get("Precise|Vague", ""),
            "complete_incomplete": task.get("Complete|Incomplete", ""),
            "single_multiple_objects": task.get("Single|Multiple_objects", ""),
            "single_multiple_operations": task.get("Single|Multiple_operations", ""),
            "structured_unstructured": task.get("Structured/Unstructured", ""),
            "instruction": task.get("Instruction", ""),
            "groundtruth": task.get("Groundtruth", ""),
            "response_code": task.get("Response_code", ""),
            # Include additional evaluation metrics
            "success_arguments_define": task.get("Task_score", {}).get("Success_arguments_define", 0),
            "total_arguments_define": task.get("Task_score", {}).get("Total_arguments_define", 0),
            "success_variable_transfer": task.get("Task_score", {}).get("Success_variable_transfer", 0),
            "total_variable_transfer": task.get("Task_score", {}).get("Total_variable_transfer", 0),
            "success_function_calling": task.get("Task_score", {}).get("Success_function_calling", 0),
            "total_function_calling": task.get("Task_score", {}).get("Total_function_calling", 0),
            "success_single_tool_selection": task.get("Task_score", {}).get("Success_single_tool_selection", 0),
            "total_single_tool_selection": task.get("Task_score", {}).get("Total_single_tool_selection", 0),
            "success_multi_tool_selection": task.get("Task_score", {}).get("Success_multi_tool_selection", 0),
            "total_multi_tool_selection": task.get("Task_score", {}).get("Total_multi_tool_selection", 0),
            "intersected_plan_execution": task.get("Task_score", {}).get("Intersected_plan_execution", 0),
            "total_plans_appeared": task.get("Task_score", {}).get("Total_plans_appeared", 0),
            "ground_plan_execution": task.get("Task_score", {}).get("Ground_plan_execution", 0)
        }
    }
    
    return converted_result


def convert_drafterbench_file(input_file: str, output_dir: str, model_path: str = None, benchmark_name: str = "drafterbench"):
    """Convert a DrafterBench results file to the target format and save by subtask."""
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract model path from filename if not provided
    if model_path is None:
        filename = os.path.basename(input_file)
        # Extract from directory structure: openai_gpt-4o-20240806/2025-07-27-10-27_All.json
        dir_name = os.path.basename(os.path.dirname(input_file))
        model_path = extract_model_path_from_filename(dir_name)
    
    # Get model_name from mapping
    model_name = model_path_to_name[model_path]
    
    # Group results by subtask
    subtask_groups = {}
    all_results = []  # Collect all results for the combined file
    for task in data:
        subtask = task.get("Tasktype", "unknown")
        if subtask not in subtask_groups:
            subtask_groups[subtask] = []
        converted_result = convert_drafterbench_result(task, model_path, benchmark_name)
        subtask_groups[subtask].append(converted_result)
        all_results.append(converted_result)
    
    # Create output directory structure
    output_base_dir = os.path.join(output_dir, benchmark_name, model_path.replace('/', '_'))
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Save each subtask to separate jsonl file
    total_converted = 0
    for subtask, results in subtask_groups.items():
        output_file = os.path.join(output_base_dir, f"{model_name}_{subtask}.jsonl")
        
        # Write as jsonl (one JSON object per line)
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(results)} results for subtask '{subtask}' to {output_file}")
        total_converted += len(results)
    
    # Save combined file with all results for this model
    combined_output_file = os.path.join(output_base_dir, f"{model_name}.jsonl")
    with open(combined_output_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(all_results)} total results to combined file {combined_output_file}")
    print(f"Total: Converted {total_converted} results from {input_file} to {len(subtask_groups)} subtask files in {output_base_dir}")


def main():
    """Main function to handle command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert DrafterBench results to specified format')
    parser.add_argument('input_file', help='Input DrafterBench results file')
    parser.add_argument('output_dir', help='Output directory path')
    parser.add_argument('--model-path', help='Model path (e.g., openai/gpt-4o-20240806)')
    parser.add_argument('--benchmark-name', default='DrafterBench', help='Benchmark name')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        sys.exit(1)
    
    # Convert the file
    convert_drafterbench_file(
        args.input_file, 
        args.output_dir, 
        args.model_path, 
        args.benchmark_name
    )


if __name__ == "__main__":
    main()
