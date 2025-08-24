#!/usr/bin/env python3
"""
Convert Multi-Challenge results to unified format.

This script converts Multi-Challenge detailed results from CSV format to the unified JSONL format.
"""

import os
import json
import csv
import argparse
from typing import Dict, List, Any
from datetime import datetime


def extract_model_path_from_filename(filename: str) -> str:
    """Extract model path from the filename."""
    # Remove file extension and common suffixes
    base_name = filename.replace('_evaluation_results.txt', '').replace('_detailed_results.csv', '')
    
    # Map model names to standard format
    model_mapping = {
        'gpt4o_20240806': 'openai/gpt-4o-20240806',
        'gpt4o_mini': 'openai/gpt-4o-mini',
        'gpt_o4_mini_high': 'openai/o4-mini-high',
        'gpt_o3_high': 'openai/o3-high',
        'gpt4_1': 'openai/gpt-4.1',
        'deepseekv3': 'deepseek-ai/DeepSeek-V3-0324',
        'deepseekR1': 'deepseek-ai/DeepSeek-R1-0528',
        'claude4_sonnect': 'anthropic/claude-4-sonnet-thinking-off',
        'claude4_thinking_sonnect': 'anthropic/claude-4-sonnet-thinking-on-10k',
        'grok4': 'xai/grok-4',
        'kimi_k2': 'togetherai/moonshotai/Kimi-K2-Instruct',
        'Qwen3-8B': 'Qwen/Qwen3-8B',
        'Qwen3-32B': 'Qwen/Qwen3-32B',
        'qwen3-235B': 'Qwen/Qwen3-235B',
        'Qwen235B_instruct': 'togetherai/Qwen_Qwen3-235B-A22B-Instruct-2507-FP8',
        'qwen_thinking': 'togetherai/Qwen_Qwen3-235B-A22B-Thinking-2507-FP8',
    }
    
    return model_mapping[base_name]


def load_original_data() -> Dict[str, Dict[str, Any]]:
    """Load original benchmark data to get conversation context."""
    original_data = {}
    data_file = './data/benchmark_questions.jsonl'
    
    if os.path.exists(data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                original_data[data['QUESTION_ID']] = data
    
    return original_data


def convert_multi_challenge_result(row: Dict[str, Any], model_path: str, original_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Convert a single Multi-Challenge result row to the target format."""
    
    question_id = row['question_id']
    original_item = original_data.get(question_id, {})
    
    # Extract conversation from original data
    messages = original_item['CONVERSATION'].copy()
    
    messages.append({
        "role": "assistant",
        "content": row["model_response"]
    })
    
    # Calculate score based on passed field
    score = 1.0 if row.get('passed', '').lower() == 'true' else 0.0
    
    return {
        "model_path": model_path,
        "user_model_path": None,
        "benchmark_name": "multi_challenge",
        "task_name": row.get('axis', 'unknown'),
        "sampling_params": {
            "max_tokens": 4096,  # Default for multi_challenge
            "temperature": 0.0
        },
        "user_sampling_params": {},
        "messages": messages,
        "eval_result": {
            "score": score
        },
        "meta": {
            "id": question_id,
            "axis": row.get('axis', ''),
            "target_question": row.get('target_question', ''),
            "pass_criteria": row.get('pass_criteria', ''),
            "attempt_number": int(row.get('attempt_number')),
            "judge_verdict": row.get('judge_verdict', ''),
            "passed": row.get('passed', ''),
            "reasoning": row.get('reasoning', ''),
            "final_result": row.get('final_result', ''),
            "original_conversation": row.get('original_conversation', ''),
        }
    }


def convert_multi_challenge_file(input_file: str, output_dir: str, model_path: str = None, original_data: Dict[str, Dict[str, Any]] = None):
    """Convert a Multi-Challenge detailed results file to the target format."""
    
    # Extract model path from filename if not provided
    if model_path is None:
        filename = os.path.basename(input_file)
        model_path = extract_model_path_from_filename(filename)
    
    # Create output directory structure
    output_base_dir = os.path.join(output_dir, "multi_challenge", model_path.replace('/', '_'))
    os.makedirs(output_base_dir, exist_ok=True)

    results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    
    # Convert results
    converted_results = []
    for result_data in results:
        converted_result = convert_multi_challenge_result(result_data, model_path, original_data or {})
        converted_results.append(converted_result)
    
    task_results = {}
    for result in converted_results:
        task_name = result["task_name"]
        if task_name not in task_results:
            task_results[task_name] = []
        task_results[task_name].append(result)

    for task_name in task_results:
        # Save as jsonl file
        output_file = os.path.join(output_base_dir, f"{model_path.replace('/', '_')}_{task_name}.jsonl")

        # Write as jsonl (one JSON object per line)
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in task_results[task_name]:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
        print(f"Converted {len(task_results[task_name])} results from {input_file} to {output_file}")


def convert_multi_challenge_results(results_dir: str, output_dir: str):
    """Convert all Multi-Challenge results files."""
    print("Converting Multi-Challenge results...")
    
    # Load original data for conversation context
    original_data = load_original_data()
    
    # Find all detailed results CSV files
    csv_files = []
    for filename in os.listdir(results_dir):
        if filename.endswith('_detailed_results.csv'):
            csv_files.append(os.path.join(results_dir, filename))
    
    if not csv_files:
        print("No detailed results CSV files found!")
        return
    
    # Convert each file
    for csv_file in csv_files:
        convert_multi_challenge_file(csv_file, output_dir, original_data=original_data)
    
    print("Multi-Challenge conversion completed successfully!")


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Convert Multi-Challenge results to specified format')
    parser.add_argument('results_dir', help='Multi-Challenge results directory (e.g., multi_challenge/results)')
    parser.add_argument('output_dir', help='Output directory path')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' does not exist.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert the results
    convert_multi_challenge_results(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
