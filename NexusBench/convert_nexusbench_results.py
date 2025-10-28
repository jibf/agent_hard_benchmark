#!/usr/bin/env python3
"""
Convert NexusBench results to unified format.

This script converts NexusBench detailed results from parquet format to the unified JSONL format.
"""

import re
import os
import json
import argparse
import pandas as pd
import ast
from typing import Dict, List, Any
from datetime import datetime
from typing import Any, Callable, Dict


# Mapping dictionary from model_path to model_name
model_path_to_name = {
    "xai/grok-4": "grok-4",
    "togetherai/moonshotai/Kimi-K2-Instruct": "Kimi-K2-Instruct",
    "togetherai/moonshotai/Kimi-K2-Instruct-0905": "Kimi-K2-Instruct-0905",
    "togetherai/Qwen/Qwen3-8B": "Qwen3-8B",
    "togetherai/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": "Qwen3-Coder-480B-A35B-Instruct-FP8",
    "togetherai/Qwen/Qwen3-32B": "Qwen3-32B",
    "togetherai/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8": "Qwen3-235B-A22B-Thinking-2507-FP8",
    "togetherai/Qwen/Qwen3-235B-A22B-FP8": "Qwen3-235B-A22B-FP8",
    "togetherai/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8": "Qwen3-235B-A22B-Instruct-2507-FP8",
    "togetherai/openai/gpt-oss-20b": "gpt-oss-20b",
    "togetherai/openai/gpt-oss-120b": "gpt-oss-120b",
    "openai/o4-mini-high": "o4-mini-high",
    "openai/o3-high": "o3-high",
    "openai/gpt-4o-20240806": "gpt-4o-20240806",
    "openai/gpt-4o-mini": "gpt-4o-mini",
    "openai/gpt-4.1": "gpt-4.1",
    "openai/gpt-4.1-mini": "gpt-4.1-mini",
    "openai/gpt-4.1-nano": "gpt-4.1-nano",
    "openai/gpt-5": "gpt-5",
    "openai/gpt-5-nano": "gpt-5-nano",
    "deepseek-ai/DeepSeek-V3-0324": "DeepSeek-V3-0324",
    "deepseek-ai/DeepSeek-R1-0528": "DeepSeek-R1-0528",
    "deepseek-ai/DeepSeek-V3.1-Terminus-thinking-off": "DeepSeek-V3.1-thinking-off",
    "deepseek-ai/DeepSeek-V3.1-Terminus-thinking-on": "DeepSeek-V3.1-thinking-on",
    "deepseek-ai/DeepSeek-V3.2-Exp-thinking-off": "DeepSeek-V3.2-thinking-off",
    "deepseek-ai/DeepSeek-V3.2-Exp-thinking-on": "DeepSeek-V3.2-thinking-on",
    "anthropic/claude-4-sonnet-thinking-on-10k": "claude-4-sonnet-thinking-on-10k",
    "anthropic/claude-4-sonnet-thinking-off": "claude-4-sonnet-thinking-off",
    "anthropic/claude-4.5-sonnet-thinking-on-10k": "claude-4.5-sonnet-thinking-on-10k",
    "anthropic/claude-4.5-sonnet-thinking-off": "claude-4.5-sonnet-thinking-off",
    "anthropic/claude-4-opus-thinking-on-10k": "claude-4-opus-thinking-on-10k",
    "anthropic/claude-4-opus-thinking-off": "claude-4-opus-thinking-off",
    "google/gemini-2.5-flash-thinking-off": "gemini-2.5-flash-thinking-off",
    "google/gemini-2.5-flash-thinking-on": "gemini-2.5-flash-thinking-on",
    "google/gemini-2.5-pro-thinking-on": "gemini-2.5-pro-thinking-on"
}

prompt_to_id = {}

def extract_model_path_from_filename(filename: str) -> str:
    """Extract model path from the filename."""
    # Remove file extension and common suffixes
    base_name = filename.replace('.parquet', '')
    
    # Extract model name from filename pattern: TaskName_Date_ModelName-OpenAIFC
    parts = base_name.split('_')
    if len(parts) >= 3:
        # Find the model part (usually after the date)
        model_part = '_'.join(parts[2:])  # Skip task name and date
        model_part = model_part.replace('-OpenAIFC', '')  # Remove suffix
        
        # Map model names to standard format
        model_mapping = {
            'togetherai-Qwen-Qwen3-235B-A22B-FP8': 'togetherai/Qwen/Qwen3-235B-A22B-FP8',
            'togetherai-Qwen-Qwen3-235B-A22B-Thinking-2507-FP8': 'togetherai/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8',
            'togetherai-Qwen-Qwen3-235B-A22B-Instruct-2507-FP8': 'togetherai/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8',
            'Qwen3-8B-QwenFC': 'togetherai/Qwen/Qwen3-8B',
            'Qwen3-32B-QwenFC': 'togetherai/Qwen/Qwen3-32B',
            'togetherai-moonshotai-Kimi-K2-Instruct': 'togetherai/moonshotai/Kimi-K2-Instruct',
            'togetherai-m-Kimi-K2-Instruct-0905': 'togetherai/moonshotai/Kimi-K2-Instruct-0905',
            'togetherai-openai-gpt-oss-20b': 'togetherai/openai/gpt-oss-20b',
            'togetherai-openai-gpt-oss-120b': 'togetherai/openai/gpt-oss-120b',
            'deepseek-ai-DeepSeek-R1-0528': 'deepseek-ai/DeepSeek-R1-0528',
            'deepseek-ai-DeepSeek-V3-0324': 'deepseek-ai/DeepSeek-V3-0324',
            'd-DeepSeek-V3.1-Terminus-thinking-off': 'deepseek-ai/DeepSeek-V3.1-Terminus-thinking-off',
            'd-DeepSeek-V3.1-Terminus-thinking-on': 'deepseek-ai/DeepSeek-V3.1-Terminus-thinking-on',
            'd-DeepSeek-V3.2-Exp-thinking-off': 'deepseek-ai/DeepSeek-V3.2-Exp-thinking-off',
            'd-DeepSeek-V3.2-Exp-thinking-on': 'deepseek-ai/DeepSeek-V3.2-Exp-thinking-on',
            'openai-o3-high': 'openai/o3-high',
            'openai-o4-mini-high': 'openai/o4-mini-high',
            'openai-gpt-4o-20240806': 'openai/gpt-4o-20240806',
            'openai-gpt-4.1': 'openai/gpt-4.1',
            'openai-gpt-4.1-mini': 'openai/gpt-4.1-mini',
            'openai-gpt-4.1-nano': 'openai/gpt-4.1-nano',
            'openai-gpt-5': 'openai/gpt-5',
            'openai-gpt-5-nano': 'openai/gpt-5-nano',
            'a-claude-4.5-sonnet-thinking-on-10k': 'anthropic/claude-4.5-sonnet-thinking-on-10k',
        }
        # Return mapped value if explicitly provided above
        if model_part in model_mapping:
            return model_mapping[model_part]

        # Generic handling: convert provider-prefixed strings like
        #   "openai-gpt-4.1-mini"  -> "openai/gpt-4.1-mini"
        #   "anthropic-claude-4-sonnet-thinking-off" -> "anthropic/claude-4-sonnet-thinking-off"
        for provider in ("openai", "anthropic", "deepseek-ai", "togetherai", "google", ""):
            prefix = f"{provider}-"
            if model_part.startswith(prefix):
                return f"{provider}/" + model_part[len(prefix):]

        # Fallback: return as-is
        return model_part
    
    return "unknown"


def extract_task_name_from_filename(filename: str) -> str:
    """Extract task name from the filename."""
    # Remove file extension
    base_name = filename.replace('.parquet', '')
    
    # Extract task name (first part before underscore)
    parts = base_name.split('_')
    if parts:
        return parts[0]
    
    return "unknown"


def parse_messages_from_prompt(prompt_str: str) -> List[Dict[str, Any]]:
    """Parse messages from prompt string."""

    try:
        prompt_dict = ast.literal_eval(prompt_str)
    except:
        prompt_dict = parse_payload_to_dict(prompt_str)
    return prompt_dict


def convert_nexusbench_result(row: Dict[str, Any], model_path: str, task_name: str) -> Dict[str, Any]:
    """Convert a single NexusBench result row to the target format."""
    
    # Build complete conversation from multiple prompts and responses
    messages = []
    
    # Check for multiple prompts (prompt_0, prompt_1, prompt_2, etc.)
    prompt_index = 0
    while f'query_{prompt_index}' in row and row[f'query_{prompt_index}'] != "None":
        query_str = row[f'query_{prompt_index}']
        response_str = row[f'response_{prompt_index}']
        messages.append({
            "role": "user",
            "content": query_str,
        })
        messages.append({
            "role": "assistant",
            "content": response_str,
        })
        prompt_index += 1
    
    prompt_str = row['prompt_0']
    prompt_dict = parse_messages_from_prompt(prompt_str)
    tools = prompt_dict.get("tools", [])

    roles = []
    for item in messages:
        roles.append(item["role"])
    assert "user" in roles and "assistant" in roles
    
    # Calculate score based on Final Accuracy
    score = 1.0 if row.get('Final Accuracy', '').lower() == 'true' else 0.0
    
    # Extract ground truth
    ground_truth = row.get('ground_truth', '')
    ground_truth = ground_truth if ground_truth is not None else ''

    meta = {}
    for key, value in row.items():
        if key not in ['Final Accuracy', 'ground_truth'] and row[key] != "None":
            meta[key] = value
    
    quest_sigh = row['query_0']
    if task_name not in prompt_to_id:
        prompt_to_id[task_name] = {}
    
    if quest_sigh not in prompt_to_id[task_name]:
        prompt_to_id[task_name][quest_sigh] = len(prompt_to_id[task_name])

    # meta["tool_info"] = tool_info
    return {
        "model_path": model_path,
        "user_model_path": None,
        "benchmark_name": "nexusbench",
        "task_name": task_name,
        "sampling_params": {
            "max_tokens": 4096,  # Default for nexusbench
            "temperature": 0.0 if model_path != "anthropic/claude-4-sonnet-thinking-on-10k" else 1.0,
        },
        "user_sampling_params": {},
        "messages": messages,
        "eval_result": {
            "score": score
        },
        "meta": {
            "id": f"{task_name}_{prompt_to_id[task_name][quest_sigh]}",  # Use index as ID
            "task_name": task_name,
            "ground_truth": ground_truth,
            "tools": tools,
            **meta,
        }
    }


def convert_nexusbench_file(input_file: str, output_dir: str, model_path: str = None, task_name: str = None):
    """Convert a NexusBench parquet file to the target format."""
    
    # Extract model path and task name from filename if not provided
    filename = os.path.basename(input_file)
    if model_path is None:
        model_path = extract_model_path_from_filename(filename)
    if task_name is None:
        task_name = extract_task_name_from_filename(filename)
    
    # Get model_name from mapping
    model_name = model_path_to_name.get(model_path, model_path.replace('/', '_'))
    
    # Create output directory structure
    output_base_dir = os.path.join(output_dir, "nexusbench", model_path.replace('/', '_'))
    os.makedirs(output_base_dir, exist_ok=True)

    # Read parquet file
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")
    
    # Convert results
    converted_results = []
    for index, row in df.iterrows():
        row_dict = row.to_dict()
        converted_result = convert_nexusbench_result(row_dict, model_path, task_name)
        converted_results.append(converted_result)
    
    # Save as jsonl file
    output_file = os.path.join(output_base_dir, f"{model_name}_{task_name}.jsonl")

    # Write as jsonl (one JSON object per line)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in converted_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(converted_results)} results from {input_file} to {output_file}")
    
    return converted_results


def convert_nexusbench_results(parquets_dir: str, output_dir: str):
    """Convert all NexusBench parquet files."""
    print("Converting NexusBench results...")
    
    # Find all parquet files
    parquet_files = []
    for filename in os.listdir(parquets_dir):
        if filename.endswith('.parquet'):
            parquet_files.append(os.path.join(parquets_dir, filename))
    
    if not parquet_files:
        print("No parquet files found!")
        return
    
    # Group files by model to create combined files
    model_results = {}
    
    # Convert each file
    for parquet_file in parquet_files:

        converted_results = convert_nexusbench_file(parquet_file, output_dir)
        
        # Group by model for combined file
        if converted_results:
            model_path = converted_results[0]["model_path"]
            if model_path not in model_results:
                model_results[model_path] = []
            model_results[model_path].extend(converted_results)
    
    # Save combined files for each model
    for model_path, results in model_results.items():
        model_name = model_path_to_name.get(model_path, model_path)
        model_name = model_name.split('/', 1)[-1]
        output_base_dir = os.path.join(output_dir, "nexusbench", model_path.replace('/', '_'))
        print(output_base_dir)
        
        combined_output_file = os.path.join(output_base_dir, f"{model_name}.jsonl")
        with open(combined_output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(results)} total results to combined file {combined_output_file}")
    
    print("NexusBench conversion completed successfully!")


def _make_obj(type_name: str) -> Callable[..., Dict[str, Any]]:
    """
    Return a constructor that turns calls like TypeName(a=1, b=2, x)
    into {'__type__': 'TypeName', 'args': [...], 'kwargs': {...}}.
    """
    def _ctor(*args, **kwargs):
        obj = {"__type__": type_name}
        if args:
            obj["__args__"] = list(args)
        if kwargs:
            obj.update(kwargs)
        return obj
    return _ctor
class _AutoCtorEnv(dict):
    """
    Mapping for eval() that auto-creates dummy constructors
    for any unknown identifier (e.g., ChatCompletionMessage).
    """
    def __missing__(self, key: str):
        # Allow attribute-style names; treat everything as a constructor
        ctor = _make_obj(key)
        self[key] = ctor
        return ctor
def _clean_outer_quotes_and_escapes(s: str) -> str:
    """
    The provided payload is wrapped in single quotes and uses backslash-escaped
    quotes (e.g., \'). Normalize it so we can parse it.
    """
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] == "'":
        s = s[1:-1]
    # Unescape backslash-escaped quotes that appear in the payload
    # s = s.replace("\\'", "'").replace('\\"', '"')
    return s
def _fix_syntax_errors(s: str) -> str:
    """
    Fix common syntax errors in the string before parsing.
    """
    # Fix missing commas between object constructors
    # Pattern: ObjectConstructor(...) ObjectConstructor(...)
    s = re.sub(r'\)\s+([A-Z][a-zA-Z0-9_]*\()', r'), \1', s)
    
    # Fix missing commas between dictionary items
    # Pattern: 'key': value 'key': value
    s = re.sub(r"'\s+([^:]+):", r"', '\1:", s)
    
    # Fix unescaped quotes in content fields
    # This is a more complex fix - we need to be careful not to break valid JSON
    return s

def parse_payload_to_dict(text: str) -> Dict[str, Any]:
    """
    Convert the given string into a Python dict.
    Strategy:
      1) Normalize quotes/escapes.
      2) Try ast.literal_eval() for pure-literal inputs.
      3) If that fails (because of items like ChatCompletionMessage(...)),
         fall back to a sandboxed eval with dummy constructors.
    """
    cleaned = _clean_outer_quotes_and_escapes(text)
    # First attempt: safe literal parsing (fast path)
    try:
        value = ast.literal_eval(cleaned)
        if isinstance(value, dict):
            return value
        # If it's not a dict (e.g., list), wrap in a dict for consistency
        return {"value": value}
    except Exception:
        pass
    # Fallback: sandboxed eval with auto-constructors, no builtins
    env = _AutoCtorEnv()
    # Explicitly disallow builtins for safety
    safe_globals = {"__builtins__": {}}
    try:
        value = eval(cleaned.replace("\\'", "'").replace('\\"', '"'), safe_globals, env)  # noqa: S307 (intentional, sandboxed)
    except:
        value = eval(cleaned, safe_globals, env)

    return value


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Convert NexusBench results to specified format')
    parser.add_argument('parquets_dir', help='NexusBench parquets directory (e.g., NexusBench/parquets)')
    parser.add_argument('output_dir', help='Output directory path')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.parquets_dir):
        print(f"Error: Parquets directory '{args.parquets_dir}' does not exist.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert the results
    convert_nexusbench_results(args.parquets_dir, args.output_dir)


if __name__ == "__main__":
    main()
