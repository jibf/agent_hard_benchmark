import json
import glob
import os
import re

def reformat_model_name(model_name: str) -> str:
    """
    A helper function to reformat model names to a path-like structure.
    """
    if "gpt-4o" in model_name.lower():
        if "20240806" in model_name:
            return "openai/gpt-4o-20240806"
        elif "mini" in model_name.lower():
            return "openai/gpt-4o-mini"
    elif "gpt-4.1-mini" in model_name.lower():
        return "openai/gpt-4.1-mini"
    elif "gpt-4.1-nano" in model_name.lower():
        return "openai/gpt-4.1-nano"
    elif "gpt-5" in model_name.lower():
        return "openai/gpt-5"
    elif "gpt-4.1" in model_name.lower():
        return "openai/gpt-4.1"
    elif "claude-4-sonnet" in model_name.lower():
        if "thinking-on-10k" in model_name.lower():
            return "anthropic/claude-4-sonnet-thinking-on-10k"
        elif "thinking-off" in model_name.lower():
            return "anthropic/claude-4-sonnet-thinking-off"
        else:
            return "anthropic/claude-4-sonnet"
    elif "claude-4-opus" in model_name.lower():
        if "thinking-on-10k" in model_name.lower():
            return "anthropic/claude-4-opus-thinking-on-10k"
        elif "thinking-off" in model_name.lower():
            return "anthropic/claude-4-opus-thinking-off"
        else:
            return "anthropic/claude-4-sonnet"
    elif "qwen3" in model_name.lower():
        if "8b" in model_name.lower():
            return "togetherai/Qwen/Qwen3-8B"
        elif "32b" in model_name.lower():
            return "togetherai/Qwen/Qwen3-32B"
        elif "235b-a22b-thinking" in model_name.lower():
            return "togetherai/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8"
        elif "235b-a22b-instruct" in model_name.lower():
            return "togetherai/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
        elif "235b-a22b-fp8" in model_name.lower():
            return "togetherai/Qwen/Qwen3-235B-A22B-FP8"
        elif "coder" in model_name.lower():
            return "togetherai/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
    elif "deepseek" in model_name.lower():
        if "v3.1-thinking-off" in model_name.lower():
            return "deepseek-ai/DeepSeek-V3.1-thinking-off"
        elif "v3.1-thinking-on" in model_name.lower():
            return "deepseek-ai/DeepSeek-V3.1-thinking-on"
        elif "v3" in model_name.lower():
            return "deepseek-ai/DeepSeek-V3-0324"
        elif "r1" in model_name.lower():
            return "deepseek-ai/DeepSeek-R1-0528"
        
    elif "kimi" in model_name.lower():
        return "togetherai/moonshot/Kimi-K2-Instruct"
    elif "grok" in model_name.lower():
        return "xai/grok-4"
    elif "o3" in model_name.lower():
        return "openai/o3-high"
    elif "o4" in model_name.lower():
        return "openai/o4-mini-high"
    return model_name

def extract_task_name_from_filename(filename: str) -> str:
    """
    Extract task name from filename based on the pattern.
    Returns 'airline', 'retail', or 'telecom' based on what's in the filename.
    """
    filename_lower = filename.lower()
    if 'airline' in filename_lower:
        return 'airline'
    elif 'retail' in filename_lower:
        return 'retail'
    elif 'telecom' in filename_lower:
        return 'telecom'
    else:
        return 'unknown_task'

def extract_model_name_from_filename(filename: str) -> str:
    """
    Extract model name from taubench filename.
    Pattern: tool-calling-{task}-{model_name}-0.0_range_0--1_user-{user_model}-llm_{timestamp}.json
    """
    # Pattern to match model name after 'tool-calling-{task}-' and before '-0.0_range'
    pattern = r'tool-calling-(?:airline|retail|telecom)-([^-]+(?:-[^-]+)*?)-0\.0_range'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return "unknown"

def extract_user_model_from_filename(filename: str) -> str:
    """
    Extract user model from taubench filename.
    Pattern: tool-calling-{task}-{model_name}-0.0_range_0--1_user-{user_model}-llm_{timestamp}.json
    """
    # Pattern to match user model after 'user-' and before '-llm'
    pattern = r'user-([^-]+(?:-[^-]+)*?)-llm'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return "unknown"

def convert_taubench_to_sglang_jsonl(input_json_path: str, output_jsonl_path: str):
    """
    Converts a taubench result JSON file to sglang eval JSONL format.
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        taubench_data = json.load(f)

    if not isinstance(taubench_data, list):
        print(f"Error: Expected list of tasks, got {type(taubench_data)}")
        return

    # Extract model names from filename
    filename = os.path.basename(input_json_path)
    agent_model = extract_model_name_from_filename(filename)
    user_model = extract_user_model_from_filename(filename)
    task_name = extract_task_name_from_filename(filename)

    total_tasks = 0
    
    with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
        for task in taubench_data:
            if not isinstance(task, dict):
                continue
                
            task_id = task.get('task_id')
            reward = task.get('reward', 0.0)
            info = task.get('info', {})
            traj = task.get('traj', [])
            
            if task_id is None:
                continue

            # Extract task details
            task_info = info.get('task', {})
            instruction = task_info.get('instruction', '')
            actions = task_info.get('actions', [])
            
            # Extract reward info
            reward_info = info.get('reward_info', {}) or {}
            user_cost = info.get('user_cost', 0.0)

            # Convert trajectory to messages format
            messages = []
            for i, msg in enumerate(traj):
                if isinstance(msg, dict):
                    new_msg = {
                        "role": msg.get("role", "unknown"),
                        "content": msg.get("content", ""),
                        "turn_idx": i
                    }
                    # Include tool_calls if present
                    if msg.get("tool_calls"):
                        new_msg["tool_calls"] = msg["tool_calls"]
                    if msg.get("function_call"):
                        new_msg["function_call"] = msg["function_call"]
                    if msg.get("tool_call_id"):
                        new_msg["tool_call_id"] = msg["tool_call_id"]
                    if msg.get("name"):
                        new_msg["name"] = msg["name"]
                    messages.append(new_msg)

            # Create sglang object
            sglang_obj = {
                "model_path": reformat_model_name(agent_model),
                "user_model_path": reformat_model_name(user_model),
                "benchmark_name": "taubench",
                "task_name": task_name,
                "sampling_params": {
                    "max_tokens": 16384,
                    "temperature": 0.0
                },
                "user_sampling_params": {
                    "temperature": 0.0
                },
                "messages": messages,
                "eval_result": {
                    "score": reward,
                    "db_match": reward_info.get('info', {}).get('r_actions', 0.0) == 1.0
                },
                "meta": {
                    "id": str(task_id),
                    "is_correct": reward_info.get('info', {}).get('r_actions', 0.0) == 1.0,
                    "finish_reason": "user_stop",  # Default for taubench
                    "run_timestamp": None,  # Not available in taubench
                    "simulation_start_time": None,  # Not available in taubench
                    "simulation_end_time": None,  # Not available in taubench
                    "duration_seconds": None,  # Not available in taubench
                    "agent_cost": 0.0,  # Not available in taubench
                    "user_cost": user_cost,
                    "reward_details": {
                        "db_reward": reward_info.get('info', {}).get('r_actions'),
                        "reward_breakdown": None,
                        "nl_assertions": None
                    },
                    "task_description": {
                        "instruction": instruction,
                        "actions": actions
                    },
                    "source_file": filename
                }
            }

            f_out.write(json.dumps(sglang_obj) + '\n')
            total_tasks += 1

    print(f"Successfully converted {total_tasks} tasks from {filename}")
    print(f"Output saved to: {output_jsonl_path}")

def convert_airline_retail_to_sglang_jsonl(input_pattern: str, output_jsonl_path: str):
    """
    Converts airline and retail taubench result JSON files to a single sglang eval JSONL file.
    Finds all JSON files matching the pattern and combines airline and retail results.
    """
    # Find all JSON files matching the pattern
    json_files = glob.glob(input_pattern)
    
    if not json_files:
        print(f"No JSON files found matching pattern: {input_pattern}")
        return
    
    # Check if we have airline and retail files
    task_files = {}
    for file_path in json_files:
        task_name = extract_task_name_from_filename(os.path.basename(file_path))
        if task_name in ['airline', 'retail']:
            task_files[task_name] = file_path
    
    if len(task_files) != 2:
        print(f"Warning: Expected 2 files (airline, retail), found {len(task_files)}")
        print(f"Found task files: {list(task_files.keys())}")
    
    total_tasks = 0
    
    with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
        for task_name, file_path in task_files.items():
            print(f"Processing {task_name} file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                taubench_data = json.load(f)

            if not isinstance(taubench_data, list):
                print(f"Error: Expected list of tasks in {file_path}")
                continue

            # Extract model names from filename
            filename = os.path.basename(file_path)
            agent_model = extract_model_name_from_filename(filename)
            user_model = extract_user_model_from_filename(filename)

            for task in taubench_data:
                if not isinstance(task, dict):
                    continue
                    
                task_id = task.get('task_id')
                reward = task.get('reward', 0.0)
                info = task.get('info', {})
                traj = task.get('traj', [])
                
                if task_id is None:
                    continue

                # Extract task details
                task_info = info.get('task', {})
                instruction = task_info.get('instruction', '')
                actions = task_info.get('actions', [])
                
                # Extract reward info
                reward_info = info.get('reward_info', {}) or {}
                user_cost = info.get('user_cost', 0.0)

                # Convert trajectory to messages format
                messages = []
                for i, msg in enumerate(traj):
                    if isinstance(msg, dict):
                        new_msg = {
                            "role": msg.get("role", "unknown"),
                            "content": msg.get("content", ""),
                            "turn_idx": i
                        }
                        # Include tool_calls if present
                        if msg.get("tool_calls"):
                            new_msg["tool_calls"] = msg["tool_calls"]
                        if msg.get("function_call"):
                            new_msg["function_call"] = msg["function_call"]
                        if msg.get("tool_call_id"):
                            new_msg["tool_call_id"] = msg["tool_call_id"]
                        if msg.get("name"):
                            new_msg["name"] = msg["name"]
                        messages.append(new_msg)

                # Create sglang object
                sglang_obj = {
                    "model_path": reformat_model_name(agent_model),
                    "user_model_path": reformat_model_name(user_model),
                    "benchmark_name": "taubench",
                    "task_name": task_name,
                    "sampling_params": {
                        "max_tokens": 16384,
                        "temperature": 0.0
                    },
                    "user_sampling_params": {
                        "temperature": 0.0
                    },
                    "messages": messages,
                    "eval_result": {
                        "score": reward,
                        "db_match": reward_info.get('info', {}).get('r_actions', 0.0) == 1.0
                    },
                    "meta": {
                        "id": str(task_id),
                        "is_correct": reward_info.get('info', {}).get('r_actions', 0.0) == 1.0,
                        "finish_reason": "user_stop",  # Default for taubench
                        "run_timestamp": None,  # Not available in taubench
                        "simulation_start_time": None,  # Not available in taubench
                        "simulation_end_time": None,  # Not available in taubench
                        "duration_seconds": None,  # Not available in taubench
                        "agent_cost": 0.0,  # Not available in taubench
                        "user_cost": user_cost,
                        "reward_details": {
                            "db_reward": reward_info.get('info', {}).get('r_actions'),
                            "reward_breakdown": None,
                            "nl_assertions": None
                        },
                        "task_description": {
                            "instruction": instruction,
                            "actions": actions
                        },
                        "source_file": filename
                    }
                }

                f_out.write(json.dumps(sglang_obj) + '\n')
                total_tasks += 1

    print(f"Successfully converted {total_tasks} tasks from {len(task_files)} files.")
    print(f"Combined output saved to: {output_jsonl_path}")

def auto_convert_models(model_names: list, input_directory: str = "."):
    """
    Automatically convert airline and retail results for multiple models.
    Looks for files matching pattern: *{model_name}*.json in the input directory.
    """
    for model_name in model_names:
        print(f"\nProcessing model: {model_name}")
        
        # Create pattern to find airline and retail files for this model
        pattern = os.path.join(input_directory, f"*{model_name}*.json")
        
        # Check if files exist
        json_files = glob.glob(pattern)
        if not json_files:
            print(f"No files found for model {model_name} with pattern: {pattern}")
            continue
        
        # Filter for airline and retail files
        task_files = {}
        for file_path in json_files:
            task_name = extract_task_name_from_filename(os.path.basename(file_path))
            if task_name in ['airline', 'retail']:
                task_files[task_name] = file_path
        
        if len(task_files) < 2:
            print(f"Warning: Expected 2 files (airline, retail) for {model_name}, found {len(task_files)}")
            print(f"Found task files: {list(task_files.keys())}")
            if len(task_files) == 0:
                continue
        
        # Create output filename
        output_filename = f"{model_name}.jsonl"
        
        # Convert the files
        try:
            convert_airline_retail_to_sglang_jsonl(pattern, output_filename)
            print(f"Successfully created: {output_filename}")
        except Exception as e:
            print(f"Error processing {model_name}: {e}")

if __name__ == "__main__":
    import sys
    
    # Define the model names list
    model_names = [
        "grok-4",
        "Kimi-K2-Instruct", 
        "Qwen3-8B",
        "Qwen3-32B",
        "Qwen3-235B-A22B-Thinking-2507-FP8",
        "Qwen3-235B-A22B-FP8",
        "Qwen3-235B-A22B-Instruct-2507-FP8",
        "o4-mini-high",
        "o3-high",
        "gpt-4o-20240806",
        "gpt-4o-mini",
        "gpt-4.1",
        "DeepSeek-V3-0324",
        "DeepSeek-R1-0528",
        "claude-4-sonnet-thinking-on-10k",
        "claude-4-sonnet-thinking-off",
        "claude-4-opus-thinking-on-10k",
        "claude-4-opus-thinking-off",
        "Qwen3-Coder-480B-A35B-Instruct-FP8",
        "DeepSeek-V3.1-thinking-off",
        "DeepSeek-V3.1-thinking-on",
        "gpt-4.1-mini",
        "gpt-5",
        "gpt-4.1-nano",
    ]
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file: python convert_taubench.py <input_json_path> <output_jsonl_path>")
        print("  Multiple files: python convert_taubench.py --pattern <input_pattern> <output_jsonl_path>")
        print("  Airline+Retail: python convert_taubench.py --airline-retail <input_pattern> <output_jsonl_path>")
        print("  Auto convert: python convert_taubench.py --auto [input_directory]")
        print("  Example pattern: '*claude-4-sonnet-thinking-on-10k*.json'")
    elif len(sys.argv) == 3 and sys.argv[1] != '--pattern' and sys.argv[1] != '--airline-retail' and sys.argv[1] != '--auto':
        # Single file conversion
        convert_taubench_to_sglang_jsonl(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == '--pattern':
        # Multiple file conversion
        convert_airline_retail_to_sglang_jsonl(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == '--airline-retail':
        # Airline + Retail conversion
        convert_airline_retail_to_sglang_jsonl(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 2 and sys.argv[1] == '--auto':
        # Auto convert all models
        input_directory = sys.argv[2] if len(sys.argv) > 2 else "."
        auto_convert_models(model_names, input_directory)
    else:
        print("Invalid arguments. Use --help for usage information.") 