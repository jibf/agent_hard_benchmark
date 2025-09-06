import json
import glob
import os

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

def convert_airline_retail_to_sglang_jsonl(input_pattern: str, output_jsonl_path: str):
    """
    Converts airline and retail tau2-bench result JSON files to a single sglang eval JSONL file.
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
    
    total_simulations = 0
    
    with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
        for task_name, file_path in task_files.items():
            print(f"Processing {task_name} file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                tau2_data = json.load(f)

            tasks_map = {task['id']: task for task in tau2_data['tasks']}

            global_info = tau2_data.get('info', {})
            agent_info = global_info.get('agent_info', {})
            user_info = global_info.get('user_info', {})
            env_info = global_info.get('environment_info', {})
            agent_llm_args = agent_info.get('llm_args', {})
            user_llm_args = user_info.get('llm_args', {})

            for simulation in tau2_data['simulations']:
                task_id = simulation['task_id']
                task = tasks_map.get(task_id)
                if not task:
                    print(f"Warning: No task found for simulation with task_id: {task_id}")
                    continue

                # Create a richer message list with system prompt
                messages = []
                
                # Add system message with agent policy
                if env_info.get('policy'):
                    messages.append({
                        "role": "system",
                        "content": env_info['policy'],
                        "turn_idx": -1  # System message gets turn_idx -1
                    })
                
                # Add existing conversation messages with adjusted turn_idx
                for msg in simulation.get('messages', []):
                    new_msg = {
                        "role": msg.get("role"),
                        "content": msg.get("content"),
                        "turn_idx": msg.get("turn_idx") + 1 if msg.get("turn_idx") is not None else len(messages)
                    }
                    # Only include tool_calls if it's not null/empty
                    if msg.get("tool_calls"):
                        new_msg["tool_calls"] = msg["tool_calls"]
                    messages.append(new_msg)

                reward_info = simulation.get('reward_info', {})
                db_check = reward_info.get('db_check', {}) or {}

                sglang_obj = {
                    "model_path": reformat_model_name(agent_info.get('llm', 'unknown_model')),
                    "user_model_path": reformat_model_name(user_info.get('llm', 'unknown_model')),
                    "benchmark_name": "tau2-bench",
                    "task_name": task_name,  # Use the extracted task name from filename
                    "sampling_params": {
                        "max_tokens": 16384,
                        "temperature": agent_llm_args.get('temperature', 0.0)
                    },
                    "user_sampling_params": {
                        "temperature": user_llm_args.get('temperature', 0.0)
                    },
                    "messages": messages,
                    "eval_result": {
                        "score": reward_info.get('reward', 0.0),
                        "db_match": db_check.get('db_match', False)
                    },
                    "meta": {
                        "id": task_id,
                        "is_correct": db_check.get('db_match', False),
                        "finish_reason": simulation.get('termination_reason'),
                        "run_timestamp": tau2_data.get('timestamp'),
                        "simulation_start_time": simulation.get('start_time'),
                        "simulation_end_time": simulation.get('end_time'),
                        "duration_seconds": simulation.get('duration'),
                        "agent_cost": simulation.get('agent_cost'),
                        "user_cost": simulation.get('user_cost'),
                        "reward_details": {
                            "db_reward": reward_info.get('db_reward'),
                            "reward_breakdown": reward_info.get('reward_breakdown'),
                            "nl_assertions": reward_info.get('nl_assertions')
                        },
                        "task_description": task.get('description'),
                        "task_instructions": task.get('user_scenario', {}).get('instructions') if task else None,
                        "source_file": os.path.basename(file_path)  # Add source file info
                    }
                }

                f_out.write(json.dumps(sglang_obj) + '\n')
                total_simulations += 1

    print(f"Successfully converted {total_simulations} simulations from {len(task_files)} files.")
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

def convert_multiple_tau2_to_sglang_jsonl(input_pattern: str, output_jsonl_path: str):
    """
    Converts multiple tau2-bench result JSON files to a single sglang eval JSONL file.
    Finds all JSON files matching the pattern and combines them.
    """
    # Find all JSON files matching the pattern
    json_files = glob.glob(input_pattern)
    
    if not json_files:
        print(f"No JSON files found matching pattern: {input_pattern}")
        return
    
    # Check if we have exactly 3 files with the expected task names
    task_files = {}
    for file_path in json_files:
        task_name = extract_task_name_from_filename(os.path.basename(file_path))
        if task_name in ['airline', 'retail', 'telecom']:
            task_files[task_name] = file_path
    
    if len(task_files) != 3:
        print(f"Warning: Expected 3 files (airline, retail, telecom), found {len(task_files)}")
        print(f"Found task files: {list(task_files.keys())}")
    
    total_simulations = 0
    
    with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
        for task_name, file_path in task_files.items():
            print(f"Processing {task_name} file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                tau2_data = json.load(f)

            tasks_map = {task['id']: task for task in tau2_data['tasks']}

            global_info = tau2_data.get('info', {})
            agent_info = global_info.get('agent_info', {})
            user_info = global_info.get('user_info', {})
            env_info = global_info.get('environment_info', {})
            agent_llm_args = agent_info.get('llm_args', {})
            user_llm_args = user_info.get('llm_args', {})

            for simulation in tau2_data['simulations']:
                task_id = simulation['task_id']
                task = tasks_map.get(task_id)
                if not task:
                    print(f"Warning: No task found for simulation with task_id: {task_id}")
                    continue

                # Create a richer message list with system prompt
                messages = []
                
                # Add system message with agent policy
                if env_info.get('policy'):
                    messages.append({
                        "role": "system",
                        "content": env_info['policy'],
                        "turn_idx": -1  # System message gets turn_idx -1
                    })
                
                # Add existing conversation messages with adjusted turn_idx
                for msg in simulation.get('messages', []):
                    new_msg = {
                        "role": msg.get("role"),
                        "content": msg.get("content"),
                        "turn_idx": msg.get("turn_idx") + 1 if msg.get("turn_idx") is not None else len(messages)
                    }
                    # Only include tool_calls if it's not null/empty
                    if msg.get("tool_calls"):
                        new_msg["tool_calls"] = msg["tool_calls"]
                    messages.append(new_msg)

                reward_info = simulation.get('reward_info', {})
                db_check = reward_info.get('db_check', {}) or {}

                sglang_obj = {
                    "model_path": reformat_model_name(agent_info.get('llm', 'unknown_model')),
                    "user_model_path": reformat_model_name(user_info.get('llm', 'unknown_model')),
                    # "model_path": agent_info.get('llm', 'unknown_model'),
                    # "user_model_path": user_info.get('llm', 'unknown_model'),
                    "benchmark_name": "tau2-bench",
                    "task_name": task_name,  # Use the extracted task name from filename
                    "sampling_params": {
                        "max_tokens": 16384,
                        "temperature": agent_llm_args.get('temperature', 0.0)
                    },
                    "user_sampling_params": {
                        "temperature": user_llm_args.get('temperature', 0.0)
                    },
                    "messages": messages,
                    "eval_result": {
                        "score": reward_info.get('reward', 0.0),
                        "db_match": db_check.get('db_match', False)
                    },
                    "meta": {
                        "id": task_id,
                        "is_correct": db_check.get('db_match', False),
                        "finish_reason": simulation.get('termination_reason'),
                        "run_timestamp": tau2_data.get('timestamp'),
                        "simulation_start_time": simulation.get('start_time'),
                        "simulation_end_time": simulation.get('end_time'),
                        "duration_seconds": simulation.get('duration'),
                        "agent_cost": simulation.get('agent_cost'),
                        "user_cost": simulation.get('user_cost'),
                        "reward_details": {
                            "db_reward": reward_info.get('db_reward'),
                            "reward_breakdown": reward_info.get('reward_breakdown'),
                            "nl_assertions": reward_info.get('nl_assertions')
                        },
                        "task_description": task.get('description'),
                        "task_instructions": task.get('user_scenario', {}).get('instructions') if task else None,
                        "source_file": os.path.basename(file_path)  # Add source file info
                    }
                }

                f_out.write(json.dumps(sglang_obj) + '\n')
                total_simulations += 1

    print(f"Successfully converted {total_simulations} simulations from {len(task_files)} files.")
    print(f"Combined output saved to: {output_jsonl_path}")

def convert_tau2_to_sglang_jsonl(input_json_path: str, output_jsonl_path: str):
    """
    Converts a tau2-bench result JSON file to a richer sglang eval JSONL file,
    preserving full message history, timestamps, and task descriptions.
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        tau2_data = json.load(f)

    tasks_map = {task['id']: task for task in tau2_data['tasks']}

    global_info = tau2_data.get('info', {})
    agent_info = global_info.get('agent_info', {})
    user_info = global_info.get('user_info', {})
    env_info = global_info.get('environment_info', {})
    agent_llm_args = agent_info.get('llm_args', {})
    user_llm_args = user_info.get('llm_args', {}) # NEW: Get user model args

    with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
        for simulation in tau2_data['simulations']:
            task_id = simulation['task_id']
            task = tasks_map.get(task_id)
            if not task:
                print(f"Warning: No task found for simulation with task_id: {task_id}")
                continue

            # MODIFIED: Create a richer message list with system prompt
            messages = []
            
            # Add system message with agent policy
            if env_info.get('policy'):
                messages.append({
                    "role": "system",
                    "content": env_info['policy'],
                    "turn_idx": -1  # System message gets turn_idx -1
                })
            
            # Add existing conversation messages with adjusted turn_idx
            for msg in simulation.get('messages', []):
                new_msg = {
                    "role": msg.get("role"),
                    "content": msg.get("content"),
                    "turn_idx": msg.get("turn_idx") + 1 if msg.get("turn_idx") is not None else len(messages)
                }
                # Only include tool_calls if it's not null/empty
                if msg.get("tool_calls"):
                    new_msg["tool_calls"] = msg["tool_calls"]
                messages.append(new_msg)

            reward_info = simulation.get('reward_info', {})
            # db_check = reward_info.get('db_check', {})
            db_check = reward_info.get('db_check', {}) or {}

            sglang_obj = {
                "model_path": reformat_model_name(agent_info.get('llm', 'unknown_model')),
                "user_model_path": reformat_model_name(user_info.get('llm', 'unknown_model')),
                "benchmark_name": "tau2-bench",
                "task_name": env_info.get('domain_name', 'unknown_task'),
                "sampling_params": {
                    "max_tokens": 16384,
                    "temperature": agent_llm_args.get('temperature', 0.0)
                },
                # NEW: Add separate sampling params for the user model
                "user_sampling_params": {
                    "temperature": user_llm_args.get('temperature', 0.0)
                },
                "messages": messages, # Use the new richer messages list
                "eval_result": {
                    "score": reward_info.get('reward', 0.0),
                    "db_match": db_check.get('db_match', False)
                },
                # MODIFIED: Expanded meta object
                "meta": {
                    "id": task_id,
                    "is_correct": db_check.get('db_match', False),
                    "finish_reason": simulation.get('termination_reason'),
                    "run_timestamp": tau2_data.get('timestamp'), # NEW: Global run timestamp
                    "simulation_start_time": simulation.get('start_time'), # NEW: Task start time
                    "simulation_end_time": simulation.get('end_time'), # NEW: Task end time
                    "duration_seconds": simulation.get('duration'),
                    "agent_cost": simulation.get('agent_cost'),
                    "user_cost": simulation.get('user_cost'),
                    "reward_details": {
                        "db_reward": reward_info.get('db_reward'),
                        "reward_breakdown": reward_info.get('reward_breakdown'),
                        "nl_assertions": reward_info.get('nl_assertions')
                    },
                    "task_description": task.get('description'), # NEW: Full original task info
                    "task_instructions": task.get('user_scenario', {}).get('instructions') if task else None # NEW: Add task instructions to meta
                }
            }

            f_out.write(json.dumps(sglang_obj) + '\n')

    print(f"Successfully converted {len(tau2_data['simulations'])} simulations.")
    print(f"Richer output saved to: {output_jsonl_path}")

# --- Example Usage ---
# convert_tau2_to_sglang_jsonl_v2("my_tau2_results.json", "sglang_results_rich.jsonl")

# --- Example Usage ---
# Create a dummy input file for demonstration if you don't have one.
# In a real scenario, you would use your actual result file.
# create_dummy_tau2_file("tau2_results.json")

# Run the conversion
# Make sure you have a 'tau2_results.json' file in the same directory
# or provide the correct path.
# convert_tau2_to_sglang_jsonl("tau2_results.json", "sglang_results.jsonl")

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
        print("  Single file: python convert.py <input_json_path> <output_jsonl_path>")
        print("  Multiple files: python convert.py --pattern <input_pattern> <output_jsonl_path>")
        print("  Airline+Retail: python convert.py --airline-retail <input_pattern> <output_jsonl_path>")
        print("  Auto convert: python convert.py --auto [input_directory]")
        print("  Example pattern: '*claude-4-sonnet-thinking-on-10k*.json'")
    elif len(sys.argv) == 3 and sys.argv[1] != '--pattern' and sys.argv[1] != '--airline-retail' and sys.argv[1] != '--auto':
        # Single file conversion
        convert_tau2_to_sglang_jsonl(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == '--pattern':
        # Multiple file conversion
        convert_multiple_tau2_to_sglang_jsonl(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == '--airline-retail':
        # Airline + Retail conversion
        convert_airline_retail_to_sglang_jsonl(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 2 and sys.argv[1] == '--auto':
        # Auto convert all models
        input_directory = sys.argv[2] if len(sys.argv) > 2 else "."
        auto_convert_models(model_names, input_directory)
    else:
        print("Invalid arguments. Use --help for usage information.")