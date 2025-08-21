import json

def reformat_model_name(model_name: str) -> str:
    """
    A helper function to reformat model names to a path-like structure.
    This is an example and may need to be adjusted for your specific model names.
    """
    if "gpt-4o" in model_name.lower():
        return "openai/gpt-4o-20240806"
    return model_name

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

            # MODIFIED: Create a richer message list
            messages = []
            for msg in simulation.get('messages', []):
                new_msg = {
                    "role": msg.get("role"),
                    "content": msg.get("content"),
                    "turn_idx": msg.get("turn_idx")
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
                    "db_match": db_check.get('db_match', False),
                    "overall_reward": reward_info.get('reward', 0.0)
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
                    "task_description": task.get('description') # NEW: Full original task info
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
    if len(sys.argv) != 3:
        print("Usage: python convert.py <input_json_path> <output_jsonl_path>")
    else:
        convert_tau2_to_sglang_jsonl(sys.argv[1], sys.argv[2])