import json
import glob
import os
from collections import defaultdict, Counter

def load_json_file(filepath):
    """Load and parse a JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_task_results(data):
    """Extract task results from simulation data"""
    if not data or 'simulations' not in data:
        return {}
    
    task_results = {}
    
    for simulation in data['simulations']:
        task_id = simulation.get('task_id')
        reward_info = simulation.get('reward_info', {})
        reward = reward_info.get('reward', 0.0)
        
        if task_id is not None:
            task_results[task_id] = reward
    
    return task_results

def get_task_info(data, task_id):
    """Get task information for a specific task ID"""
    if not data or 'tasks' not in data:
        return None
    
    for task in data['tasks']:
        if task.get('task_id') == task_id:
            return task
    
    return None

def analyze_failing_tasks():
    """Analyze all JSON files to find tasks that all models get wrong"""
    
    # Get all JSON files
    json_files = glob.glob("*.json")
    
    # Group files by domain
    domain_files = defaultdict(list)
    
    for filepath in json_files:
        if filepath.endswith('.json') and not filepath.endswith('convert.py') and not filepath.endswith('analyze_files.py'):
            data = load_json_file(filepath)
            if data:
                domain = data.get('info', {}).get('environment_info', {}).get('domain_name', 'unknown')
                agent_model = data.get('info', {}).get('agent_info', {}).get('llm', 'unknown')
                domain_files[domain].append({
                    'filepath': filepath,
                    'data': data,
                    'agent_model': agent_model
                })
    
    # Analyze each domain
    for domain in ['telecom', 'airline', 'retail']:
        print(f"\n{'='*60}")
        print(f"ANALYZING {domain.upper()} DOMAIN")
        print(f"{'='*60}")
        
        if domain not in domain_files:
            print(f"No files found for {domain} domain")
            continue
        
        files = domain_files[domain]
        print(f"Found {len(files)} files for {domain} domain")
        
        # Collect all task results
        all_task_results = defaultdict(list)
        task_info_map = {}
        
        for file_info in files:
            agent_model = file_info['agent_model']
            data = file_info['data']
            
            # Get task results for this model
            task_results = extract_task_results(data)
            
            # Store task info from first file (they should be the same)
            if not task_info_map:
                for task in data.get('tasks', []):
                    task_id = task.get('task_id')
                    if task_id:
                        task_info_map[task_id] = task
            
            # Add results to collection
            for task_id, reward in task_results.items():
                all_task_results[task_id].append((agent_model, reward))
        
        # Find tasks where all models got reward=0
        failing_tasks = []
        
        for task_id, results in all_task_results.items():
            all_zero = all(reward == 0.0 for _, reward in results)
            if all_zero:
                failing_tasks.append((task_id, results))
        
        print(f"\nTasks where ALL models got reward=0: {len(failing_tasks)}")
        
        if failing_tasks:
            print(f"\nFAILING TASKS FOR {domain.upper()}:")
            print("-" * 50)
            
            for task_id, results in failing_tasks:
                print(f"\nTask ID: {task_id}")
                
                # Get task information
                task_info = task_info_map.get(task_id)
                if task_info:
                    print(f"Task Description: {task_info.get('description', 'N/A')}")
                    print(f"Task Type: {task_info.get('task_type', 'N/A')}")
                    print(f"Difficulty: {task_info.get('difficulty', 'N/A')}")
                    
                    # Print initial state if available
                    initial_state = task_info.get('initial_state', {})
                    if initial_state:
                        print(f"Initial State: {initial_state}")
                
                print(f"Models that failed (reward=0):")
                for agent_model, reward in results:
                    print(f"  - {agent_model}: {reward}")
                
                print("-" * 30)
        else:
            print(f"No tasks found where all models failed for {domain} domain")

if __name__ == "__main__":
    analyze_failing_tasks()
