import json
import os
import glob
import shutil
from collections import defaultdict
from datetime import datetime

def analyze_json_files():
    """Analyze JSON files to identify complete task results"""
    
    # Expected task counts for each domain
    expected_tasks = {
        'airline': 50,
        'retail': 114, 
        'telecom': 114
    }
    
    # Get all JSON files in the directory
    json_files = glob.glob("*.json")
    
    complete_files = []
    incomplete_files = []
    
    for file_path in json_files:
        if file_path == "convert.py" or file_path == "analyze_files.py":
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract domain and model information
            domain = data.get('info', {}).get('environment_info', {}).get('domain_name', 'unknown')
            agent_model = data.get('info', {}).get('agent_info', {}).get('llm', 'unknown')
            user_model = data.get('info', {}).get('user_info', {}).get('llm', 'unknown')
            
            # Extract timestamp
            timestamp_str = data.get('timestamp', '')
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')) if timestamp_str else datetime.min
            except:
                timestamp = datetime.min
            
            # Count tasks
            num_tasks = len(data.get('tasks', []))
            num_simulations = len(data.get('simulations', []))
            
            # Check if this is a complete file
            expected_count = expected_tasks.get(domain, 0)
            is_complete = num_tasks == expected_count
            
            file_info = {
                'filename': file_path,
                'domain': domain,
                'agent_model': agent_model,
                'user_model': user_model,
                'num_tasks': num_tasks,
                'num_simulations': num_simulations,
                'expected_tasks': expected_count,
                'is_complete': is_complete,
                'timestamp': timestamp,
                'data': data
            }
            
            if is_complete:
                complete_files.append(file_info)
            else:
                incomplete_files.append(file_info)
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Group complete files by agent_model/user_model pairs
    model_pairs = defaultdict(list)
    for file_info in complete_files:
        pair_key = f"{file_info['agent_model']} + {file_info['user_model']}"
        model_pairs[pair_key].append(file_info)
    
    return complete_files, incomplete_files, model_pairs

def calculate_pass_rate(data):
    """Calculate pass^1 rate from simulation data"""
    simulations = data.get('simulations', [])
    if not simulations:
        return 0.0
    
    passed_count = 0
    total_count = len(simulations)
    
    for simulation in simulations:
        reward_info = simulation.get('reward_info', {})
        reward = reward_info.get('reward', 0.0)
        if reward >= 1.0:  # Pass threshold
            passed_count += 1
    
    return passed_count / total_count if total_count > 0 else 0.0

def select_most_recent_per_domain(model_pairs):
    """Select the most recent file per domain for each model pair"""
    selected_files = {}
    
    for pair_key, files in model_pairs.items():
        # Group files by domain
        domain_files = defaultdict(list)
        for file_info in files:
            domain_files[file_info['domain']].append(file_info)
        
        # Select most recent file per domain
        for domain, domain_file_list in domain_files.items():
            most_recent = max(domain_file_list, key=lambda x: x['timestamp'])
            selected_files[f"{pair_key}_{domain}"] = most_recent
    
    return selected_files

def filter_and_move_files(selected_files):
    """Filter files for gpt-4o user model and non-zero pass rates, move others"""
    
    # Create target directory
    target_dir = "../simulations_trials"
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all JSON files in current directory
    all_json_files = glob.glob("*.json")
    
    # Files to keep (filtered)
    files_to_keep = []
    files_to_move = []
    
    # Check each selected file
    for key, file_info in selected_files.items():
        user_model = file_info['user_model']
        pass_rate = calculate_pass_rate(file_info['data'])
        
        # Check if user model contains gpt-4o and pass rate is not 0
        if 'gpt-4o' in user_model.lower() and pass_rate > 0:
            files_to_keep.append(file_info['filename'])
            print(f"KEEPING: {file_info['filename']}")
            print(f"  User Model: {user_model}")
            print(f"  Pass Rate: {pass_rate:.3f}")
        else:
            files_to_move.append(file_info['filename'])
            print(f"MOVING: {file_info['filename']}")
            print(f"  User Model: {user_model}")
            print(f"  Pass Rate: {pass_rate:.3f}")
    
    # Move all other JSON files (not in selected_files)
    for json_file in all_json_files:
        if json_file not in [f['filename'] for f in selected_files.values()]:
            files_to_move.append(json_file)
            print(f"MOVING (not selected): {json_file}")
    
    # Actually move the files
    print(f"\nMoving {len(files_to_move)} files to {target_dir}/")
    for file_to_move in files_to_move:
        try:
            shutil.move(file_to_move, os.path.join(target_dir, file_to_move))
            print(f"  Moved: {file_to_move}")
        except Exception as e:
            print(f"  Error moving {file_to_move}: {e}")
    
    print(f"\nKept {len(files_to_keep)} files in current directory")
    for file_to_keep in files_to_keep:
        print(f"  Kept: {file_to_keep}")
    
    return files_to_keep, files_to_move

def main():
    print("Analyzing JSON files for complete task results...")
    print("=" * 60)
    
    complete_files, incomplete_files, model_pairs = analyze_json_files()
    
    print(f"\nCOMPLETE FILES ({len(complete_files)}):")
    print("-" * 40)
    for file_info in complete_files:
        print(f"{file_info['filename']}")
        print(f"  Domain: {file_info['domain']}")
        print(f"  Agent: {file_info['agent_model']}")
        print(f"  User: {file_info['user_model']}")
        print(f"  Tasks: {file_info['num_tasks']}/{file_info['expected_tasks']}")
        print(f"  Timestamp: {file_info['timestamp']}")
        print()
    
    print(f"\nINCOMPLETE FILES ({len(incomplete_files)}):")
    print("-" * 40)
    for file_info in incomplete_files:
        print(f"{file_info['filename']}")
        print(f"  Domain: {file_info['domain']}")
        print(f"  Agent: {file_info['agent_model']}")
        print(f"  User: {file_info['user_model']}")
        print(f"  Tasks: {file_info['num_tasks']}/{file_info['expected_tasks']}")
        print()
    
    print(f"\nMODEL PAIRS WITH COMPLETE RESULTS:")
    print("-" * 40)
    for pair_key, files in model_pairs.items():
        print(f"\n{pair_key}")
        print(f"  Complete files: {len(files)}")
        domains = set(f['domain'] for f in files)
        print(f"  Domains: {', '.join(domains)}")
        for file_info in files:
            print(f"    - {file_info['filename']} ({file_info['domain']})")
    
    # Select most recent files per domain and calculate pass rates
    print(f"\n" + "=" * 60)
    print("MOST RECENT FILES PER DOMAIN WITH PASS^1 RATES:")
    print("=" * 60)
    
    selected_files = select_most_recent_per_domain(model_pairs)
    
    for key, file_info in selected_files.items():
        pass_rate = calculate_pass_rate(file_info['data'])
        print(f"\n{key}")
        print(f"  File: {file_info['filename']}")
        print(f"  Domain: {file_info['domain']}")
        print(f"  Agent: {file_info['agent_model']}")
        print(f"  User: {file_info['user_model']}")
        print(f"  Timestamp: {file_info['timestamp']}")
        print(f"  Pass^1 Rate: {pass_rate:.3f} ({pass_rate*100:.1f}%)")
    
    # Filter and move files
    print(f"\n" + "=" * 60)
    print("FILTERING AND MOVING FILES:")
    print("=" * 60)
    print("Keeping only files with user_model=gpt-4o and pass_rate > 0")
    
    files_to_keep, files_to_move = filter_and_move_files(selected_files)

if __name__ == "__main__":
    main() 