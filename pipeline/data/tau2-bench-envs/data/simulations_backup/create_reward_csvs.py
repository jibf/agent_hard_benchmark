import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import numpy as np

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
        
        if task_id is not None:
            # Extract global reward
            global_reward = reward_info.get('reward', 0.0)
            
            # Extract fine-grained rewards
            db_check = reward_info.get('db_check')
            fine_grained = {
                'global_reward': global_reward,
                'db_reward': db_check.get('db_reward', 0.0) if db_check else 0.0,
                'env_assertion_reward': 0.0,
                'action_reward': 0.0,
                'nl_assertion_reward': 0.0,
                'communicate_reward': 0.0
            }
            
            # Calculate env assertion reward (average of all env assertions)
            env_assertions = reward_info.get('env_assertions', [])
            if env_assertions:
                env_rewards = [assertion.get('reward', 0.0) for assertion in env_assertions]
                fine_grained['env_assertion_reward'] = sum(env_rewards) / len(env_rewards)
            
            # Calculate action reward (average of all action checks)
            action_checks = reward_info.get('action_checks', [])
            if action_checks:
                action_rewards = [check.get('action_reward', 0.0) for check in action_checks]
                fine_grained['action_reward'] = sum(action_rewards) / len(action_rewards)
            
            # Calculate nl assertion reward (average of all nl assertions)
            nl_assertions = reward_info.get('nl_assertions', [])
            if nl_assertions:
                nl_rewards = [assertion.get('reward', 0.0) for assertion in nl_assertions]
                fine_grained['nl_assertion_reward'] = sum(nl_rewards) / len(nl_rewards)
            
            # Get communicate reward if available
            communicate_checks = reward_info.get('communicate_checks')
            if communicate_checks:
                if isinstance(communicate_checks, dict):
                    fine_grained['communicate_reward'] = communicate_checks.get('reward', 0.0)
                elif isinstance(communicate_checks, list) and communicate_checks:
                    # If it's a list, calculate average reward
                    comm_rewards = [check.get('reward', 0.0) for check in communicate_checks if isinstance(check, dict)]
                    fine_grained['communicate_reward'] = sum(comm_rewards) / len(comm_rewards) if comm_rewards else 0.0
            
            task_results[task_id] = fine_grained
    
    return task_results

def get_task_info(data, task_id):
    """Get task information for a specific task ID"""
    if not data or 'tasks' not in data:
        return None
    
    for task in data['tasks']:
        if task.get('task_id') == task_id:
            return task
    
    return None

def create_reward_csvs():
    """Create CSV files for each domain with task rewards"""
    
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
    
    # Process each domain
    for domain in ['telecom', 'airline', 'retail']:
        print(f"\n{'='*60}")
        print(f"PROCESSING {domain.upper()} DOMAIN")
        print(f"{'='*60}")
        
        if domain not in domain_files:
            print(f"No files found for {domain} domain")
            continue
        
        files = domain_files[domain]
        print(f"Found {len(files)} files for {domain} domain")
        
        # Collect all task results
        all_task_results = defaultdict(dict)
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
            for task_id, rewards in task_results.items():
                if task_id not in all_task_results:
                    all_task_results[task_id] = {}
                all_task_results[task_id][agent_model] = rewards
        
        # Create DataFrames for different reward types
        reward_types = ['global_reward', 'db_reward', 'env_assertion_reward', 'action_reward', 'nl_assertion_reward', 'communicate_reward']
        
        for reward_type in reward_types:
            # Create DataFrame
            df_data = []
            task_ids = sorted(all_task_results.keys())
            agent_models = sorted(set([model for task_data in all_task_results.values() for model in task_data.keys()]))
            
            for task_id in task_ids:
                row = {'task_id': task_id}
                
                # Add task description if available
                task_info = task_info_map.get(task_id)
                if task_info:
                    row['description'] = task_info.get('description', 'N/A')
                    row['task_type'] = task_info.get('task_type', 'N/A')
                    row['difficulty'] = task_info.get('difficulty', 'N/A')
                
                # Add rewards for each model
                for agent_model in agent_models:
                    if agent_model in all_task_results[task_id]:
                        row[agent_model] = all_task_results[task_id][agent_model].get(reward_type, 0.0)
                    else:
                        row[agent_model] = np.nan
                
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            
            # Save CSV
            csv_filename = f"{domain}_{reward_type}_rewards.csv"
            df.to_csv(csv_filename, index=False)
            print(f"Created {csv_filename} with {len(df)} tasks and {len(agent_models)} models")
        
        # Create visualization for fine-grained rewards
        create_fine_grained_visualization(domain, all_task_results, agent_models)

def create_fine_grained_visualization(domain, all_task_results, agent_models):
    """Create visualization for fine-grained rewards"""
    
    reward_types = ['global_reward', 'db_reward', 'env_assertion_reward', 'action_reward', 'nl_assertion_reward', 'communicate_reward']
    
    # Prepare data for visualization
    viz_data = []
    
    for agent_model in agent_models:
        for reward_type in reward_types:
            total_reward = 0.0
            count = 0
            
            for task_id, task_data in all_task_results.items():
                if agent_model in task_data:
                    reward = task_data[agent_model].get(reward_type, 0.0)
                    total_reward += reward
                    count += 1
            
            if count > 0:
                avg_reward = total_reward / count
                viz_data.append({
                    'agent_model': agent_model,
                    'reward_type': reward_type,
                    'avg_reward': avg_reward,
                    'total_reward': total_reward,
                    'task_count': count
                })
    
    df_viz = pd.DataFrame(viz_data)
    
    # Create heatmap
    plt.figure(figsize=(15, 8))
    
    # Pivot data for heatmap
    heatmap_data = df_viz.pivot(index='agent_model', columns='reward_type', values='avg_reward')
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=0.5, 
                fmt='.2f', cbar_kws={'label': 'Average Reward'})
    
    plt.title(f'{domain.upper()} Domain - Fine-grained Reward Analysis')
    plt.xlabel('Reward Type')
    plt.ylabel('Agent Model')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'{domain}_fine_grained_rewards_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Created {domain}_fine_grained_rewards_heatmap.png")
    
    # Create bar chart for global rewards
    plt.figure(figsize=(12, 6))
    
    global_rewards = df_viz[df_viz['reward_type'] == 'global_reward']
    global_rewards = global_rewards.sort_values('avg_reward', ascending=True)
    
    plt.barh(global_rewards['agent_model'], global_rewards['avg_reward'])
    plt.xlabel('Average Global Reward')
    plt.ylabel('Agent Model')
    plt.title(f'{domain.upper()} Domain - Global Reward Comparison')
    plt.xlim(0, 1)
    
    # Add value labels on bars
    for i, (_, row) in enumerate(global_rewards.iterrows()):
        plt.text(row['avg_reward'] + 0.01, i, f'{row["avg_reward"]:.3f}', 
                va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{domain}_global_rewards_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Created {domain}_global_rewards_comparison.png")
    
    plt.close('all')

if __name__ == "__main__":
    create_reward_csvs() 