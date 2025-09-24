#!/usr/bin/env python3

import json
import sys
import os
import uuid
sys.path.append('/nethome/hsuh45/agenthard/tau2-bench/src')

from tau2.evaluator.evaluator import evaluate_simulation, EvaluationType
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.registry import registry
from tau2.utils.utils import get_now

def analyze_empty_trajectory_rewards():
    """
    Analyze which tasks in the telecom domain give reward=1.0 for agents that do nothing.
    Uses the proper evaluation method based on reward_basis.
    """
    # Load telecom tasks (full set)
    with open('/nethome/hsuh45/agenthard/tau2-bench/data/tau2/domains/telecom/tasks.json', 'r') as f:
        tasks_data = json.load(f)
    
    # Convert to Task objects
    tasks = [Task.model_validate(task_data) for task_data in tasks_data]
    
    results = []
    
    print("Analyzing telecom tasks with empty trajectory...")
    print("=" * 80)
    print(f"Total tasks loaded: {len(tasks)}")
    
    for task in tasks:
        # Create empty simulation (agent does nothing) with all required fields
        empty_simulation = SimulationRun(
            id=str(uuid.uuid4()),
            task_id=task.id,
            start_time=get_now(),
            end_time=get_now(),
            duration=0.0,
            termination_reason=TerminationReason.AGENT_STOP,
            messages=[]
        )
        
        # Determine the correct evaluation type based on reward_basis
        if not task.evaluation_criteria or not task.evaluation_criteria.reward_basis:
            evaluation_type = EvaluationType.ENV  # Default fallback
        else:
            reward_basis = set(task.evaluation_criteria.reward_basis)
            if len(reward_basis) == 1:
                if 'ENV_ASSERTION' in reward_basis or 'DB' in reward_basis:
                    evaluation_type = EvaluationType.ENV
                elif 'ACTION' in reward_basis:
                    evaluation_type = EvaluationType.ACTION
                elif 'COMMUNICATE' in reward_basis:
                    evaluation_type = EvaluationType.COMMUNICATE
                elif 'NL_ASSERTION' in reward_basis:
                    evaluation_type = EvaluationType.NL_ASSERTIONS
                else:
                    evaluation_type = EvaluationType.ENV  # Default fallback
            else:
                # Multiple reward bases - use ALL evaluation
                evaluation_type = EvaluationType.ALL
        
        # Test with the appropriate evaluation type
        reward_info = evaluate_simulation(
            simulation=empty_simulation,
            task=task,
            evaluation_type=evaluation_type,
            solo_mode=False,
            domain="telecom"
        )
        
        # Get detailed information about the task
        has_actions = bool(task.evaluation_criteria.actions) if task.evaluation_criteria else False
        has_env_assertions = bool(task.evaluation_criteria.env_assertions) if task.evaluation_criteria else False
        reward_basis = task.evaluation_criteria.reward_basis if task.evaluation_criteria else None
        
        # Check if env assertions are met with empty trajectory
        env_assertions_met = None
        if reward_info.env_assertions:
            env_assertions_met = all(check.met for check in reward_info.env_assertions)
        
        # Check if actions are met with empty trajectory
        actions_met = None
        if reward_info.action_checks:
            actions_met = all(check.action_match for check in reward_info.action_checks)
        
        result = {
            'task_id': task.id,
            'reward': reward_info.reward,
            'evaluation_type_used': evaluation_type.value,
            'db_match': reward_info.db_check.db_match if reward_info.db_check else None,
            'db_reward': reward_info.db_check.db_reward if reward_info.db_check else None,
            'env_assertions_met': env_assertions_met,
            'actions_met': actions_met,
            'has_actions': has_actions,
            'has_env_assertions': has_env_assertions,
            'reward_basis': reward_basis,
            'num_actions': len(task.evaluation_criteria.actions) if task.evaluation_criteria and task.evaluation_criteria.actions else 0,
            'num_env_assertions': len(task.evaluation_criteria.env_assertions) if task.evaluation_criteria and task.evaluation_criteria.env_assertions else 0,
            'description': task.description.purpose if task.description else None,
            'reward_breakdown': reward_info.reward_breakdown
        }
        
        results.append(result)
    
    return results

def print_analysis(results):
    """Print detailed analysis of the results."""
    
    # Filter tasks that get reward=1.0 with empty trajectory
    auto_reward_tasks = [r for r in results if r['reward'] == 1.0]
    
    print(f"\nTASKS THAT GET REWARD=1.0 WITH EMPTY TRAJECTORY:")
    print("=" * 80)
    print(f"Found {len(auto_reward_tasks)} out of {len(results)} tasks")
    print()
    
    if auto_reward_tasks:
        for i, result in enumerate(auto_reward_tasks, 1):
            print(f"{i}. Task ID: {result['task_id']}")
            print(f"   Description: {result['description']}")
            print(f"   Reward: {result['reward']}")
            print(f"   Evaluation Type Used: {result['evaluation_type_used']}")
            print(f"   DB Match: {result['db_match']}")
            print(f"   DB Reward: {result['db_reward']}")
            print(f"   Env Assertions Met: {result['env_assertions_met']}")
            print(f"   Actions Met: {result['actions_met']}")
            print(f"   Has Actions: {result['has_actions']} ({result['num_actions']} actions)")
            print(f"   Has Env Assertions: {result['has_env_assertions']} ({result['num_env_assertions']} assertions)")
            print(f"   Reward Basis: {result['reward_basis']}")
            print(f"   Reward Breakdown: {result['reward_breakdown']}")
            print()
    else:
        print("No tasks found that give reward=1.0 with empty trajectory.")
    
    # Analysis by evaluation type used
    print("\nANALYSIS BY EVALUATION TYPE USED:")
    print("=" * 80)
    
    eval_type_counts = {}
    for result in results:
        eval_type = result['evaluation_type_used']
        if eval_type not in eval_type_counts:
            eval_type_counts[eval_type] = {'total': 0, 'auto_reward': 0}
        eval_type_counts[eval_type]['total'] += 1
        if result['reward'] == 1.0:
            eval_type_counts[eval_type]['auto_reward'] += 1
    
    for eval_type, counts in eval_type_counts.items():
        auto_rate = counts['auto_reward'] / counts['total'] * 100
        print(f"{eval_type}: {counts['auto_reward']}/{counts['total']} tasks ({auto_rate:.1f}%) get auto reward")
    
    # Analysis by reward basis
    print("\nANALYSIS BY REWARD BASIS:")
    print("=" * 80)
    
    reward_basis_counts = {}
    for result in results:
        if result['reward_basis']:
            # Convert to string for counting
            basis_str = str(sorted(result['reward_basis']))
            if basis_str not in reward_basis_counts:
                reward_basis_counts[basis_str] = {'total': 0, 'auto_reward': 0}
            reward_basis_counts[basis_str]['total'] += 1
            if result['reward'] == 1.0:
                reward_basis_counts[basis_str]['auto_reward'] += 1
    
    for basis, counts in reward_basis_counts.items():
        auto_rate = counts['auto_reward'] / counts['total'] * 100
        print(f"{basis}: {counts['auto_reward']}/{counts['total']} tasks ({auto_rate:.1f}%) get auto reward")
    
    # Summary statistics
    print(f"\nSUMMARY STATISTICS:")
    print("=" * 80)
    print(f"Total tasks: {len(results)}")
    print(f"Tasks with auto reward (1.0): {len(auto_reward_tasks)}")
    print(f"Auto reward rate: {len(auto_reward_tasks)/len(results)*100:.1f}%")
    
    # Tasks with no evaluation criteria
    no_criteria = [r for r in results if r['reward_basis'] is None]
    print(f"Tasks with no evaluation criteria: {len(no_criteria)}")
    
    # Tasks that use ALL evaluation (multiple reward bases)
    all_eval_tasks = [r for r in results if r['evaluation_type_used'] == 'all']
    print(f"Tasks using ALL evaluation (multiple reward bases): {len(all_eval_tasks)}")

if __name__ == "__main__":
    results = analyze_empty_trajectory_rewards()
    print_analysis(results)
    
    # Also save results to JSON for further analysis
    with open('empty_trajectory_analysis_full.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: empty_trajectory_analysis_full.json")
