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
    Uses individual evaluation types and combines them properly.
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
        
        # Calculate reward based on individual evaluation types
        if not task.evaluation_criteria or not task.evaluation_criteria.reward_basis:
            final_reward = 1.0  # No evaluation criteria
            reward_breakdown = {}
            env_assertions_met = None
            actions_met = None
            evaluation_types_used = []
        else:
            reward_basis = set(task.evaluation_criteria.reward_basis)
            final_reward = 1.0
            reward_breakdown = {}
            env_assertions_met = None
            actions_met = None
            evaluation_types_used = []
            
            # Run individual evaluators based on reward basis
            if 'ENV_ASSERTION' in reward_basis or 'DB' in reward_basis:
                env_reward_info = evaluate_simulation(
                    simulation=empty_simulation,
                    task=task,
                    evaluation_type=EvaluationType.ENV,
                    solo_mode=False,
                    domain="telecom"
                )
                final_reward *= env_reward_info.reward
                if env_reward_info.reward_breakdown:
                    reward_breakdown.update(env_reward_info.reward_breakdown)
                if env_reward_info.env_assertions:
                    env_assertions_met = all(check.met for check in env_reward_info.env_assertions)
                evaluation_types_used.append('ENV')
            
            if 'ACTION' in reward_basis:
                action_reward_info = evaluate_simulation(
                    simulation=empty_simulation,
                    task=task,
                    evaluation_type=EvaluationType.ACTION,
                    solo_mode=False,
                    domain="telecom"
                )
                final_reward *= action_reward_info.reward
                if action_reward_info.reward_breakdown:
                    reward_breakdown.update(action_reward_info.reward_breakdown)
                if action_reward_info.action_checks:
                    actions_met = all(check.action_match for check in action_reward_info.action_checks)
                evaluation_types_used.append('ACTION')
            
            if 'COMMUNICATE' in reward_basis:
                comm_reward_info = evaluate_simulation(
                    simulation=empty_simulation,
                    task=task,
                    evaluation_type=EvaluationType.COMMUNICATE,
                    solo_mode=False,
                    domain="telecom"
                )
                final_reward *= comm_reward_info.reward
                if comm_reward_info.reward_breakdown:
                    reward_breakdown.update(comm_reward_info.reward_breakdown)
                evaluation_types_used.append('COMMUNICATE')
            
            if 'NL_ASSERTION' in reward_basis:
                nl_reward_info = evaluate_simulation(
                    simulation=empty_simulation,
                    task=task,
                    evaluation_type=EvaluationType.NL_ASSERTIONS,
                    solo_mode=False,
                    domain="telecom"
                )
                final_reward *= nl_reward_info.reward
                if nl_reward_info.reward_breakdown:
                    reward_breakdown.update(nl_reward_info.reward_breakdown)
                evaluation_types_used.append('NL_ASSERTIONS')
        
        # Get detailed information about the task
        has_actions = bool(task.evaluation_criteria.actions) if task.evaluation_criteria else False
        has_env_assertions = bool(task.evaluation_criteria.env_assertions) if task.evaluation_criteria else False
        reward_basis = task.evaluation_criteria.reward_basis if task.evaluation_criteria else None
        
        result = {
            'task_id': task.id,
            'reward': final_reward,
            'evaluation_types_used': evaluation_types_used,
            'env_assertions_met': env_assertions_met,
            'actions_met': actions_met,
            'has_actions': has_actions,
            'has_env_assertions': has_env_assertions,
            'reward_basis': reward_basis,
            'num_actions': len(task.evaluation_criteria.actions) if task.evaluation_criteria and task.evaluation_criteria.actions else 0,
            'num_env_assertions': len(task.evaluation_criteria.env_assertions) if task.evaluation_criteria and task.evaluation_criteria.env_assertions else 0,
            'description': task.description.purpose if task.description else None,
            'reward_breakdown': reward_breakdown
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
            print(f"   Evaluation Types Used: {result['evaluation_types_used']}")
            print(f"   Env Assertions Met: {result['env_assertions_met']}")
            print(f"   Actions Met: {result['actions_met']}")
            print(f"   Has Actions: {result['has_actions']} ({result['num_actions']} actions)")
            print(f"   Has Env Assertions: {result['has_env_assertions']} ({result['num_env_assertions']} assertions)")
            print(f"   Reward Basis: {result['reward_basis']}")
            print(f"   Reward Breakdown: {result['reward_breakdown']}")
            print()
    else:
        print("No tasks found that give reward=1.0 with empty trajectory.")
    
    # Analysis by evaluation types used
    print("\nANALYSIS BY EVALUATION TYPES USED:")
    print("=" * 80)
    
    eval_type_counts = {}
    for result in results:
        eval_types = tuple(sorted(result['evaluation_types_used']))
        if eval_types not in eval_type_counts:
            eval_type_counts[eval_types] = {'total': 0, 'auto_reward': 0}
        eval_type_counts[eval_types]['total'] += 1
        if result['reward'] == 1.0:
            eval_type_counts[eval_types]['auto_reward'] += 1
    
    for eval_types, counts in eval_type_counts.items():
        auto_rate = counts['auto_reward'] / counts['total'] * 100
        print(f"{eval_types}: {counts['auto_reward']}/{counts['total']} tasks ({auto_rate:.1f}%) get auto reward")
    
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

if __name__ == "__main__":
    results = analyze_empty_trajectory_rewards()
    print_analysis(results)
    
    # Also save results to JSON for further analysis
    with open('empty_trajectory_analysis_corrected.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: empty_trajectory_analysis_corrected.json")
