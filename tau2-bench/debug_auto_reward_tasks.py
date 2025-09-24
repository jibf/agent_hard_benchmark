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

def debug_auto_reward_tasks():
    """
    Debug the two tasks that give auto rewards to understand why.
    """
    # Load telecom tasks
    with open('/nethome/hsuh45/agenthard/tau2-bench/data/tau2/domains/telecom/tasks_small.json', 'r') as f:
        tasks_data = json.load(f)
    
    # Convert to Task objects
    tasks = [Task.model_validate(task_data) for task_data in tasks_data]
    
    # Find the two auto-reward tasks
    auto_reward_task_ids = [
        "[service_issue]lock_sim_card_pin[PERSONA:Hard]",
        "[service_issue]contract_end_suspension[PERSONA:Hard]"
    ]
    
    for task in tasks:
        if task.id in auto_reward_task_ids:
            print(f"\n{'='*80}")
            print(f"DEBUGGING TASK: {task.id}")
            print(f"{'='*80}")
            
            # Create empty simulation
            empty_simulation = SimulationRun(
                id=str(uuid.uuid4()),
                task_id=task.id,
                start_time=get_now(),
                end_time=get_now(),
                duration=0.0,
                termination_reason=TerminationReason.AGENT_STOP,
                messages=[]
            )
            
            # Test with different evaluation types
            print("\n1. ENV Evaluation (what we used before):")
            env_reward_info = evaluate_simulation(
                simulation=empty_simulation,
                task=task,
                evaluation_type=EvaluationType.ENV,
                solo_mode=False,
                domain="telecom"
            )
            print(f"   Reward: {env_reward_info.reward}")
            print(f"   DB Match: {env_reward_info.db_check.db_match if env_reward_info.db_check else None}")
            print(f"   DB Reward: {env_reward_info.db_check.db_reward if env_reward_info.db_check else None}")
            if env_reward_info.env_assertions:
                for i, assertion in enumerate(env_reward_info.env_assertions):
                    print(f"   Env Assertion {i+1}: {assertion.env_assertion.func_name} -> {assertion.met} (reward: {assertion.reward})")
            
            print("\n2. ACTION Evaluation:")
            action_reward_info = evaluate_simulation(
                simulation=empty_simulation,
                task=task,
                evaluation_type=EvaluationType.ACTION,
                solo_mode=False,
                domain="telecom"
            )
            print(f"   Reward: {action_reward_info.reward}")
            if action_reward_info.action_checks:
                for i, check in enumerate(action_reward_info.action_checks):
                    print(f"   Action {i+1}: {check.action.name} -> {check.action_match} (reward: {check.action_reward})")
            
            print("\n3. ALL Evaluation (combined):")
            all_reward_info = evaluate_simulation(
                simulation=empty_simulation,
                task=task,
                evaluation_type=EvaluationType.ALL,
                solo_mode=False,
                domain="telecom"
            )
            print(f"   Reward: {all_reward_info.reward}")
            print(f"   Reward Breakdown: {all_reward_info.reward_breakdown}")
            
            # Print task details
            print(f"\n4. Task Details:")
            print(f"   Reward Basis: {task.evaluation_criteria.reward_basis}")
            print(f"   Expected Actions: {len(task.evaluation_criteria.actions) if task.evaluation_criteria.actions else 0}")
            for i, action in enumerate(task.evaluation_criteria.actions or []):
                print(f"     Action {i+1}: {action.name} by {action.requestor}")
            print(f"   Expected Env Assertions: {len(task.evaluation_criteria.env_assertions) if task.evaluation_criteria.env_assertions else 0}")
            for i, assertion in enumerate(task.evaluation_criteria.env_assertions or []):
                print(f"     Assertion {i+1}: {assertion.func_name} -> {assertion.arguments}")

if __name__ == "__main__":
    debug_auto_reward_tasks()
