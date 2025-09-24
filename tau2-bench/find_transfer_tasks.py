#!/usr/bin/env python3

import json
import sys
import os
sys.path.append('/nethome/hsuh45/agenthard/tau2-bench/src')

from tau2.data_model.tasks import Task

def find_transfer_to_human_tasks():
    """
    Find all task IDs that require the 'transfer_to_human_agents' action.
    """
    # Load telecom tasks
    with open('/nethome/hsuh45/agenthard/tau2-bench/data/tau2/domains/telecom/tasks.json', 'r') as f:
        tasks_data = json.load(f)
    
    # Convert to Task objects
    tasks = [Task.model_validate(task_data) for task_data in tasks_data]
    
    transfer_tasks = []
    
    for task in tasks:
        if task.evaluation_criteria and task.evaluation_criteria.actions:
            for action in task.evaluation_criteria.actions:
                if action.name == "transfer_to_human_agents":
                    transfer_tasks.append({
                        'task_id': task.id,
                        'action_id': action.action_id,
                        'requestor': action.requestor,
                        'description': task.description.purpose if task.description else None
                    })
                    break  # Found the transfer action, no need to check other actions
    
    return transfer_tasks

if __name__ == "__main__":
    transfer_tasks = find_transfer_to_human_tasks()
    
    print("TASKS THAT REQUIRE 'transfer_to_human_agents' ACTION:")
    print("=" * 80)
    print(f"Found {len(transfer_tasks)} tasks")
    print()
    
    for i, task in enumerate(transfer_tasks, 1):
        print(f"{i}. Task ID: {task['task_id']}")
        print(f"   Action ID: {task['action_id']}")
        print(f"   Requestor: {task['requestor']}")
        print(f"   Description: {task['description']}")
        print()
    
    # Also print just the task IDs for easy copying
    print("TASK IDs ONLY:")
    print("=" * 40)
    for task in transfer_tasks:
        print(f"'{task['task_id']}'")
