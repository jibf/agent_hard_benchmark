#!/usr/bin/env python3
"""
Script to verify the actual reward conditions for airline and retail tasks.
This corrects the previous analysis by using the proper default reward_basis.
"""

import json
import os
from typing import Dict, List, Any

def load_tasks(domain_path: str) -> List[Dict[str, Any]]:
    """Load tasks from tasks.json file."""
    tasks_file = os.path.join(domain_path, "tasks.json")
    with open(tasks_file, 'r') as f:
        return json.load(f)

def get_actual_reward_basis(task: Dict[str, Any]) -> List[str]:
    """
    Get the actual reward basis for a task.
    Default is [DB, COMMUNICATE] if not specified.
    """
    evaluation_criteria = task.get("evaluation_criteria", {})
    
    # Check if reward_basis is explicitly specified
    if "reward_basis" in evaluation_criteria:
        return evaluation_criteria["reward_basis"]
    
    # Default behavior: [DB, COMMUNICATE] if not specified
    return ["DB", "COMMUNICATE"]

def analyze_task_reward_conditions(task: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze reward conditions for a single task."""
    task_id = task.get("id", "unknown")
    evaluation_criteria = task.get("evaluation_criteria", {})
    
    # Get actual reward basis
    reward_basis = get_actual_reward_basis(task)
    
    # Extract criteria (handle None values)
    actions = evaluation_criteria.get("actions") or []
    communicate_info = evaluation_criteria.get("communicate_info") or []
    nl_assertions = evaluation_criteria.get("nl_assertions") or []
    env_assertions = evaluation_criteria.get("env_assertions") or []
    
    # Determine what will actually be evaluated
    actual_evaluation = {
        "DB": len(actions) > 0,
        "COMMUNICATE": len(communicate_info) > 0,
        "NL_ASSERTION": len(nl_assertions) > 0,
        "ENV_ASSERTION": len(env_assertions) > 0,
        "ACTION": len(actions) > 0  # Actions are evaluated as both DB and ACTION
    }
    
    return {
        "task_id": task_id,
        "reward_basis": reward_basis,
        "actual_evaluation": actual_evaluation,
        "criteria_present": {
            "actions": len(actions),
            "communicate_info": len(communicate_info),
            "nl_assertions": len(nl_assertions),
            "env_assertions": len(env_assertions)
        },
        "description": task.get("description", {}).get("purpose", "No description")
    }

def analyze_domain_corrected(domain_name: str) -> Dict[str, Any]:
    """Analyze a domain with corrected reward basis understanding."""
    domain_path = os.path.join(".", domain_name)
    
    if not os.path.exists(domain_path):
        print(f"Domain path {domain_path} does not exist")
        return {}
    
    tasks = load_tasks(domain_path)
    print(f"\n=== {domain_name.upper()} DOMAIN (CORRECTED ANALYSIS) ===")
    print(f"Total tasks: {len(tasks)}")
    
    results = []
    db_only_count = 0
    communicate_only_count = 0
    db_and_communicate_count = 0
    nl_only_count = 0
    other_count = 0
    
    for task in tasks:
        analysis = analyze_task_reward_conditions(task)
        results.append(analysis)
        
        # Count based on actual evaluation criteria present
        actual = analysis["actual_evaluation"]
        criteria = analysis["criteria_present"]
        
        if actual["DB"] and not actual["COMMUNICATE"] and not actual["NL_ASSERTION"]:
            db_only_count += 1
        elif actual["COMMUNICATE"] and not actual["DB"] and not actual["NL_ASSERTION"]:
            communicate_only_count += 1
        elif actual["DB"] and actual["COMMUNICATE"]:
            db_and_communicate_count += 1
        elif actual["NL_ASSERTION"] and not actual["DB"] and not actual["COMMUNICATE"]:
            nl_only_count += 1
        else:
            other_count += 1
    
    print(f"DB only: {db_only_count}")
    print(f"COMMUNICATE only: {communicate_only_count}")
    print(f"DB & COMMUNICATE: {db_and_communicate_count}")
    print(f"NL_ASSERTION only: {nl_only_count}")
    print(f"Other combinations: {other_count}")
    
    return {
        "domain": domain_name,
        "total_tasks": len(tasks),
        "summary": {
            "db_only": db_only_count,
            "communicate_only": communicate_only_count,
            "db_and_communicate": db_and_communicate_count,
            "nl_only": nl_only_count,
            "other": other_count
        },
        "tasks": results
    }

def print_examples(domain_results: Dict[str, Any], max_examples: int = 3):
    """Print examples of different reward condition types."""
    domain_name = domain_results["domain"]
    tasks = domain_results["tasks"]
    
    print(f"\n=== EXAMPLES FOR {domain_name.upper()} ===")
    
    # Find examples of each type
    db_only_examples = [t for t in tasks if t["actual_evaluation"]["DB"] and not t["actual_evaluation"]["COMMUNICATE"] and not t["actual_evaluation"]["NL_ASSERTION"]][:max_examples]
    communicate_examples = [t for t in tasks if t["actual_evaluation"]["COMMUNICATE"]][:max_examples]
    nl_examples = [t for t in tasks if t["actual_evaluation"]["NL_ASSERTION"] and not t["actual_evaluation"]["DB"] and not t["actual_evaluation"]["COMMUNICATE"]][:max_examples]
    
    if db_only_examples:
        print(f"\n--- DB ONLY Examples ---")
        for i, task in enumerate(db_only_examples, 1):
            print(f"\nExample {i} (Task ID: {task['task_id']}):")
            print(f"Description: {task['description']}")
            print(f"Reward Basis: {task['reward_basis']}")
            print(f"Actions: {task['criteria_present']['actions']}")
    
    if communicate_examples:
        print(f"\n--- COMMUNICATE Examples ---")
        for i, task in enumerate(communicate_examples, 1):
            print(f"\nExample {i} (Task ID: {task['task_id']}):")
            print(f"Description: {task['description']}")
            print(f"Reward Basis: {task['reward_basis']}")
            print(f"Communicate Info: {task['criteria_present']['communicate_info']}")
            print(f"Actions: {task['criteria_present']['actions']}")
    
    if nl_examples:
        print(f"\n--- NL_ASSERTION ONLY Examples ---")
        for i, task in enumerate(nl_examples, 1):
            print(f"\nExample {i} (Task ID: {task['task_id']}):")
            print(f"Description: {task['description']}")
            print(f"Reward Basis: {task['reward_basis']}")
            print(f"NL Assertions: {task['criteria_present']['nl_assertions']}")

def main():
    """Main function to analyze both domains with corrected understanding."""
    domains = ["airline", "retail"]
    all_results = {}
    
    for domain in domains:
        results = analyze_domain_corrected(domain)
        all_results[domain] = results
        print_examples(results)
    
    # Summary across domains
    print(f"\n=== CORRECTED CROSS-DOMAIN SUMMARY ===")
    total_db_only = sum(r["summary"]["db_only"] for r in all_results.values())
    total_communicate_only = sum(r["summary"]["communicate_only"] for r in all_results.values())
    total_db_and_communicate = sum(r["summary"]["db_and_communicate"] for r in all_results.values())
    total_nl_only = sum(r["summary"]["nl_only"] for r in all_results.values())
    total_other = sum(r["summary"]["other"] for r in all_results.values())
    total_tasks = sum(r["total_tasks"] for r in all_results.values())
    
    print(f"Total tasks across domains: {total_tasks}")
    print(f"DB only: {total_db_only}")
    print(f"COMMUNICATE only: {total_communicate_only}")
    print(f"DB & COMMUNICATE: {total_db_and_communicate}")
    print(f"NL_ASSERTION only: {total_nl_only}")
    print(f"Other combinations: {total_other}")
    
    # Save corrected results
    output_file = "corrected_reward_conditions_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCorrected results saved to: {output_file}")

if __name__ == "__main__":
    main()
