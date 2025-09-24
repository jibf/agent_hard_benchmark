#!/usr/bin/env python3
"""
Script to extract evaluation criteria from airline and retail tasks.json files.
Extracts DB ground truth (actions) and COMMUNICATE criteria for each task.
"""

import json
import os
from typing import Dict, List, Any

def load_tasks(domain_path: str) -> List[Dict[str, Any]]:
    """Load tasks from tasks.json file."""
    tasks_file = os.path.join(domain_path, "tasks.json")
    with open(tasks_file, 'r') as f:
        return json.load(f)

def extract_evaluation_criteria(task: Dict[str, Any]) -> Dict[str, Any]:
    """Extract evaluation criteria from a single task."""
    task_id = task.get("id", "unknown")
    evaluation_criteria = task.get("evaluation_criteria", {})
    
    # Extract actions (DB ground truth)
    actions = evaluation_criteria.get("actions", [])
    
    # Extract communicate_info (COMMUNICATE criteria)
    communicate_info = evaluation_criteria.get("communicate_info", [])
    
    # Extract nl_assertions
    nl_assertions = evaluation_criteria.get("nl_assertions", [])
    
    # Determine reward basis
    reward_basis = []
    if actions:
        reward_basis.append("DB")
    if communicate_info:
        reward_basis.append("COMMUNICATE")
    if nl_assertions:
        reward_basis.append("NL_ASSERTION")
    
    return {
        "task_id": task_id,
        "reward_basis": reward_basis,
        "db_ground_truth": actions,
        "communicate_criteria": communicate_info,
        "nl_assertions": nl_assertions,
        "description": task.get("description", {}).get("purpose", "No description")
    }

def analyze_domain(domain_name: str) -> Dict[str, Any]:
    """Analyze a domain and extract all evaluation criteria."""
    domain_path = os.path.join(".", domain_name)
    
    if not os.path.exists(domain_path):
        print(f"Domain path {domain_path} does not exist")
        return {}
    
    tasks = load_tasks(domain_path)
    print(f"\n=== {domain_name.upper()} DOMAIN ===")
    print(f"Total tasks: {len(tasks)}")
    
    results = []
    db_only_count = 0
    communicate_only_count = 0
    db_and_communicate_count = 0
    other_count = 0
    
    for task in tasks:
        criteria = extract_evaluation_criteria(task)
        results.append(criteria)
        
        # Count reward basis types
        reward_basis = criteria["reward_basis"]
        if reward_basis == ["DB"]:
            db_only_count += 1
        elif reward_basis == ["COMMUNICATE"]:
            communicate_only_count += 1
        elif "DB" in reward_basis and "COMMUNICATE" in reward_basis:
            db_and_communicate_count += 1
        else:
            other_count += 1
    
    print(f"DB only: {db_only_count}")
    print(f"COMMUNICATE only: {communicate_only_count}")
    print(f"DB & COMMUNICATE: {db_and_communicate_count}")
    print(f"Other combinations: {other_count}")
    
    return {
        "domain": domain_name,
        "total_tasks": len(tasks),
        "summary": {
            "db_only": db_only_count,
            "communicate_only": communicate_only_count,
            "db_and_communicate": db_and_communicate_count,
            "other": other_count
        },
        "tasks": results
    }

def print_detailed_examples(domain_results: Dict[str, Any], max_examples: int = 3):
    """Print detailed examples of tasks with different reward basis."""
    domain_name = domain_results["domain"]
    tasks = domain_results["tasks"]
    
    print(f"\n=== DETAILED EXAMPLES FOR {domain_name.upper()} ===")
    
    # Find examples of each type
    db_only_examples = [t for t in tasks if t["reward_basis"] == ["DB"]][:max_examples]
    communicate_examples = [t for t in tasks if "COMMUNICATE" in t["reward_basis"]][:max_examples]
    
    if db_only_examples:
        print(f"\n--- DB ONLY Examples ---")
        for i, task in enumerate(db_only_examples, 1):
            print(f"\nExample {i} (Task ID: {task['task_id']}):")
            print(f"Description: {task['description']}")
            print(f"DB Actions ({len(task['db_ground_truth'])}):")
            for action in task['db_ground_truth']:
                print(f"  - {action.get('name', 'unknown')}: {action.get('arguments', {})}")
    
    if communicate_examples:
        print(f"\n--- COMMUNICATE Examples ---")
        for i, task in enumerate(communicate_examples, 1):
            print(f"\nExample {i} (Task ID: {task['task_id']}):")
            print(f"Description: {task['description']}")
            print(f"Reward Basis: {task['reward_basis']}")
            print(f"Communicate Criteria: {task['communicate_criteria']}")
            if task['db_ground_truth']:
                print(f"DB Actions ({len(task['db_ground_truth'])}):")
                for action in task['db_ground_truth']:
                    print(f"  - {action.get('name', 'unknown')}: {action.get('arguments', {})}")

def main():
    """Main function to analyze both domains."""
    domains = ["airline", "retail"]
    all_results = {}
    
    for domain in domains:
        results = analyze_domain(domain)
        all_results[domain] = results
        print_detailed_examples(results)
    
    # Summary across domains
    print(f"\n=== CROSS-DOMAIN SUMMARY ===")
    total_db_only = sum(r["summary"]["db_only"] for r in all_results.values())
    total_communicate_only = sum(r["summary"]["communicate_only"] for r in all_results.values())
    total_db_and_communicate = sum(r["summary"]["db_and_communicate"] for r in all_results.values())
    total_other = sum(r["summary"]["other"] for r in all_results.values())
    total_tasks = sum(r["total_tasks"] for r in all_results.values())
    
    print(f"Total tasks across domains: {total_tasks}")
    print(f"DB only: {total_db_only}")
    print(f"COMMUNICATE only: {total_communicate_only}")
    print(f"DB & COMMUNICATE: {total_db_and_communicate}")
    print(f"Other combinations: {total_other}")
    
    # Save detailed results to JSON
    output_file = "evaluation_criteria_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
