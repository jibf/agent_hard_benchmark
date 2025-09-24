#!/usr/bin/env python3
"""
Script to identify tasks where all expected actions are READ-only (don't modify the database).
These tasks will always have DB reward = 1.0 regardless of agent performance.
"""

import json
import os
from typing import Dict, List, Any, Set

def load_tasks(domain_path: str) -> List[Dict[str, Any]]:
    """Load tasks from tasks.json file."""
    tasks_file = os.path.join(domain_path, "tasks.json")
    with open(tasks_file, 'r') as f:
        return json.load(f)

def get_read_only_tools(domain: str) -> Set[str]:
    """Get the set of READ-only tools for a domain."""
    # Based on the grep results, these are the READ-only tools
    if domain == "retail":
        return {
            "find_user_id_by_name_zip",
            "find_user_id_by_email", 
            "get_order_details",
            "get_product_details",
            "get_user_details",
            "list_all_product_types"
        }
    elif domain == "airline":
        return {
            "get_reservation_details",
            "get_user_details",
            "get_flight_details",
            "search_flights",
            "get_airport_info",
            "get_certificate_details"
        }
    else:
        return set()

def get_write_tools(domain: str) -> Set[str]:
    """Get the set of WRITE tools for a domain."""
    if domain == "retail":
        return {
            "cancel_pending_order",
            "exchange_delivered_order_items",
            "modify_pending_order_address",
            "modify_pending_order_items", 
            "modify_pending_order_payment",
            "modify_user_address",
            "return_delivered_order_items"
        }
    elif domain == "airline":
        return {
            "book_reservation",
            "cancel_reservation",
            "update_reservation_flights",
            "send_certificate",
            "modify_reservation_passengers",
            "update_reservation_payment"
        }
    else:
        return set()

def analyze_task_db_impact(task: Dict[str, Any], domain: str) -> Dict[str, Any]:
    """Analyze whether a task's expected actions modify the database."""
    task_id = task.get("id", "unknown")
    evaluation_criteria = task.get("evaluation_criteria", {})
    actions = evaluation_criteria.get("actions") or []
    
    read_only_tools = get_read_only_tools(domain)
    write_tools = get_write_tools(domain)
    
    action_names = [action.get("name", "") for action in actions]
    
    # Categorize actions
    read_actions = [name for name in action_names if name in read_only_tools]
    write_actions = [name for name in action_names if name in write_tools]
    unknown_actions = [name for name in action_names if name not in read_only_tools and name not in write_tools]
    
    # Determine if all actions are read-only
    all_read_only = len(actions) > 0 and len(write_actions) == 0 and len(unknown_actions) == 0
    
    return {
        "task_id": task_id,
        "total_actions": len(actions),
        "read_actions": read_actions,
        "write_actions": write_actions,
        "unknown_actions": unknown_actions,
        "all_read_only": all_read_only,
        "has_db_impact": len(write_actions) > 0,
        "description": task.get("description", {}).get("purpose", "No description")
    }

def analyze_domain_read_only_tasks(domain_name: str) -> Dict[str, Any]:
    """Analyze read-only tasks in a domain."""
    domain_path = os.path.join(".", domain_name)
    
    if not os.path.exists(domain_path):
        print(f"Domain path {domain_path} does not exist")
        return {}
    
    tasks = load_tasks(domain_path)
    print(f"\n=== {domain_name.upper()} DOMAIN - READ-ONLY TASK ANALYSIS ===")
    print(f"Total tasks: {len(tasks)}")
    
    results = []
    read_only_count = 0
    has_db_impact_count = 0
    no_actions_count = 0
    
    for task in tasks:
        analysis = analyze_task_db_impact(task, domain_name)
        results.append(analysis)
        
        if analysis["total_actions"] == 0:
            no_actions_count += 1
        elif analysis["all_read_only"]:
            read_only_count += 1
        elif analysis["has_db_impact"]:
            has_db_impact_count += 1
    
    print(f"Tasks with no actions: {no_actions_count}")
    print(f"Tasks with all READ-only actions: {read_only_count}")
    print(f"Tasks with WRITE actions (DB impact): {has_db_impact_count}")
    print(f"Tasks with mixed/unknown actions: {len(tasks) - no_actions_count - read_only_count - has_db_impact_count}")
    
    return {
        "domain": domain_name,
        "total_tasks": len(tasks),
        "summary": {
            "no_actions": no_actions_count,
            "all_read_only": read_only_count,
            "has_db_impact": has_db_impact_count,
            "mixed_unknown": len(tasks) - no_actions_count - read_only_count - has_db_impact_count
        },
        "tasks": results
    }

def print_read_only_examples(domain_results: Dict[str, Any], max_examples: int = 5):
    """Print examples of read-only tasks."""
    domain_name = domain_results["domain"]
    tasks = domain_results["tasks"]
    
    print(f"\n=== READ-ONLY TASK EXAMPLES FOR {domain_name.upper()} ===")
    
    read_only_examples = [t for t in tasks if t["all_read_only"]][:max_examples]
    
    if read_only_examples:
        for i, task in enumerate(read_only_examples, 1):
            print(f"\nExample {i} (Task ID: {task['task_id']}):")
            print(f"Description: {task['description']}")
            print(f"Actions ({task['total_actions']}): {task['read_actions']}")
            print(f"⚠️  This task will ALWAYS get DB reward = 1.0 (no DB changes expected)")
    else:
        print("No read-only tasks found in this domain.")

def main():
    """Main function to analyze read-only tasks in both domains."""
    domains = ["airline", "retail"]
    all_results = {}
    
    for domain in domains:
        results = analyze_domain_read_only_tasks(domain)
        all_results[domain] = results
        print_read_only_examples(results)
    
    # Summary across domains
    print(f"\n=== CROSS-DOMAIN READ-ONLY TASK SUMMARY ===")
    total_no_actions = sum(r["summary"]["no_actions"] for r in all_results.values())
    total_read_only = sum(r["summary"]["all_read_only"] for r in all_results.values())
    total_has_db_impact = sum(r["summary"]["has_db_impact"] for r in all_results.values())
    total_mixed = sum(r["summary"]["mixed_unknown"] for r in all_results.values())
    total_tasks = sum(r["total_tasks"] for r in all_results.values())
    
    print(f"Total tasks across domains: {total_tasks}")
    print(f"Tasks with no actions: {total_no_actions}")
    print(f"Tasks with all READ-only actions: {total_read_only}")
    print(f"Tasks with WRITE actions (DB impact): {total_has_db_impact}")
    print(f"Tasks with mixed/unknown actions: {total_mixed}")
    
    print(f"\n⚠️  CRITICAL FINDING:")
    print(f"   {total_read_only} tasks will ALWAYS get DB reward = 1.0")
    print(f"   because they only contain READ-only actions that don't modify the database.")
    
    # Save results
    output_file = "read_only_tasks_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
