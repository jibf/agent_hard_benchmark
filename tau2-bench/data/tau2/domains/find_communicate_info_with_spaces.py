#!/usr/bin/env python3
"""
Script to find tasks with communicate_info that contains spaces.
This helps identify tasks affected by the space handling issue in COMMUNICATE evaluator.
"""

import json
from typing import Dict, List, Any

def load_tasks(domain: str) -> List[Dict[str, Any]]:
    """Load tasks from tasks.json file."""
    tasks_file = f"{domain}/tasks.json"
    with open(tasks_file, 'r') as f:
        return json.load(f)

def find_communicate_info_with_spaces():
    """Find all tasks with communicate_info containing spaces."""
    
    results = {}
    
    for domain in ['airline', 'retail']:
        print(f"\n{'='*60}")
        print(f"{domain.upper()} DOMAIN - COMMUNICATE_INFO WITH SPACES")
        print(f"{'='*60}")
        
        tasks = load_tasks(domain)
        tasks_with_spaces = []
        total_tasks_with_communicate = 0
        
        for task in tasks:
            task_id = task.get('id', 'unknown')
            evaluation_criteria = task.get('evaluation_criteria', {})
            communicate_info = evaluation_criteria.get('communicate_info', [])
            
            if communicate_info:  # Task has communicate_info
                total_tasks_with_communicate += 1
                
                # Check if any communicate_info item contains spaces
                items_with_spaces = []
                for item in communicate_info:
                    if ' ' in str(item):
                        items_with_spaces.append(item)
                
                if items_with_spaces:
                    tasks_with_spaces.append({
                        'task_id': task_id,
                        'all_communicate_info': communicate_info,
                        'items_with_spaces': items_with_spaces,
                        'description': task.get('description', {}).get('purpose', 'No description')
                    })
        
        print(f"Total tasks with communicate_info: {total_tasks_with_communicate}")
        print(f"Tasks with spaces in communicate_info: {len(tasks_with_spaces)}")
        
        if tasks_with_spaces:
            print(f"\n📋 TASKS WITH SPACES IN COMMUNICATE_INFO:")
            print("-" * 50)
            
            for i, task_info in enumerate(tasks_with_spaces, 1):
                print(f"\n{i}. Task ID: {task_info['task_id']}")
                print(f"   All communicate_info: {task_info['all_communicate_info']}")
                print(f"   Items with spaces: {task_info['items_with_spaces']}")
                
                # Show potential issues
                for item in task_info['items_with_spaces']:
                    item_no_spaces = str(item).replace(' ', '')
                    print(f"   ⚠️  '{item}' → '{item_no_spaces}' (space removed)")
                
                # Show description
                description = task_info['description']
                if description and len(description) > 100:
                    description = description[:100] + "..."
                print(f"   Description: {description}")
        else:
            print(f"\n✅ No tasks found with spaces in communicate_info")
        
        results[domain] = {
            'total_tasks_with_communicate': total_tasks_with_communicate,
            'tasks_with_spaces': len(tasks_with_spaces),
            'tasks_with_spaces_details': tasks_with_spaces
        }
    
    # Cross-domain summary
    print(f"\n{'='*80}")
    print("CROSS-DOMAIN SUMMARY")
    print(f"{'='*80}")
    
    total_communicate_tasks = sum(r['total_tasks_with_communicate'] for r in results.values())
    total_spaces_tasks = sum(r['tasks_with_spaces'] for r in results.values())
    
    print(f"Total tasks with communicate_info: {total_communicate_tasks}")
    print(f"Tasks with spaces in communicate_info: {total_spaces_tasks}")
    print(f"Percentage affected by space issue: {(total_spaces_tasks/total_communicate_tasks*100):.1f}%")
    
    if total_spaces_tasks > 0:
        print(f"\n⚠️  CRITICAL FINDING:")
        print(f"   {total_spaces_tasks} tasks are affected by the space handling issue")
        print(f"   in the COMMUNICATE evaluator. These tasks may fail evaluation")
        print(f"   even when the agent communicates the correct information.")
    
    return results

def analyze_specific_examples():
    """Analyze specific examples of space issues."""
    print(f"\n{'='*80}")
    print("DETAILED SPACE ISSUE ANALYSIS")
    print(f"{'='*80}")
    
    results = find_communicate_info_with_spaces()
    
    all_spaces_examples = []
    for domain, data in results.items():
        for task_info in data['tasks_with_spaces_details']:
            all_spaces_examples.append((domain, task_info))
    
    if all_spaces_examples:
        print(f"\n🔍 EXAMPLES OF SPACE ISSUES:")
        print("-" * 40)
        
        for domain, task_info in all_spaces_examples:
            print(f"\n{domain.upper()} Task {task_info['task_id']}:")
            for item in task_info['items_with_spaces']:
                print(f"  Expected: '{item}'")
                print(f"  Agent says: '64 GB' → ❌ FAILS (space present)")
                print(f"  Agent says: '64GB'  → ✅ PASSES (no space)")
                print(f"  Issue: Current evaluator can't handle space variations")

def main():
    """Main function."""
    results = find_communicate_info_with_spaces()
    analyze_specific_examples()
    
    # Save results
    output_file = "communicate_info_spaces_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Results saved to: {output_file}")

if __name__ == "__main__":
    main()
