#!/usr/bin/env python3
"""
Script to analyze communicate_info for tasks with no actions or all read-only actions.
"""

import json
from typing import Dict, List, Any

def load_analysis_data() -> Dict[str, Any]:
    """Load the read-only tasks analysis data."""
    with open('read_only_tasks_analysis.json', 'r') as f:
        return json.load(f)

def load_original_tasks(domain: str) -> List[Dict[str, Any]]:
    """Load original tasks from tasks.json."""
    tasks_file = f"{domain}/tasks.json"
    with open(tasks_file, 'r') as f:
        return json.load(f)

def analyze_communicate_info_for_special_tasks():
    """Analyze communicate_info for tasks with no actions or all read-only actions."""
    
    # Load analysis data
    analysis_data = load_analysis_data()
    
    results = {}
    
    for domain in ['airline', 'retail']:
        print(f"\n{'='*60}")
        print(f"{domain.upper()} DOMAIN - COMMUNICATE_INFO ANALYSIS")
        print(f"{'='*60}")
        
        # Load original tasks
        original_tasks = load_original_tasks(domain)
        task_dict = {task['id']: task for task in original_tasks}
        
        # Get special task categories from analysis
        domain_analysis = analysis_data[domain]['tasks']
        
        no_actions_tasks = []
        read_only_tasks = []
        
        for task_analysis in domain_analysis:
            task_id = task_analysis['task_id']
            original_task = task_dict[task_id]
            
            if task_analysis['total_actions'] == 0:
                no_actions_tasks.append((task_id, original_task))
            elif task_analysis['all_read_only']:
                read_only_tasks.append((task_id, original_task))
        
        # Analyze no actions tasks
        print(f"\n🚫 NO ACTIONS TASKS ({len(no_actions_tasks)} tasks)")
        print("-" * 40)
        
        no_actions_with_communicate = 0
        no_actions_without_communicate = 0
        
        for task_id, task in no_actions_tasks:
            evaluation_criteria = task.get('evaluation_criteria', {})
            communicate_info = evaluation_criteria.get('communicate_info', [])
            nl_assertions = evaluation_criteria.get('nl_assertions', [])
            
            has_communicate = len(communicate_info) > 0
            has_nl_assertions = len(nl_assertions) > 0 if nl_assertions else False
            
            if has_communicate:
                no_actions_with_communicate += 1
                print(f"✅ Task {task_id}: HAS communicate_info ({len(communicate_info)} items)")
                print(f"   Communicate info: {communicate_info}")
                print(f"   NL assertions: {len(nl_assertions) if nl_assertions else 0}")
            else:
                no_actions_without_communicate += 1
                print(f"❌ Task {task_id}: NO communicate_info")
                print(f"   NL assertions: {len(nl_assertions) if nl_assertions else 0}")
            
            # Show description (handle None)
            description = task.get('description', {}).get('purpose', 'No description')
            if description and len(description) > 100:
                description = description[:100] + "..."
            print(f"   Description: {description}")
            print()
        
        # Analyze read-only tasks
        print(f"\n📖 ALL READ-ONLY ACTIONS TASKS ({len(read_only_tasks)} tasks)")
        print("-" * 50)
        
        read_only_with_communicate = 0
        read_only_without_communicate = 0
        
        for task_id, task in read_only_tasks:
            evaluation_criteria = task.get('evaluation_criteria', {})
            communicate_info = evaluation_criteria.get('communicate_info', [])
            nl_assertions = evaluation_criteria.get('nl_assertions', [])
            actions = evaluation_criteria.get('actions', [])
            
            has_communicate = len(communicate_info) > 0
            has_nl_assertions = len(nl_assertions) > 0 if nl_assertions else False
            
            if has_communicate:
                read_only_with_communicate += 1
                print(f"✅ Task {task_id}: HAS communicate_info ({len(communicate_info)} items)")
                print(f"   Communicate info: {communicate_info}")
                print(f"   Actions: {len(actions)} READ-only actions")
                print(f"   NL assertions: {len(nl_assertions) if nl_assertions else 0}")
            else:
                read_only_without_communicate += 1
                print(f"❌ Task {task_id}: NO communicate_info")
                print(f"   Actions: {len(actions)} READ-only actions")
                print(f"   NL assertions: {len(nl_assertions) if nl_assertions else 0}")
            
            # Show description (handle None)
            description = task.get('description', {}).get('purpose', 'No description')
            if description and len(description) > 100:
                description = description[:100] + "..."
            print(f"   Description: {description}")
            print()
        
        # Summary for domain
        print(f"\n📊 {domain.upper()} DOMAIN SUMMARY:")
        print(f"   No Actions Tasks:")
        print(f"     - With communicate_info: {no_actions_with_communicate}")
        print(f"     - Without communicate_info: {no_actions_without_communicate}")
        print(f"   Read-Only Tasks:")
        print(f"     - With communicate_info: {read_only_with_communicate}")
        print(f"     - Without communicate_info: {read_only_without_communicate}")
        
        results[domain] = {
            'no_actions': {
                'with_communicate': no_actions_with_communicate,
                'without_communicate': no_actions_without_communicate,
                'total': len(no_actions_tasks)
            },
            'read_only': {
                'with_communicate': read_only_with_communicate,
                'without_communicate': read_only_without_communicate,
                'total': len(read_only_tasks)
            }
        }
    
    # Cross-domain summary
    print(f"\n{'='*80}")
    print("CROSS-DOMAIN SUMMARY")
    print(f"{'='*80}")
    
    total_no_actions_with = sum(r['no_actions']['with_communicate'] for r in results.values())
    total_no_actions_without = sum(r['no_actions']['without_communicate'] for r in results.values())
    total_read_only_with = sum(r['read_only']['with_communicate'] for r in results.values())
    total_read_only_without = sum(r['read_only']['without_communicate'] for r in results.values())
    
    print(f"NO ACTIONS TASKS:")
    print(f"  - With communicate_info: {total_no_actions_with}")
    print(f"  - Without communicate_info: {total_no_actions_without}")
    print(f"  - Total: {total_no_actions_with + total_no_actions_without}")
    
    print(f"\nREAD-ONLY TASKS:")
    print(f"  - With communicate_info: {total_read_only_with}")
    print(f"  - Without communicate_info: {total_read_only_without}")
    print(f"  - Total: {total_read_only_with + total_read_only_without}")
    
    print(f"\n🔍 KEY INSIGHTS:")
    print(f"  - {total_no_actions_with + total_read_only_with} tasks have communicate_info")
    print(f"  - {total_no_actions_without + total_read_only_without} tasks rely only on NL_ASSERTION")
    print(f"  - Tasks without communicate_info depend entirely on LLM-based NL assertion evaluation")
    
    return results

def main():
    """Main function."""
    results = analyze_communicate_info_for_special_tasks()
    
    # Save results
    output_file = "communicate_info_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Results saved to: {output_file}")

if __name__ == "__main__":
    main()
