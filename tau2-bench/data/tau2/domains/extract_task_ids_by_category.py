#!/usr/bin/env python3
"""
Script to extract task IDs organized by category from the read-only tasks analysis.
"""

import json
from typing import Dict, List, Any

def load_analysis_data() -> Dict[str, Any]:
    """Load the read-only tasks analysis data."""
    with open('read_only_tasks_analysis.json', 'r') as f:
        return json.load(f)

def extract_task_ids_by_category(data: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    """Extract task IDs organized by category for each domain."""
    results = {}
    
    for domain, domain_data in data.items():
        if domain in ['airline', 'retail']:
            tasks = domain_data['tasks']
            
            # Initialize categories
            categories = {
                'all_read_only': [],      # Always DB reward = 1.0
                'has_db_impact': [],      # Can fail DB evaluation
                'no_actions': [],         # No actions
                'mixed_unknown': []       # Mixed/unknown actions
            }
            
            for task in tasks:
                task_id = task['task_id']
                
                if task['total_actions'] == 0:
                    categories['no_actions'].append(task_id)
                elif task['all_read_only']:
                    categories['all_read_only'].append(task_id)
                elif task['has_db_impact']:
                    categories['has_db_impact'].append(task_id)
                else:
                    categories['mixed_unknown'].append(task_id)
            
            results[domain] = categories
    
    return results

def print_task_ids_by_category(results: Dict[str, Dict[str, List[str]]]):
    """Print task IDs organized by category."""
    
    for domain in ['airline', 'retail']:
        if domain not in results:
            continue
            
        print(f"\n{'='*60}")
        print(f"{domain.upper()} DOMAIN")
        print(f"{'='*60}")
        
        categories = results[domain]
        
        # All READ-only actions → Always DB reward = 1.0
        print(f"\n📖 All READ-only actions → Always DB reward = 1.0")
        print(f"   Count: {len(categories['all_read_only'])}")
        if categories['all_read_only']:
            print(f"   Task IDs: {', '.join(categories['all_read_only'])}")
        else:
            print("   Task IDs: None")
        
        # Have WRITE actions → Can fail DB evaluation
        print(f"\n✏️  Have WRITE actions → Can fail DB evaluation")
        print(f"   Count: {len(categories['has_db_impact'])}")
        if categories['has_db_impact']:
            # Print in chunks of 10 for readability
            task_ids = categories['has_db_impact']
            for i in range(0, len(task_ids), 10):
                chunk = task_ids[i:i+10]
                print(f"   Task IDs: {', '.join(chunk)}")
        else:
            print("   Task IDs: None")
        
        # No actions
        print(f"\n🚫 No actions")
        print(f"   Count: {len(categories['no_actions'])}")
        if categories['no_actions']:
            print(f"   Task IDs: {', '.join(categories['no_actions'])}")
        else:
            print("   Task IDs: None")
        
        # Mixed/unknown actions
        print(f"\n❓ Mixed/unknown actions")
        print(f"   Count: {len(categories['mixed_unknown'])}")
        if categories['mixed_unknown']:
            print(f"   Task IDs: {', '.join(categories['mixed_unknown'])}")
        else:
            print("   Task IDs: None")

def print_summary_table(results: Dict[str, Dict[str, List[str]]]):
    """Print a summary table across domains."""
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    
    print(f"{'Category':<40} {'Airline':<15} {'Retail':<15} {'Total':<10}")
    print(f"{'-'*80}")
    
    categories = ['all_read_only', 'has_db_impact', 'no_actions', 'mixed_unknown']
    category_names = [
        'All READ-only actions',
        'Have WRITE actions', 
        'No actions',
        'Mixed/unknown actions'
    ]
    
    for i, category in enumerate(categories):
        airline_count = len(results['airline'][category])
        retail_count = len(results['retail'][category])
        total_count = airline_count + retail_count
        
        print(f"{category_names[i]:<40} {airline_count:<15} {retail_count:<15} {total_count:<10}")

def main():
    """Main function to extract and display task IDs by category."""
    data = load_analysis_data()
    results = extract_task_ids_by_category(data)
    
    print_task_ids_by_category(results)
    print_summary_table(results)
    
    # Save organized results
    output_file = "task_ids_by_category.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Organized task IDs saved to: {output_file}")

if __name__ == "__main__":
    main()
