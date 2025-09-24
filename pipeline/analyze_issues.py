#!/usr/bin/env python3
"""
Analyze issues from the results JSONL file and compile summaries.
"""

import json
import sys
import os
from collections import defaultdict, Counter
from typing import Dict, List, Any

def load_results_from_file(results_file: str) -> List[Dict[str, Any]]:
    """Load all results from the JSONL file."""
    results = []
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                task_data = json.loads(line.strip())
                results.append(task_data)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    return results

def analyze_issues(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze issues and compile summaries."""
    
    # Initialize counters
    total_tasks = len(results)
    flawed_tasks = []
    
    # Task type analysis
    task_type_stats = defaultdict(lambda: {
        'total': 0,
        'flawed': 0,
        'flawed_tasks': []
    })
    
    # Issue type analysis
    issue_type_stats = defaultdict(lambda: {
        'count': 0,
        'tasks': []
    })
    
    # Benchmark analysis
    benchmark_stats = defaultdict(lambda: {
        'total': 0,
        'flawed': 0,
        'flawed_tasks': []
    })
    
    # Process each task
    for task_data in results:
        benchmark = task_data.get('benchmark', 'Unknown')
        task_name = task_data.get('task_name', 'Unknown')
        question_id = task_data.get('question_id', 'Unknown')
        
        # Update benchmark stats
        benchmark_stats[benchmark]['total'] += 1
        
        # Update task type stats
        task_type_stats[task_name]['total'] += 1
        
        # Check if flawed
        llm_judge_output = task_data.get('llm_judge_output', {})
        specific_filter = llm_judge_output.get('specific_filter', {})
        is_flawed = specific_filter.get('is_flawed', False)
        
        if is_flawed:
            flawed_tasks.append(task_data)
            
            # Update flawed counts
            benchmark_stats[benchmark]['flawed'] += 1
            benchmark_stats[benchmark]['flawed_tasks'].append({
                'question_id': question_id,
                'error_category': specific_filter.get('error_category', 'Unknown'),
                'reasoning_summary': specific_filter.get('reasoning_summary', '')
            })
            
            task_type_stats[task_name]['flawed'] += 1
            task_type_stats[task_name]['flawed_tasks'].append({
                'question_id': question_id,
                'error_category': specific_filter.get('error_category', 'Unknown'),
                'reasoning_summary': specific_filter.get('reasoning_summary', '')
            })
            
            # Update issue type stats
            error_category = specific_filter.get('error_category', 'Unknown')
            issue_type_stats[error_category]['count'] += 1
            issue_type_stats[error_category]['tasks'].append({
                'benchmark': benchmark,
                'task_name': task_name,
                'question_id': question_id,
                'reasoning_summary': specific_filter.get('reasoning_summary', '')
            })
    
    return {
        'total_tasks': total_tasks,
        'total_flawed': len(flawed_tasks),
        'flawed_percentage': (len(flawed_tasks) / total_tasks * 100) if total_tasks > 0 else 0,
        'benchmark_stats': dict(benchmark_stats),
        'task_type_stats': dict(task_type_stats),
        'issue_type_stats': dict(issue_type_stats),
        'flawed_tasks': flawed_tasks
    }

def print_summary(analysis: Dict[str, Any]):
    """Print the analysis summary."""
    
    print("=" * 80)
    print("ISSUE ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Overall statistics
    print(f"Total Tasks: {analysis['total_tasks']}")
    print(f"Flawed Tasks: {analysis['total_flawed']}")
    print(f"Flawed Percentage: {analysis['flawed_percentage']:.2f}%")
    print()
    
    # Benchmark breakdown
    print("BENCHMARK BREAKDOWN:")
    print("-" * 40)
    for benchmark, stats in analysis['benchmark_stats'].items():
        flawed_pct = (stats['flawed'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"{benchmark}:")
        print(f"  Total: {stats['total']}")
        print(f"  Flawed: {stats['flawed']} ({flawed_pct:.2f}%)")
        if stats['flawed'] > 0:
            print(f"  Flawed Tasks: {[task['question_id'] for task in stats['flawed_tasks']]}")
        print()
    
    # Task type breakdown
    print("TASK TYPE BREAKDOWN:")
    print("-" * 40)
    for task_type, stats in analysis['task_type_stats'].items():
        flawed_pct = (stats['flawed'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"{task_type}:")
        print(f"  Total: {stats['total']}")
        print(f"  Flawed: {stats['flawed']} ({flawed_pct:.2f}%)")
        if stats['flawed'] > 0:
            print(f"  Flawed Tasks: {[task['question_id'] for task in stats['flawed_tasks']]}")
        print()
    
    # Issue type breakdown
    print("ISSUE TYPE BREAKDOWN:")
    print("-" * 40)
    for issue_type, stats in analysis['issue_type_stats'].items():
        print(f"{issue_type}: {stats['count']} occurrences")
        print(f"  Affected Tasks:")
        for task in stats['tasks']:
            print(f"    - {task['benchmark']}/{task['task_name']}/{task['question_id']}")
            print(f"      Summary: {task['reasoning_summary']}")
        print()
    
    # Top problematic task types
    print("TOP PROBLEMATIC TASK TYPES (by flawed percentage):")
    print("-" * 50)
    task_flawed_pct = []
    for task_type, stats in analysis['task_type_stats'].items():
        if stats['total'] > 0:
            flawed_pct = stats['flawed'] / stats['total'] * 100
            task_flawed_pct.append((task_type, flawed_pct, stats['flawed'], stats['total']))
    
    task_flawed_pct.sort(key=lambda x: x[1], reverse=True)
    for task_type, flawed_pct, flawed_count, total_count in task_flawed_pct:
        if flawed_count > 0:
            print(f"{task_type}: {flawed_pct:.2f}% ({flawed_count}/{total_count})")
    print()
    
    # Issue type frequency
    print("ISSUE TYPE FREQUENCY:")
    print("-" * 30)
    issue_counts = [(issue_type, stats['count']) for issue_type, stats in analysis['issue_type_stats'].items()]
    issue_counts.sort(key=lambda x: x[1], reverse=True)
    for issue_type, count in issue_counts:
        print(f"{issue_type}: {count}")

def print_detailed_flawed_tasks(analysis: Dict[str, Any]):
    """Print detailed information about each flawed task."""
    
    print("\n" + "=" * 80)
    print("DETAILED FLAWED TASKS")
    print("=" * 80)
    
    for i, task in enumerate(analysis['flawed_tasks'], 1):
        print(f"\n{i}. {task.get('benchmark', 'Unknown')}/{task.get('task_name', 'Unknown')}/{task.get('question_id', 'Unknown')}")
        
        llm_judge_output = task.get('llm_judge_output', {})
        specific_filter = llm_judge_output.get('specific_filter', {})
        
        print(f"   Error Category: {specific_filter.get('error_category', 'Unknown')}")
        print(f"   Reasoning: {specific_filter.get('reasoning', 'N/A')}")
        print(f"   Summary: {specific_filter.get('reasoning_summary', 'N/A')}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_issues.py <results_file.jsonl>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    if not os.path.exists(results_file):
        print(f"Error: Results file '{results_file}' not found")
        sys.exit(1)
    
    print(f"Loading results from: {results_file}")
    results = load_results_from_file(results_file)
    
    print(f"Analyzing {len(results)} tasks...")
    analysis = analyze_issues(results)
    
    print_summary(analysis)
    
    # Ask if user wants detailed view
    print("\n" + "=" * 80)
    response = input("Show detailed flawed tasks? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        print_detailed_flawed_tasks(analysis)

if __name__ == "__main__":
    main()




