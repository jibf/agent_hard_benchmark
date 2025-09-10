#!/usr/bin/env python3
"""
Multi-turn Task Corrected Analysis
Re-analyze multi-turn tasks with proper function loading to get accurate results
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import concurrent.futures
from functools import partial
import argparse

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from enhanced_functionality_analyzer import EnhancedFunctionalityAnalyzer


def main():
    parser = argparse.ArgumentParser(description='Multi-turn Task Corrected Analysis')
    parser.add_argument('--workers', type=int, default=16, help='Number of parallel workers')
    parser.add_argument('--task', type=str, choices=[
        'multi_turn_base', 
        'multi_turn_long_context', 
        'multi_turn_miss_func', 
        'multi_turn_miss_param',
        'all'
    ], default='all', help='Specific task to analyze or all multi-turn tasks')
    
    args = parser.parse_args()
    
    # Multi-turn tasks to analyze
    if args.task == 'all':
        multi_turn_tasks = [
            'multi_turn_base',
            'multi_turn_long_context', 
            'multi_turn_miss_func',
            'multi_turn_miss_param'
        ]
    else:
        multi_turn_tasks = [args.task]
    
    print("="*80)
    print("MULTI-TURN TASK CORRECTED ANALYSIS")
    print("="*80)
    print(f"Tasks to analyze: {multi_turn_tasks}")
    print(f"Workers: {args.workers}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize analyzer
    analyzer = EnhancedFunctionalityAnalyzer(num_workers=args.workers)
    
    all_results = {}
    start_time = datetime.now()
    
    for idx, task_type in enumerate(multi_turn_tasks, 1):
        print(f"[{idx}/{len(multi_turn_tasks)}] Analyzing {task_type}...")
        print("-" * 60)
        
        # Run analysis
        result = analyzer.analyze_task_parallel(task_type)
        all_results[task_type] = result
        
        # Save individual results
        output_dir = Path("score")
        output_dir.mkdir(exist_ok=True)
        
        task_output = output_dir / f"corrected_multi_turn_analysis_{task_type}.json"
        with open(task_output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to: {task_output}")
        print(f"Cases: {result.get('processed_cases', 0)}")
        print(f"Mismatches: {result.get('mismatch_cases', 0)}")
        print(f"Mismatch rate: {result.get('mismatch_rate', 0):.1f}%")
        print()
    
    # Save combined results
    combined_output = output_dir / "corrected_multi_turn_analysis_all_tasks.json"
    with open(combined_output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # Calculate and display final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("="*80)
    print("CORRECTED MULTI-TURN ANALYSIS SUMMARY")
    print("="*80)
    
    total_cases = 0
    total_mismatches = 0
    
    for task_type, result in all_results.items():
        cases = result.get('processed_cases', 0)
        mismatches = result.get('mismatch_cases', 0)
        rate = result.get('mismatch_rate', 0)
        
        total_cases += cases
        total_mismatches += mismatches
        
        print(f"{task_type:25} - Cases: {cases:4}, Mismatches: {mismatches:4} ({rate:.1f}%)")
    
    if total_cases > 0:
        overall_rate = (total_mismatches / total_cases) * 100
        print(f"\n{'OVERALL':25} - Cases: {total_cases:4}, Mismatches: {total_mismatches:4} ({overall_rate:.1f}%)")
    
    print(f"\nTotal analysis time: {duration}")
    print(f"Analysis completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Combined results: {combined_output}")
    
    # Compare with previous incorrect results
    print("\n" + "="*80)
    print("COMPARISON WITH PREVIOUS RESULTS")
    print("="*80)
    print("Previous analysis (incorrect - EMPTY_FUNCTIONS):")
    print("- multi_turn_base: 99.5% (199/200)")
    print("- multi_turn_long_context: 100.0% (200/200)")
    print("- multi_turn_miss_func: 100.0% (200/200)")
    print("- multi_turn_miss_param: 100.0% (200/200)")
    print("- Overall: 99.9% (799/800)")
    print()
    print("This corrected analysis includes proper function loading from multi_turn_func_doc/")
    print("and should provide much more accurate mismatch detection.")


if __name__ == "__main__":
    main()