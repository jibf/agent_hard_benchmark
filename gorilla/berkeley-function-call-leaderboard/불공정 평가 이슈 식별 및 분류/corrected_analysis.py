"""
Corrected BFCL Analysis - Fix column name and data type issues
"""

import pandas as pd
import numpy as np
from data_loader import load_bfcl_results
from pathlib import Path
import json

def analyze_bfcl_data():
    """Analyze BFCL data with correct column names"""
    
    print("Loading BFCL data...")
    df = load_bfcl_results()
    print(f"Total records: {len(df)}")
    
    # Analysis results
    results = {
        "total_records": len(df),
        "model_count": df['model_name'].nunique() if 'model_name' in df.columns else 0,
        "test_categories": df['test_category'].nunique() if 'test_category' in df.columns else 0
    }
    
    # Check token counts (with proper column names)
    if 'input_token_count' in df.columns:
        # Handle potential list values
        def safe_convert(x):
            try:
                if isinstance(x, list) and len(x) > 0:
                    return float(x[0])
                elif pd.isna(x):
                    return 0
                else:
                    return float(x)
            except:
                return 0
        
        input_tokens = df['input_token_count'].apply(safe_convert)
        valid_input = input_tokens > 0
        results['records_with_input_tokens'] = valid_input.sum()
        results['input_token_percentage'] = (valid_input.sum() / len(df)) * 100
        print(f"Records with input_token_count > 0: {valid_input.sum()} ({results['input_token_percentage']:.1f}%)")
    
    if 'output_token_count' in df.columns:
        # Handle potential list values
        output_tokens = df['output_token_count'].apply(safe_convert)
        valid_output = output_tokens > 0
        results['records_with_output_tokens'] = valid_output.sum()
        results['output_token_percentage'] = (valid_output.sum() / len(df)) * 100
        print(f"Records with output_token_count > 0: {valid_output.sum()} ({results['output_token_percentage']:.1f}%)")
    
    # Check latency (actual column name)
    if 'latency' in df.columns:
        # Handle potential list values
        latency = df['latency'].apply(safe_convert)
        normal_latency = latency > 1  # More than 1 second
        results['records_with_normal_latency'] = normal_latency.sum()
        results['normal_latency_percentage'] = (normal_latency.sum() / len(df)) * 100
        print(f"Records with latency > 1s: {normal_latency.sum()} ({results['normal_latency_percentage']:.1f}%)")
    
    # Analyze by model
    print("\nModel Analysis:")
    model_stats = []
    for model in df['model_name'].unique():
        model_data = df[df['model_name'] == model]
        
        stats = {
            'model': model,
            'count': len(model_data),
            'avg_score': model_data['score'].mean() if 'score' in model_data.columns else 0
        }
        
        if 'input_token_count' in df.columns:
            model_input = model_data['input_token_count'].apply(safe_convert)
            stats['valid_input_tokens'] = (model_input > 0).sum()
            stats['input_token_rate'] = (stats['valid_input_tokens'] / len(model_data)) * 100
        
        if 'output_token_count' in df.columns:
            model_output = model_data['output_token_count'].apply(safe_convert)
            stats['valid_output_tokens'] = (model_output > 0).sum()
            stats['output_token_rate'] = (stats['valid_output_tokens'] / len(model_data)) * 100
        
        if 'latency' in df.columns:
            model_latency = model_data['latency'].apply(safe_convert)
            stats['normal_latency'] = (model_latency > 1).sum()
            stats['latency_rate'] = (stats['normal_latency'] / len(model_data)) * 100
        
        model_stats.append(stats)
        print(f"  {model[:40]:40} - Records: {stats['count']:5}, "
              f"Input tokens OK: {stats.get('input_token_rate', 0):5.1f}%, "
              f"Output tokens OK: {stats.get('output_token_rate', 0):5.1f}%, "
              f"Normal latency: {stats.get('latency_rate', 0):5.1f}%")
    
    # Check for infrastructure issues
    print("\nInfrastructure Analysis:")
    
    # Define what constitutes a successful evaluation
    if 'input_token_count' in df.columns and 'output_token_count' in df.columns and 'latency' in df.columns:
        input_ok = df['input_token_count'].apply(safe_convert) > 0
        output_ok = df['output_token_count'].apply(safe_convert) > 0
        latency_ok = df['latency'].apply(safe_convert) > 0.1  # At least 100ms
        
        successful = input_ok & output_ok & latency_ok
        results['successful_evaluations'] = successful.sum()
        results['success_rate'] = (successful.sum() / len(df)) * 100
        
        print(f"Successful evaluations: {successful.sum()} / {len(df)} ({results['success_rate']:.1f}%)")
        
        # Check for systematic failures
        failed = ~successful
        results['failed_evaluations'] = failed.sum()
        results['failure_rate'] = (failed.sum() / len(df)) * 100
        
        print(f"Failed evaluations: {failed.sum()} / {len(df)} ({results['failure_rate']:.1f}%)")
    
    # Save results (convert int64 to int for JSON serialization)
    output_file = Path("corrected_analysis_results.json")
    # Convert numpy int64 to regular int
    for key, value in results.items():
        if isinstance(value, (np.int64, np.int32)):
            results[key] = int(value)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Create corrected report
    create_corrected_report(df, results, model_stats)
    
    return df, results

def create_corrected_report(df, results, model_stats):
    """Create a corrected analysis report"""
    
    report = f"""# BFCL Benchmark Analysis - Corrected Results

**Date**: September 2, 2025
**Total Records Analyzed**: {results['total_records']:,}
**Models**: {results['model_count']}
**Test Categories**: {results['test_categories']}

## Executive Summary

Based on corrected column names and proper data type handling:

- **Success Rate**: {results.get('success_rate', 0):.1f}% of evaluations completed successfully
- **Input Token Coverage**: {results.get('input_token_percentage', 0):.1f}% of records have input tokens
- **Output Token Coverage**: {results.get('output_token_percentage', 0):.1f}% of records have output tokens
- **Normal Latency**: {results.get('normal_latency_percentage', 0):.1f}% of records have latency > 1s

## Key Findings

"""

    if results.get('success_rate', 0) < 10:
        report += """### ⚠️ Critical Infrastructure Issues Detected

The analysis reveals significant infrastructure problems:
- Very low success rate indicates systematic failures
- Most evaluations did not complete properly
- Immediate investigation and fixes required

"""
    elif results.get('success_rate', 0) < 50:
        report += """### ⚠️ Moderate Infrastructure Issues Detected

The analysis reveals some infrastructure problems:
- Success rate below 50% indicates frequent failures
- Many evaluations experiencing issues
- Investigation recommended

"""
    else:
        report += """### ✅ Infrastructure Generally Functional

The analysis shows:
- Majority of evaluations completed successfully
- Token tracking and latency measurements working
- Some optimization opportunities may exist

"""

    report += f"""## Model-Specific Results

| Model | Records | Input OK | Output OK | Latency OK |
|-------|---------|----------|-----------|------------|
"""

    for stats in model_stats[:10]:  # Top 10 models
        report += f"| {stats['model'][:30]} | {stats['count']} | {stats.get('input_token_rate', 0):.1f}% | {stats.get('output_token_rate', 0):.1f}% | {stats.get('latency_rate', 0):.1f}% |\n"

    report += "\n## Recommendations\n\n"
    
    if results.get('failure_rate', 0) > 50:
        report += """1. **Immediate Action Required**: Infrastructure experiencing widespread failures
2. **Root Cause Analysis**: Investigate token tracking and latency measurement systems
3. **Data Validation**: Verify data collection pipeline integrity
4. **Re-evaluation**: Consider re-running failed evaluations
"""
    else:
        report += """1. **Optimize Failed Cases**: Investigate and fix specific failure patterns
2. **Performance Monitoring**: Implement continuous monitoring
3. **Data Quality**: Enhance data validation and error handling
"""

    # Save report
    report_file = Path("CORRECTED_BFCL_ANALYSIS_REPORT.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report saved to {report_file}")

if __name__ == "__main__":
    df, results = analyze_bfcl_data()