#!/usr/bin/env python3
"""
Simplified runner for enhanced BFCL analysis to demonstrate capabilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
from statistical_validation import StatisticalValidator
from family_task_analysis import FamilyTaskAnalyzer

def run_demo_analysis():
    """Run a demonstration of the enhanced analysis capabilities."""
    
    print("=" * 60)
    print("ENHANCED BFCL ANALYSIS DEMONSTRATION")  
    print("=" * 60)
    
    # Load existing analysis results
    analysis_dir = Path(".")
    
    # Check if we have previous results
    if (analysis_dir / "bfcl_analysis_detailed.csv").exists():
        print("Loading existing analysis data...")
        df = pd.read_csv("bfcl_analysis_detailed.csv")
        print(f"Loaded {len(df)} records")
    else:
        print("Creating synthetic data for demonstration...")
        # Create synthetic data to demonstrate capabilities
        df = create_synthetic_data()
    
    # Initialize enhanced components
    validator = StatisticalValidator(confidence_level=0.95, bootstrap_iterations=500)
    family_analyzer = FamilyTaskAnalyzer()
    
    print("\n1. STATISTICAL VALIDATION DEMONSTRATION")
    print("-" * 40)
    
    # Demonstrate statistical validation
    strong_scores = [0.75, 0.80, 0.78, 0.82, 0.79]
    weak_scores = [0.85, 0.88, 0.90, 0.87, 0.89]
    
    validation_result = validator.calculate_performance_delta_with_ci(strong_scores, weak_scores)
    
    print(f"Performance Delta: {validation_result['delta']:.2%}")
    print(f"95% Confidence Interval: [{validation_result['ci_lower']:.2%}, {validation_result['ci_upper']:.2%}]")
    print(f"P-value: {validation_result['p_value']:.3f}")
    print(f"Statistically Significant: {validation_result['is_significant']}")
    print(f"Sample-weighted Delta: {validation_result['delta_weighted']:.3f}")
    
    print("\n2. FAMILY-TASK ANALYSIS DEMONSTRATION")  
    print("-" * 40)
    
    if 'model_name' in df.columns and 'test_category' in df.columns:
        # Create family-task matrix
        matrix_df = family_analyzer.create_family_task_matrix(df)
        print(f"Created family-task matrix: {len(matrix_df)} entries")
        
        if not matrix_df.empty:
            # Show top performing family-task combinations
            top_performances = matrix_df.nlargest(5, 'mean_accuracy')
            print("\\nTop 5 Family-Task Performances:")
            for _, row in top_performances.iterrows():
                print(f"  {row['model_family']} on {row['test_category']}: {row['mean_accuracy']:.1%}")
            
            # Identify systematic failures
            failures = family_analyzer.identify_systematic_family_failures(matrix_df)
            print(f"\\nIdentified systematic failures in {len(failures)} families")
    
    print("\n3. ERROR CLASSIFICATION DEMONSTRATION")
    print("-" * 40)
    
    if 'error' in df.columns or 'error_type' in df.columns:
        # Classify errors
        technical_patterns = [r"timeout", r"connection error", r"rate limit"]
        format_patterns = [r"JSON.*decode", r"decoder_success", r"format error"]
        
        if 'error' in df.columns:
            df['is_technical_error'] = df['error'].str.contains(
                '|'.join(technical_patterns), case=False, na=False
            )
            df['is_format_error'] = df['error'].str.contains(
                '|'.join(format_patterns), case=False, na=False
            )
            
            tech_count = df['is_technical_error'].sum()
            format_count = df['is_format_error'].sum() 
            
            print(f"Technical Errors: {tech_count:,} ({tech_count/len(df):.1%})")
            print(f"Format Errors: {format_count:,} ({format_count/len(df):.1%}")
            print(f"Valid Evaluations: {len(df) - tech_count - format_count:,}")
    
    print("\n4. CONFIDENCE INTERVAL CALCULATION")
    print("-" * 40)
    
    # Demonstrate CI calculation for accuracy scores
    if 'correct_count' in df.columns and 'total_count' in df.columns:
        sample_data = df[['correct_count', 'total_count']].dropna().head(5)
        sample_data['accuracy'] = sample_data['correct_count'] / sample_data['total_count']
        
        ci_data = validator.calculate_confidence_intervals_for_accuracy(sample_data)
        
        print("Sample Accuracy Confidence Intervals:")
        for _, row in ci_data.iterrows():
            if not pd.isna(row['ci_lower']):
                print(f"  Accuracy: {row['accuracy']:.1%} (95% CI: {row['ci_lower']:.1%} - {row['ci_upper']:.1%})")
    
    print("\n5. PERFORMANCE SUMMARY") 
    print("-" * 40)
    
    if 'accuracy' in df.columns:
        accuracy_data = df[df['accuracy'].notna()]
        if not accuracy_data.empty:
            print(f"Total Evaluations Analyzed: {len(df):,}")
            print(f"Mean Accuracy: {accuracy_data['accuracy'].mean():.1%}")
            print(f"Accuracy Std Dev: {accuracy_data['accuracy'].std():.1%}")
            print(f"Median Accuracy: {accuracy_data['accuracy'].median():.1%}")
            
            if 'model_name' in df.columns:
                model_count = df['model_name'].nunique()
                category_count = df.get('test_category', pd.Series()).nunique()
                print(f"Models Analyzed: {model_count}")
                print(f"Test Categories: {category_count}")
    
    # Save demonstration results
    results = {
        'statistical_validation_demo': validation_result,
        'analysis_summary': {
            'total_records': len(df),
            'mean_accuracy': df.get('accuracy', pd.Series()).mean(),
            'unique_models': df.get('model_name', pd.Series()).nunique(),
            'unique_categories': df.get('test_category', pd.Series()).nunique()
        },
        'demonstration_completed': True
    }
    
    with open("enhanced_analysis_demo_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("ENHANCED ANALYSIS DEMONSTRATION COMPLETE")
    print("Results saved to: enhanced_analysis_demo_results.json")
    print("=" * 60)

def create_synthetic_data():
    """Create synthetic data for demonstration purposes."""
    np.random.seed(42)
    
    models = [
        "anthropic_claude-4-sonnet-thinking-off",
        "openai_gpt-4o-20240806", 
        "deepseek-ai_DeepSeek-V3-0324",
        "_data_jibf_.cache_huggingface_hub_models--Qwen--Qwen3-32B",
        "openai_gpt-4.1"
    ]
    
    categories = [
        "irrelevance", "simple", "multiple", "parallel", 
        "live_simple", "multi_turn_base", "java", "javascript"
    ]
    
    data = []
    
    for model in models:
        for category in categories:
            # Summary data
            accuracy = np.random.beta(5, 2)  # Skewed towards higher accuracy
            total_count = np.random.randint(50, 200)
            correct_count = int(accuracy * total_count)
            
            data.append({
                'model_name': model,
                'test_category': category,
                'accuracy': accuracy,
                'correct_count': correct_count,
                'total_count': total_count,
                'data_type': 'summary'
            })
            
            # Add some detailed results
            for i in range(np.random.randint(3, 8)):
                is_valid = np.random.random() > 0.2
                error_msg = ""
                error_type = ""
                
                if not is_valid:
                    if np.random.random() < 0.3:
                        error_msg = "decoder_success error"
                        error_type = "irrelevance_error:decoder_success"
                    elif np.random.random() < 0.5:
                        error_msg = "JSON decode error"
                        error_type = "format_error"
                    else:
                        error_msg = "timeout occurred"
                        error_type = "technical_error"
                
                data.append({
                    'model_name': model,
                    'test_category': category,
                    'test_id': f"{category}_{i}",
                    'valid': is_valid,
                    'error': error_msg,
                    'error_type': error_type,
                    'data_type': 'detailed'
                })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    run_demo_analysis()