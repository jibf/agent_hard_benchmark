import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import re
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Define paths
base_path = Path(r"E:\Users\김현준\Downloads\agent_hard_benchmark_2\gorilla")
project_path = base_path / "berkeley-function-call-leaderboard"
result_path = project_path / "result"
score_path = project_path / "score"

def extract_model_name(file_path: Path) -> str:
    """Extract model name from file path or directory name"""
    parts = file_path.parts
    for part in parts:
        if part not in ['score', 'result', 'berkeley-function-call-leaderboard', 'gorilla']:
            return part
    return file_path.stem

def load_score_files() -> pd.DataFrame:
    """Load all score files and create a comprehensive dataframe"""
    all_scores = []
    
    # Find all JSON score files
    score_files = glob.glob(str(score_path / "**/*.json"), recursive=True)
    
    for score_file in score_files:
        try:
            file_path = Path(score_file)
            model_name = file_path.parent.name
            test_category = file_path.stem.replace('BFCL_v3_', '').replace('_score', '')
            
            with open(score_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Try to parse as single JSON object
            try:
                data = json.loads(content)
                # Check if it's a summary file (contains accuracy)
                if isinstance(data, dict) and 'accuracy' in data:
                    all_scores.append({
                        'model_name': model_name,
                        'test_category': test_category,
                        'accuracy': data.get('accuracy', 0),
                        'correct_count': data.get('correct_count', 0),
                        'total_count': data.get('total_count', 0),
                        'file_path': str(file_path)
                    })
                # Check if it's a detailed results file (list of test results)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'id' in item:
                            all_scores.append({
                                'model_name': item.get('model_name', model_name),
                                'test_category': item.get('test_category', test_category),
                                'test_id': item.get('id'),
                                'valid': item.get('valid', False),
                                'error': str(item.get('error', [])),
                                'error_type': item.get('error_type', ''),
                                'file_path': str(file_path)
                            })
            except json.JSONDecodeError:
                # Try to parse as multiple JSON objects (one per line)
                lines = content.strip().split('\n')
                for line in lines:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            # Check if it's a summary line (contains accuracy)
                            if isinstance(data, dict) and 'accuracy' in data:
                                all_scores.append({
                                    'model_name': model_name,
                                    'test_category': test_category,
                                    'accuracy': data.get('accuracy', 0),
                                    'correct_count': data.get('correct_count', 0),
                                    'total_count': data.get('total_count', 0),
                                    'file_path': str(file_path)
                                })
                            # Check if it's a detailed result line
                            elif isinstance(data, dict) and 'id' in data:
                                all_scores.append({
                                    'model_name': data.get('model_name', model_name),
                                    'test_category': data.get('test_category', test_category),
                                    'test_id': data.get('id'),
                                    'valid': data.get('valid', False),
                                    'error': str(data.get('error', [])),
                                    'error_type': data.get('error_type', ''),
                                    'file_path': str(file_path)
                                })
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            print(f"Error loading {score_file}: {e}")
    
    return pd.DataFrame(all_scores)

def identify_technical_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Identify technical/infrastructure errors that shouldn't count as model failures"""
    
    if 'error' not in df.columns and 'error_type' not in df.columns:
        df['is_technical_error'] = False
        df['is_format_error'] = False
        df['is_suspicious'] = False
        return df
    
    technical_error_patterns = [
        r"max_tokens.*[Ff]ield required",
        r"timeout|timed out",
        r"connection error|connection refused",
        r"rate limit|RateLimitError",
        r"API error|APIError", 
        r"infrastructure error",
        r"500 Internal Server Error",
        r"502 Bad Gateway",
        r"503 Service Unavailable",
        r"error_code.*400"
    ]
    
    format_error_patterns = [
        r"JSON.*decode error|JSONDecodeError",
        r"parsing failed|parse error",
        r"invalid format|format error",
        r"unexpected token",
        r"malformed response",
        r"expecting property name",
        r"decoder_success"  # This is actually a format issue - model responded when it shouldn't
    ]
    
    # Handle error column
    if 'error' in df.columns:
        df['is_technical_error'] = df['error'].apply(
            lambda x: any(re.search(p, str(x), re.IGNORECASE) for p in technical_error_patterns) if pd.notna(x) else False
        )
        
        df['is_format_error'] = df['error'].apply(
            lambda x: any(re.search(p, str(x), re.IGNORECASE) for p in format_error_patterns) if pd.notna(x) else False
        )
    else:
        df['is_technical_error'] = False
        df['is_format_error'] = False
    
    # Handle error_type column
    if 'error_type' in df.columns:
        df['is_format_error'] = df['is_format_error'] | df['error_type'].str.contains('decoder_success|format', na=False, case=False)
    
    # Flag suspicious cases
    df['is_suspicious'] = df['is_format_error'] | df['is_technical_error']
    
    return df

def analyze_irrelevance_tests(df: pd.DataFrame) -> Dict[str, Any]:
    """Specifically analyze irrelevance test performance"""
    irrelevance_df = df[df['test_category'].str.contains('irrelevance', na=False)]
    
    if irrelevance_df.empty:
        return {}
    
    # Group by model
    model_performance = {}
    
    for model in irrelevance_df['model_name'].unique():
        model_data = irrelevance_df[irrelevance_df['model_name'] == model]
        
        # Calculate metrics
        if 'accuracy' in model_data.columns:
            accuracy_data = model_data[model_data['accuracy'].notna()]
            if not accuracy_data.empty:
                avg_accuracy = accuracy_data['accuracy'].mean()
            else:
                avg_accuracy = None
        else:
            avg_accuracy = None
        
        # Count error types
        error_analysis = {}
        if 'error_type' in model_data.columns:
            error_counts = model_data['error_type'].value_counts()
            decoder_success_errors = error_counts.get('irrelevance_error:decoder_success', 0)
            total_errors = len(model_data[model_data['valid'] == False]) if 'valid' in model_data.columns else 0
            
            error_analysis = {
                'decoder_success_errors': int(decoder_success_errors),
                'total_errors': int(total_errors),
                'decoder_success_rate': decoder_success_errors / total_errors if total_errors > 0 else 0
            }
        
        model_performance[model] = {
            'accuracy': avg_accuracy,
            'total_tests': len(model_data),
            'error_analysis': error_analysis
        }
    
    return model_performance

def analyze_performance_inversions(df: pd.DataFrame) -> pd.DataFrame:
    """Find tasks where weak models outperform strong models"""
    
    # Define model tiers based on known capabilities
    top_tier_models = [
        'gpt-4', 'gpt-4-turbo', 'gpt-4.1', 'gpt-4o',
        'claude-3-opus', 'claude-4-sonnet', 'claude-opus',
        'gemini-ultra', 'gemini-1.5-pro',
        'o3-high', 'o4-mini-high'
    ]
    
    mid_tier_models = [
        'gpt-3.5-turbo', 'claude-3-sonnet', 'claude-2',
        'gemini-pro', 'llama-3-70b', 'deepseek-v3', 'deepseek-r1'
    ]
    
    lower_tier_models = [
        'llama-3-8b', 'mistral-7b', 'gemma-7b',
        'phi-2', 'vicuna-13b', 'qwen3-8b', 'qwen3-32b'
    ]
    
    # If we have accuracy data, use that
    if 'accuracy' in df.columns:
        summary_df = df[df['accuracy'].notna()].copy()
        
        inversions = []
        
        for test_category in summary_df['test_category'].unique():
            category_df = summary_df[summary_df['test_category'] == test_category]
            
            # Find top tier performance
            top_tier_perf = []
            for model in category_df['model_name']:
                if any(tier in model.lower() for tier in top_tier_models):
                    acc = category_df[category_df['model_name'] == model]['accuracy'].values
                    if len(acc) > 0:
                        top_tier_perf.append((model, acc[0]))
            
            # Find lower tier performance
            lower_tier_perf = []
            for model in category_df['model_name']:
                if any(tier in model.lower() for tier in lower_tier_models):
                    acc = category_df[category_df['model_name'] == model]['accuracy'].values
                    if len(acc) > 0:
                        lower_tier_perf.append((model, acc[0]))
            
            # Check for inversions
            if top_tier_perf and lower_tier_perf:
                min_top = min(top_tier_perf, key=lambda x: x[1])
                max_lower = max(lower_tier_perf, key=lambda x: x[1])
                
                if max_lower[1] > min_top[1] + 0.1:  # Significant inversion
                    inversions.append({
                        'test_category': test_category,
                        'weakest_top_tier_model': min_top[0],
                        'weakest_top_tier_score': min_top[1],
                        'strongest_lower_tier_model': max_lower[0],
                        'strongest_lower_tier_score': max_lower[1],
                        'inversion_delta': max_lower[1] - min_top[1]
                    })
        
        return pd.DataFrame(inversions)
    
    return pd.DataFrame()

def analyze_model_families(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze patterns by model family"""
    
    # Define model families
    model_families = {
        'OpenAI': ['gpt-4', 'gpt-3.5', 'o3', 'o4'],
        'Anthropic': ['claude'],
        'Google': ['gemini', 'gemma'],
        'Meta': ['llama'],
        'Mistral': ['mistral', 'mixtral'],
        'Deepseek': ['deepseek'],
        'Qwen': ['qwen']
    }
    
    # Add family column
    def get_family(model_name):
        model_lower = str(model_name).lower()
        for family, patterns in model_families.items():
            if any(pattern in model_lower for pattern in patterns):
                return family
        return 'Other'
    
    df['model_family'] = df['model_name'].apply(get_family)
    
    # Analyze by family
    family_analysis = {}
    
    for family in df['model_family'].unique():
        family_df = df[df['model_family'] == family]
        
        analysis = {
            'total_evaluations': len(family_df),
            'unique_models': family_df['model_name'].nunique(),
            'models': list(family_df['model_name'].unique())
        }
        
        if 'accuracy' in family_df.columns:
            acc_df = family_df[family_df['accuracy'].notna()]
            if not acc_df.empty:
                analysis['avg_accuracy'] = acc_df['accuracy'].mean()
                analysis['min_accuracy'] = acc_df['accuracy'].min()
                analysis['max_accuracy'] = acc_df['accuracy'].max()
        
        if 'is_technical_error' in family_df.columns:
            analysis['technical_error_rate'] = family_df['is_technical_error'].mean()
        
        if 'is_format_error' in family_df.columns:
            analysis['format_error_rate'] = family_df['is_format_error'].mean()
        
        family_analysis[family] = analysis
    
    return family_analysis

def generate_report(df: pd.DataFrame):
    """Generate comprehensive analysis report"""
    
    print("=" * 80)
    print("BFCL BENCHMARK PATTERN ANALYSIS REPORT")
    print("=" * 80)
    print()
    
    # Basic statistics
    print("Dataset Overview:")
    print(f"Total records: {len(df)}")
    print(f"Unique models: {df['model_name'].nunique()}")
    print(f"Unique test categories: {df['test_category'].nunique()}")
    print()
    
    # Identify errors
    df = identify_technical_errors(df)
    
    if 'is_technical_error' in df.columns:
        print("Error Analysis:")
        print(f"Technical errors: {df['is_technical_error'].sum()}")
        print(f"Format errors: {df['is_format_error'].sum()}")
        print(f"Suspicious patterns: {df['is_suspicious'].sum()}")
        print()
    
    # Irrelevance test analysis
    print("=" * 80)
    print("IRRELEVANCE TEST ANALYSIS (Key Issue for Claude-4-Sonnet)")
    print("=" * 80)
    irrelevance_analysis = analyze_irrelevance_tests(df)
    
    for model, stats in irrelevance_analysis.items():
        if 'claude' in model.lower() or 'qwen' in model.lower():
            print(f"\nModel: {model}")
            if stats['accuracy']:
                print(f"  Accuracy: {stats['accuracy']:.2%}")
            print(f"  Total tests: {stats['total_tests']}")
            if stats['error_analysis']:
                print(f"  Decoder success errors: {stats['error_analysis']['decoder_success_errors']}")
                print(f"  Decoder success rate: {stats['error_analysis']['decoder_success_rate']:.2%}")
    
    # Performance inversions
    print("\n" + "=" * 80)
    print("PERFORMANCE INVERSIONS")
    print("=" * 80)
    inversions_df = analyze_performance_inversions(df)
    
    if not inversions_df.empty:
        inversions_df = inversions_df.sort_values('inversion_delta', ascending=False)
        print("\nTop Performance Inversions (weak models beating strong models):")
        for _, row in inversions_df.head(10).iterrows():
            print(f"\n{row['test_category']}:")
            print(f"  {row['weakest_top_tier_model']}: {row['weakest_top_tier_score']:.2%}")
            print(f"  beaten by {row['strongest_lower_tier_model']}: {row['strongest_lower_tier_score']:.2%}")
            print(f"  Delta: {row['inversion_delta']:.2%}")
    
    # Model family analysis
    print("\n" + "=" * 80)
    print("MODEL FAMILY ANALYSIS")
    print("=" * 80)
    family_analysis = analyze_model_families(df)
    
    for family, stats in sorted(family_analysis.items(), key=lambda x: x[1].get('avg_accuracy', 0), reverse=True):
        print(f"\n{family}:")
        print(f"  Models: {stats['unique_models']}")
        if 'avg_accuracy' in stats:
            print(f"  Average accuracy: {stats['avg_accuracy']:.2%}")
            print(f"  Range: {stats['min_accuracy']:.2%} - {stats['max_accuracy']:.2%}")
        if 'technical_error_rate' in stats:
            print(f"  Technical error rate: {stats['technical_error_rate']:.2%}")
        if 'format_error_rate' in stats:
            print(f"  Format error rate: {stats['format_error_rate']:.2%}")
    
    # Save detailed results
    print("\n" + "=" * 80)
    print("SAVING DETAILED RESULTS")
    print("=" * 80)
    
    # Save dataframe
    df.to_csv(project_path / 'bfcl_analysis_detailed.csv', index=False)
    print(f"Saved detailed analysis to bfcl_analysis_detailed.csv")
    
    # Save inversions
    if not inversions_df.empty:
        inversions_df.to_csv(project_path / 'performance_inversions.csv', index=False)
        print(f"Saved {len(inversions_df)} performance inversions to performance_inversions.csv")
    
    # Save family analysis
    with open(project_path / 'model_family_patterns.json', 'w') as f:
        json.dump(family_analysis, f, indent=2, default=str)
    print(f"Saved model family analysis to model_family_patterns.json")
    
    # Save irrelevance analysis
    with open(project_path / 'irrelevance_analysis.json', 'w') as f:
        json.dump(irrelevance_analysis, f, indent=2, default=str)
    print(f"Saved irrelevance test analysis to irrelevance_analysis.json")

def main():
    print("Starting BFCL Benchmark Pattern Analysis...")
    print()
    
    # Load all score files
    df = load_score_files()
    
    if df.empty:
        print("No data loaded. Please check the file paths.")
        return
    
    print(f"Loaded {len(df)} evaluation records")
    print()
    
    # Generate comprehensive report
    generate_report(df)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nKey findings:")
    print("1. Check irrelevance_analysis.json for Claude-4-Sonnet issues")
    print("2. Check performance_inversions.csv for unfair evaluations")
    print("3. Check model_family_patterns.json for systematic biases")

if __name__ == "__main__":
    main()