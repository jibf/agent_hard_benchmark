"""
Family-Task cross-analysis module for BFCL benchmark.
Analyzes performance patterns across model families and test categories.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

class FamilyTaskAnalyzer:
    """Analyze performance patterns across model families and test categories."""
    
    def __init__(self):
        self.family_patterns = {
            'OpenAI': ['gpt-4', 'gpt-3.5', 'o3', 'o4'],
            'Anthropic': ['claude'],
            'Google': ['gemini', 'gemma'],
            'Meta': ['llama'],
            'Mistral': ['mistral', 'mixtral'],
            'Deepseek': ['deepseek'],
            'Qwen': ['qwen'],
            'Together': ['together'],
            'Moonshot': ['moonshot', 'kimi']
        }
    
    def assign_model_family(self, model_name: str) -> str:
        """Assign model to family based on name patterns."""
        model_lower = str(model_name).lower()
        
        for family, patterns in self.family_patterns.items():
            if any(pattern in model_lower for pattern in patterns):
                return family
        
        return 'Other'
    
    def create_family_task_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create family x task performance matrix."""
        # Add family column
        df = df.copy()
        df['model_family'] = df['model_name'].apply(self.assign_model_family)
        
        # Aggregate by family and test category
        matrix_data = []
        
        for family in df['model_family'].unique():
            family_data = df[df['model_family'] == family]
            
            for test_category in df['test_category'].unique():
                test_data = family_data[family_data['test_category'] == test_category]
                
                if not test_data.empty and 'accuracy' in test_data.columns:
                    accuracy_scores = test_data['accuracy'].dropna()
                    
                    if len(accuracy_scores) > 0:
                        matrix_data.append({
                            'model_family': family,
                            'test_category': test_category,
                            'mean_accuracy': accuracy_scores.mean(),
                            'std_accuracy': accuracy_scores.std(),
                            'min_accuracy': accuracy_scores.min(),
                            'max_accuracy': accuracy_scores.max(),
                            'n_models': len(accuracy_scores),
                            'total_tests': len(test_data),
                            'format_error_rate': test_data.get('is_format_error', pd.Series([False])).mean(),
                            'technical_error_rate': test_data.get('is_technical_error', pd.Series([False])).mean()
                        })
        
        return pd.DataFrame(matrix_data)
    
    def identify_family_strengths_weaknesses(self, matrix_df: pd.DataFrame, 
                                           top_k: int = 5) -> Dict[str, Dict[str, List[Dict]]]:
        """Identify top strengths and weaknesses for each family."""
        results = {}
        
        for family in matrix_df['model_family'].unique():
            family_data = matrix_df[matrix_df['model_family'] == family]
            
            # Sort by accuracy for strengths and weaknesses
            strengths = family_data.nlargest(top_k, 'mean_accuracy')
            weaknesses = family_data.nsmallest(top_k, 'mean_accuracy')
            
            # Convert to dict format
            strengths_list = []
            for _, row in strengths.iterrows():
                strengths_list.append({
                    'test_category': row['test_category'],
                    'mean_accuracy': row['mean_accuracy'],
                    'std_accuracy': row['std_accuracy'],
                    'n_models': row['n_models'],
                    'rank': len(strengths_list) + 1
                })
            
            weaknesses_list = []
            for _, row in weaknesses.iterrows():
                weaknesses_list.append({
                    'test_category': row['test_category'],
                    'mean_accuracy': row['mean_accuracy'],
                    'std_accuracy': row['std_accuracy'],
                    'n_models': row['n_models'],
                    'rank': len(weaknesses_list) + 1
                })
            
            results[family] = {
                'strengths': strengths_list,
                'weaknesses': weaknesses_list,
                'overall_mean': family_data['mean_accuracy'].mean(),
                'overall_std': family_data['mean_accuracy'].std(),
                'total_test_categories': len(family_data)
            }
        
        return results
    
    def analyze_task_difficulty_ranking(self, matrix_df: pd.DataFrame) -> pd.DataFrame:
        """Rank test categories by overall difficulty (lower accuracy = harder)."""
        task_stats = []
        
        for test_category in matrix_df['test_category'].unique():
            test_data = matrix_df[matrix_df['test_category'] == test_category]
            
            if not test_data.empty:
                # Weighted average by number of models
                weighted_accuracy = np.average(test_data['mean_accuracy'], 
                                             weights=test_data['n_models'])
                
                task_stats.append({
                    'test_category': test_category,
                    'overall_mean_accuracy': weighted_accuracy,
                    'std_across_families': test_data['mean_accuracy'].std(),
                    'n_families': len(test_data),
                    'min_family_accuracy': test_data['mean_accuracy'].min(),
                    'max_family_accuracy': test_data['mean_accuracy'].max(),
                    'range_across_families': test_data['mean_accuracy'].max() - test_data['mean_accuracy'].min(),
                    'total_models': test_data['n_models'].sum()
                })
        
        task_difficulty_df = pd.DataFrame(task_stats)
        task_difficulty_df['difficulty_rank'] = task_difficulty_df['overall_mean_accuracy'].rank(method='min')
        task_difficulty_df = task_difficulty_df.sort_values('overall_mean_accuracy')
        
        return task_difficulty_df
    
    def identify_systematic_family_failures(self, matrix_df: pd.DataFrame, 
                                          accuracy_threshold: float = 0.3) -> Dict[str, List[Dict]]:
        """Identify test categories where specific families systematically fail."""
        systematic_failures = defaultdict(list)
        
        for family in matrix_df['model_family'].unique():
            family_data = matrix_df[matrix_df['model_family'] == family]
            
            # Find low-performing test categories
            low_performance = family_data[family_data['mean_accuracy'] < accuracy_threshold]
            
            for _, row in low_performance.iterrows():
                # Compare with other families on same test
                other_families = matrix_df[
                    (matrix_df['test_category'] == row['test_category']) &
                    (matrix_df['model_family'] != family)
                ]
                
                if not other_families.empty:
                    other_mean = other_families['mean_accuracy'].mean()
                    gap = other_mean - row['mean_accuracy']
                    
                    if gap > 0.2:  # Significant gap
                        systematic_failures[family].append({
                            'test_category': row['test_category'],
                            'family_accuracy': row['mean_accuracy'],
                            'other_families_mean': other_mean,
                            'performance_gap': gap,
                            'n_models': row['n_models'],
                            'format_error_rate': row['format_error_rate']
                        })
        
        # Sort by performance gap
        for family in systematic_failures:
            systematic_failures[family] = sorted(
                systematic_failures[family],
                key=lambda x: x['performance_gap'],
                reverse=True
            )
        
        return dict(systematic_failures)
    
    def generate_family_task_heatmap_data(self, matrix_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Generate data for family-task heatmap visualization."""
        # Pivot table for heatmap
        heatmap_data = matrix_df.pivot(
            index='test_category',
            columns='model_family',
            values='mean_accuracy'
        ).fillna(np.nan)
        
        # Create annotation matrix with additional info
        annotation_data = {}
        for family in heatmap_data.columns:
            family_col = {}
            for test_category in heatmap_data.index:
                row = matrix_df[
                    (matrix_df['model_family'] == family) & 
                    (matrix_df['test_category'] == test_category)
                ]
                
                if not row.empty:
                    row = row.iloc[0]
                    accuracy = row['mean_accuracy']
                    n_models = row['n_models']
                    std_acc = row['std_accuracy']
                    
                    # Create annotation text
                    if pd.notna(accuracy):
                        if pd.notna(std_acc) and std_acc > 0:
                            family_col[test_category] = f"{accuracy:.1%}\n±{std_acc:.1%}\n(n={n_models})"
                        else:
                            family_col[test_category] = f"{accuracy:.1%}\n(n={n_models})"
                    else:
                        family_col[test_category] = "N/A"
                else:
                    family_col[test_category] = "N/A"
            
            annotation_data[family] = family_col
        
        # Convert to DataFrame for consistent indexing
        annotation_df = pd.DataFrame(annotation_data, index=heatmap_data.index)
        
        return heatmap_data, annotation_df
    
    def calculate_family_consistency_scores(self, matrix_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate consistency scores for each family across test categories."""
        consistency_scores = []
        
        for family in matrix_df['model_family'].unique():
            family_data = matrix_df[matrix_df['model_family'] == family]
            
            if len(family_data) > 1:
                # Coefficient of variation (std/mean) as consistency metric
                cv = family_data['mean_accuracy'].std() / family_data['mean_accuracy'].mean()
                
                # Range-based consistency
                accuracy_range = family_data['mean_accuracy'].max() - family_data['mean_accuracy'].min()
                
                # Number of catastrophic failures (accuracy < 10%)
                catastrophic_failures = (family_data['mean_accuracy'] < 0.1).sum()
                
                consistency_scores.append({
                    'model_family': family,
                    'coefficient_of_variation': cv,
                    'accuracy_range': accuracy_range,
                    'mean_accuracy': family_data['mean_accuracy'].mean(),
                    'std_accuracy': family_data['mean_accuracy'].std(),
                    'catastrophic_failures': catastrophic_failures,
                    'total_test_categories': len(family_data),
                    'consistency_score': 1 / (1 + cv) if cv > 0 else 1.0,  # Higher is more consistent
                })
        
        consistency_df = pd.DataFrame(consistency_scores)
        consistency_df = consistency_df.sort_values('consistency_score', ascending=False)
        
        return consistency_df
    
    def generate_comprehensive_family_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive family analysis report."""
        # Create family-task matrix
        matrix_df = self.create_family_task_matrix(df)
        
        # Run all analyses
        strengths_weaknesses = self.identify_family_strengths_weaknesses(matrix_df)
        task_difficulty = self.analyze_task_difficulty_ranking(matrix_df)
        systematic_failures = self.identify_systematic_family_failures(matrix_df)
        consistency_scores = self.calculate_family_consistency_scores(matrix_df)
        heatmap_data, annotation_data = self.generate_family_task_heatmap_data(matrix_df)
        
        return {
            'family_task_matrix': matrix_df,
            'strengths_weaknesses': strengths_weaknesses,
            'task_difficulty_ranking': task_difficulty,
            'systematic_failures': systematic_failures,
            'consistency_scores': consistency_scores,
            'heatmap_data': heatmap_data,
            'heatmap_annotations': annotation_data,
            'summary': {
                'total_families': len(matrix_df['model_family'].unique()),
                'total_test_categories': len(matrix_df['test_category'].unique()),
                'most_difficult_task': task_difficulty.iloc[0]['test_category'] if not task_difficulty.empty else 'N/A',
                'easiest_task': task_difficulty.iloc[-1]['test_category'] if not task_difficulty.empty else 'N/A',
                'most_consistent_family': consistency_scores.iloc[0]['model_family'] if not consistency_scores.empty else 'N/A',
                'least_consistent_family': consistency_scores.iloc[-1]['model_family'] if not consistency_scores.empty else 'N/A'
            }
        }