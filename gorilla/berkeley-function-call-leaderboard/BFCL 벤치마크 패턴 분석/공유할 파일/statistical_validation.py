"""
Statistical validation module for BFCL benchmark analysis.
Provides rigorous statistical testing for performance inversions.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class StatisticalValidator:
    """Statistical validation for model performance comparisons."""
    
    def __init__(self, confidence_level: float = 0.95, bootstrap_iterations: int = 1000):
        self.confidence_level = confidence_level
        self.bootstrap_iterations = bootstrap_iterations
        self.alpha = 1 - confidence_level
        
    def bootstrap_confidence_interval(self, data: np.ndarray, statistic_func: callable) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for a statistic."""
        if len(data) < 2:
            return (np.nan, np.nan)
            
        bootstrap_stats = []
        for _ in range(self.bootstrap_iterations):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic_func(bootstrap_sample))
        
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return ci_lower, ci_upper
    
    def calculate_performance_delta_with_ci(self, strong_scores: List[float], weak_scores: List[float]) -> Dict[str, float]:
        """Calculate performance delta with confidence intervals."""
        if not strong_scores or not weak_scores:
            return {
                'delta': np.nan,
                'delta_weighted': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'p_value': np.nan,
                'n_strong': 0,
                'n_weak': 0,
                'is_significant': False
            }
        
        strong_mean = np.mean(strong_scores)
        weak_mean = np.mean(weak_scores)
        delta = weak_mean - strong_mean
        
        # Sample size weighted delta
        n_total = len(strong_scores) + len(weak_scores)
        delta_weighted = delta * np.sqrt(n_total)
        
        # Statistical test
        if len(strong_scores) > 1 and len(weak_scores) > 1:
            # Use Welch's t-test for unequal variances
            t_stat, p_value = stats.ttest_ind(weak_scores, strong_scores, equal_var=False)
        elif len(strong_scores) == 1 and len(weak_scores) == 1:
            # Use Cohen's d approximation
            pooled_std = np.sqrt((np.var(strong_scores) + np.var(weak_scores)) / 2)
            if pooled_std > 0:
                cohens_d = abs(delta) / pooled_std
                # Approximate p-value from effect size
                p_value = 2 * (1 - stats.norm.cdf(cohens_d))
            else:
                p_value = 1.0
        else:
            p_value = np.nan
        
        # Bootstrap CI for delta
        combined_data = np.array(weak_scores + strong_scores)
        def delta_func(data):
            mid_point = len(weak_scores)
            return np.mean(data[:mid_point]) - np.mean(data[mid_point:])
        
        ci_lower, ci_upper = self.bootstrap_confidence_interval(combined_data, delta_func)
        
        # Significance check
        is_significant = (
            abs(delta) > 0.1 and  # Practical significance threshold
            p_value < 0.05 and    # Statistical significance
            not np.isnan(p_value)
        )
        
        return {
            'delta': delta,
            'delta_weighted': delta_weighted,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'n_strong': len(strong_scores),
            'n_weak': len(weak_scores),
            'is_significant': is_significant
        }
    
    def chi_square_test_independence(self, success_counts: List[int], total_counts: List[int]) -> Dict[str, float]:
        """Perform chi-square test for independence of success rates."""
        if len(success_counts) != len(total_counts) or len(success_counts) < 2:
            return {'chi2_stat': np.nan, 'p_value': np.nan, 'effect_size': np.nan}
        
        # Create contingency table
        failure_counts = [total - success for total, success in zip(total_counts, success_counts)]
        contingency_table = np.array([success_counts, failure_counts])
        
        # Ensure all cells have at least 5 expected frequency
        if np.any(contingency_table < 5):
            # Use Fisher's exact test for small samples
            if len(success_counts) == 2:
                oddsratio, p_value = fisher_exact(contingency_table.T)
                chi2_stat = np.nan
                effect_size = np.log(oddsratio) if oddsratio > 0 else np.nan
            else:
                return {'chi2_stat': np.nan, 'p_value': np.nan, 'effect_size': np.nan}
        else:
            # Chi-square test
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Cramér's V as effect size
            n = np.sum(contingency_table)
            effect_size = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
        
        return {
            'chi2_stat': chi2_stat if not np.isnan(chi2_stat) else None,
            'p_value': p_value,
            'effect_size': effect_size
        }
    
    def validate_performance_inversion(self, 
                                     strong_model_data: pd.DataFrame,
                                     weak_model_data: pd.DataFrame,
                                     metric: str = 'accuracy') -> Dict[str, Union[float, bool]]:
        """Comprehensive validation of performance inversion between two models."""
        
        if strong_model_data.empty or weak_model_data.empty:
            return self._empty_validation_result()
        
        strong_scores = strong_model_data[metric].dropna().tolist()
        weak_scores = weak_model_data[metric].dropna().tolist()
        
        # Performance delta analysis
        delta_result = self.calculate_performance_delta_with_ci(strong_scores, weak_scores)
        
        # Success/failure counts for chi-square test
        if 'correct_count' in strong_model_data.columns and 'total_count' in strong_model_data.columns:
            strong_success = strong_model_data['correct_count'].sum()
            strong_total = strong_model_data['total_count'].sum()
            weak_success = weak_model_data['correct_count'].sum()
            weak_total = weak_model_data['total_count'].sum()
            
            chi2_result = self.chi_square_test_independence(
                [strong_success, weak_success],
                [strong_total, weak_total]
            )
        else:
            chi2_result = {'chi2_stat': np.nan, 'p_value': np.nan, 'effect_size': np.nan}
        
        # Combine results
        result = {**delta_result, **chi2_result}
        result['validation_type'] = 'performance_inversion'
        
        return result
    
    def _empty_validation_result(self) -> Dict[str, Union[float, bool]]:
        """Return empty validation result structure."""
        return {
            'delta': np.nan,
            'delta_weighted': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'p_value': np.nan,
            'n_strong': 0,
            'n_weak': 0,
            'is_significant': False,
            'chi2_stat': np.nan,
            'effect_size': np.nan,
            'validation_type': 'empty'
        }
    
    def batch_validate_inversions(self, df: pd.DataFrame, 
                                strong_models: List[str], 
                                weak_models: List[str],
                                test_categories: Optional[List[str]] = None) -> pd.DataFrame:
        """Batch validate performance inversions across multiple test categories."""
        
        if test_categories is None:
            test_categories = df['test_category'].unique()
        
        validation_results = []
        
        for test_category in test_categories:
            test_data = df[df['test_category'] == test_category]
            
            for strong_model in strong_models:
                strong_data = test_data[test_data['model_name'].str.contains('|'.join([strong_model]), na=False)]
                
                for weak_model in weak_models:
                    weak_data = test_data[test_data['model_name'].str.contains('|'.join([weak_model]), na=False)]
                    
                    if not strong_data.empty and not weak_data.empty:
                        validation = self.validate_performance_inversion(strong_data, weak_data)
                        validation.update({
                            'test_category': test_category,
                            'strong_model': strong_model,
                            'weak_model': weak_model,
                            'strong_mean_accuracy': strong_data['accuracy'].mean() if 'accuracy' in strong_data.columns else np.nan,
                            'weak_mean_accuracy': weak_data['accuracy'].mean() if 'accuracy' in weak_data.columns else np.nan
                        })
                        validation_results.append(validation)
        
        results_df = pd.DataFrame(validation_results)
        
        # Filter for significant inversions only
        if not results_df.empty:
            results_df = results_df[
                (results_df['is_significant'] == True) &
                (results_df['delta'] > 0.1)  # Weak outperforms strong by >10%
            ].sort_values('delta', ascending=False)
        
        return results_df
    
    def calculate_confidence_intervals_for_accuracy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate confidence intervals for accuracy scores using Wilson score interval."""
        results = []
        
        for _, row in df.iterrows():
            if pd.isna(row.get('correct_count')) or pd.isna(row.get('total_count')):
                ci_lower, ci_upper = np.nan, np.nan
            else:
                correct = int(row['correct_count'])
                total = int(row['total_count'])
                
                if total == 0:
                    ci_lower, ci_upper = 0, 0
                else:
                    # Wilson score interval
                    p = correct / total
                    z = stats.norm.ppf(1 - self.alpha/2)  # 97.5th percentile for 95% CI
                    
                    denominator = 1 + (z**2 / total)
                    center = (p + z**2/(2*total)) / denominator
                    margin = z * np.sqrt((p*(1-p) + z**2/(4*total)) / total) / denominator
                    
                    ci_lower = max(0, center - margin)
                    ci_upper = min(1, center + margin)
            
            results.append({
                **row.to_dict(),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'ci_width': ci_upper - ci_lower if not np.isnan(ci_lower) else np.nan
            })
        
        return pd.DataFrame(results)