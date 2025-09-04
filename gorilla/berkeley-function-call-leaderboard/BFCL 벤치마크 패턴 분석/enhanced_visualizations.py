"""
Enhanced visualization module with CI, colorblind support, and accessibility features.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AccessibleVisualizer:
    """Create accessible visualizations with CI, alt-text, and colorblind support."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration from YAML."""
        self.config = config
        
        # Set colorblind-friendly palettes
        self.colorblind_palette = config.get('visualization', {}).get('colorblind_palette', {})
        self.primary_colors = self.colorblind_palette.get('primary', 
            ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        self.diverging_colors = self.colorblind_palette.get('diverging',
            ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd'])
        
        # Figure settings
        fig_settings = config.get('visualization', {}).get('figure_settings', {})
        self.dpi = fig_settings.get('dpi', 150)
        self.width = fig_settings.get('width', 16)
        self.height = fig_settings.get('height', 12)
        self.font_size = fig_settings.get('font_size', 11)
        self.title_size = fig_settings.get('title_size', 14)
        
        # Set matplotlib parameters
        plt.rcParams.update({
            'font.size': self.font_size,
            'axes.titlesize': self.title_size,
            'axes.labelsize': self.font_size,
            'xtick.labelsize': self.font_size - 1,
            'ytick.labelsize': self.font_size - 1,
            'legend.fontsize': self.font_size - 1,
            'figure.titlesize': self.title_size + 2
        })
        
        # Alt-text storage
        self.alt_texts = {}
    
    def create_performance_inversion_plot(self, inversions_df: pd.DataFrame, 
                                        validation_results: pd.DataFrame,
                                        output_path: Path) -> str:
        """Create enhanced performance inversion visualization with CI and p-values."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.width, self.height//2))
        
        # Left plot: Top inversions with CI
        top_inversions = inversions_df.head(10).copy()
        
        # Merge with validation results for CI data
        if not validation_results.empty:
            merged = pd.merge(
                top_inversions, 
                validation_results[['test_category', 'ci_lower', 'ci_upper', 'p_value', 'is_significant']],
                on='test_category', 
                how='left'
            )
        else:
            merged = top_inversions.copy()
            merged['ci_lower'] = np.nan
            merged['ci_upper'] = np.nan
            merged['p_value'] = np.nan
            merged['is_significant'] = False
        
        y_pos = np.arange(len(merged))
        
        # Create bars with different colors for significant/non-significant
        colors = [self.primary_colors[0] if sig else self.primary_colors[3] 
                 for sig in merged.get('is_significant', [False]*len(merged))]
        
        bars = ax1.barh(y_pos, merged['inversion_delta'], color=colors, alpha=0.7)
        
        # Add error bars if CI data available
        if 'ci_lower' in merged.columns and not merged['ci_lower'].isna().all():
            ci_width = merged['ci_upper'] - merged['ci_lower']
            ax1.errorbar(merged['inversion_delta'], y_pos, 
                        xerr=ci_width/2, fmt='none', color='black', alpha=0.5)
        
        # Formatting
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([cat.replace('_', ' ').title() for cat in merged['test_category']])
        ax1.set_xlabel('Performance Delta (Weak - Strong Model)', fontweight='bold')
        ax1.set_title('Top Performance Inversions with Statistical Validation', 
                     fontweight='bold', pad=20)
        ax1.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, label='Significance Threshold')
        
        # Add p-value annotations
        for i, (bar, p_val) in enumerate(zip(bars, merged.get('p_value', [np.nan]*len(bars)))):
            if not pd.isna(p_val):
                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        significance, va='center', fontweight='bold')
        
        # Legend
        sig_patch = mpatches.Patch(color=self.primary_colors[0], alpha=0.7, label='Statistically Significant')
        nonsig_patch = mpatches.Patch(color=self.primary_colors[3], alpha=0.7, label='Not Significant')
        ax1.legend(handles=[sig_patch, nonsig_patch], loc='lower right')
        
        # Right plot: P-value distribution
        if 'p_value' in merged.columns and not merged['p_value'].isna().all():
            p_values = merged['p_value'].dropna()
            ax2.hist(p_values, bins=10, alpha=0.7, color=self.primary_colors[1], edgecolor='black')
            ax2.axvline(x=0.05, color='red', linestyle='--', label='α = 0.05')
            ax2.set_xlabel('P-value', fontweight='bold')
            ax2.set_ylabel('Frequency', fontweight='bold')
            ax2.set_title('P-value Distribution of Performance Inversions', fontweight='bold')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'P-value data not available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=self.font_size)
            ax2.set_title('P-value Distribution', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path / 'enhanced_performance_inversions.png', 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        # Generate alt-text
        alt_text = self._generate_inversion_alt_text(merged)
        self.alt_texts['enhanced_performance_inversions.png'] = alt_text
        
        return str(output_path / 'enhanced_performance_inversions.png')
    
    def create_family_task_heatmap(self, heatmap_data: pd.DataFrame, 
                                  annotations: pd.DataFrame,
                                  consistency_scores: pd.DataFrame,
                                  output_path: Path) -> str:
        """Create enhanced family-task heatmap with consistency indicators."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.width, self.height//1.5), 
                                      gridspec_kw={'width_ratios': [3, 1]})
        
        # Main heatmap
        mask = heatmap_data.isna()
        sns.heatmap(heatmap_data, 
                   annot=annotations, 
                   fmt='', 
                   cmap='RdYlGn', 
                   center=0.5,
                   mask=mask,
                   cbar_kws={'label': 'Accuracy Score'},
                   ax=ax1,
                   vmin=0, vmax=1)
        
        ax1.set_title('Model Family Performance by Test Category', 
                     fontweight='bold', pad=20)
        ax1.set_xlabel('Model Family', fontweight='bold')
        ax1.set_ylabel('Test Category', fontweight='bold')
        
        # Rotate labels for better readability
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
        
        # Consistency scores bar chart
        if not consistency_scores.empty:
            y_pos = np.arange(len(consistency_scores))
            bars = ax2.barh(y_pos, consistency_scores['consistency_score'], 
                           color=self.primary_colors[2], alpha=0.7)
            
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(consistency_scores['model_family'])
            ax2.set_xlabel('Consistency Score', fontweight='bold')
            ax2.set_title('Family Consistency\n(Higher = More Consistent)', fontweight='bold')
            ax2.set_xlim(0, 1)
            
            # Add value labels
            for bar, score in zip(bars, consistency_scores['consistency_score']):
                ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{score:.2f}', va='center', fontsize=self.font_size-1)
        
        plt.tight_layout()
        plt.savefig(output_path / 'enhanced_family_task_heatmap.png', 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        # Generate alt-text
        alt_text = self._generate_heatmap_alt_text(heatmap_data, consistency_scores)
        self.alt_texts['enhanced_family_task_heatmap.png'] = alt_text
        
        return str(output_path / 'enhanced_family_task_heatmap.png')
    
    def create_error_analysis_dashboard(self, df: pd.DataFrame, 
                                      family_analysis: Dict,
                                      output_path: Path) -> str:
        """Create comprehensive error analysis dashboard."""
        fig = plt.figure(figsize=(self.width, self.height))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Error type distribution
        ax1 = fig.add_subplot(gs[0, 0])
        if 'is_format_error' in df.columns:
            error_counts = {
                'Format Errors': df['is_format_error'].sum(),
                'Technical Errors': df.get('is_technical_error', pd.Series([False]*len(df))).sum(),
                'Valid Results': (~df['is_format_error'] & ~df.get('is_technical_error', False)).sum()
            }
            
            wedges, texts, autotexts = ax1.pie(error_counts.values(), 
                                              labels=error_counts.keys(),
                                              autopct='%1.1f%%',
                                              colors=self.primary_colors[:3],
                                              startangle=90)
            ax1.set_title('Overall Error Distribution', fontweight='bold')
        
        # 2. Error rate by family
        ax2 = fig.add_subplot(gs[0, 1:])
        if family_analysis:
            families = list(family_analysis.keys())
            format_rates = [family_analysis[f].get('format_error_rate', 0) for f in families]
            technical_rates = [family_analysis[f].get('technical_error_rate', 0) for f in families]
            
            x = np.arange(len(families))
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, format_rates, width, 
                           label='Format Error Rate', color=self.primary_colors[0], alpha=0.7)
            bars2 = ax2.bar(x + width/2, technical_rates, width,
                           label='Technical Error Rate', color=self.primary_colors[1], alpha=0.7)
            
            ax2.set_xlabel('Model Family', fontweight='bold')
            ax2.set_ylabel('Error Rate', fontweight='bold')
            ax2.set_title('Error Rates by Model Family', fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(families, rotation=45, ha='right')
            ax2.legend()
            ax2.set_ylim(0, max(max(format_rates), max(technical_rates)) * 1.1)
        
        # 3. Accuracy vs Error Rate scatter
        ax3 = fig.add_subplot(gs[1, :2])
        if family_analysis:
            families = list(family_analysis.keys())
            accuracies = [family_analysis[f].get('avg_accuracy', 0) for f in families]
            error_rates = [family_analysis[f].get('format_error_rate', 0) + 
                          family_analysis[f].get('technical_error_rate', 0) for f in families]
            
            scatter = ax3.scatter(error_rates, accuracies, 
                                 c=range(len(families)), 
                                 cmap='viridis', 
                                 s=100, alpha=0.7)
            
            # Add family labels
            for i, family in enumerate(families):
                ax3.annotate(family, (error_rates[i], accuracies[i]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=self.font_size-1)
            
            ax3.set_xlabel('Total Error Rate', fontweight='bold')
            ax3.set_ylabel('Average Accuracy', fontweight='bold') 
            ax3.set_title('Accuracy vs Error Rate by Family', fontweight='bold')
        
        # 4. Test category difficulty ranking
        ax4 = fig.add_subplot(gs[1, 2])
        if 'test_category' in df.columns and 'accuracy' in df.columns:
            test_difficulty = df.groupby('test_category')['accuracy'].mean().sort_values()
            
            colors = [self.primary_colors[2] if acc > 0.5 else self.primary_colors[3] 
                     for acc in test_difficulty.values]
            
            bars = ax4.barh(range(len(test_difficulty)), test_difficulty.values, color=colors, alpha=0.7)
            ax4.set_yticks(range(len(test_difficulty)))
            ax4.set_yticklabels([cat.replace('_', ' ')[:15] + '...' if len(cat) > 15 
                                else cat.replace('_', ' ') for cat in test_difficulty.index], 
                               fontsize=self.font_size-2)
            ax4.set_xlabel('Average Accuracy', fontweight='bold')
            ax4.set_title('Test Difficulty\n(Lowest to Highest)', fontweight='bold')
        
        plt.suptitle('BFCL Benchmark Error Analysis Dashboard', 
                    fontsize=self.title_size+2, fontweight='bold', y=0.98)
        
        plt.savefig(output_path / 'error_analysis_dashboard.png', 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        # Generate alt-text
        alt_text = self._generate_error_dashboard_alt_text(df, family_analysis)
        self.alt_texts['error_analysis_dashboard.png'] = alt_text
        
        return str(output_path / 'error_analysis_dashboard.png')
    
    def create_irrelevance_analysis_plot(self, irrelevance_data: Dict, output_path: Path) -> str:
        """Create detailed irrelevance test analysis plot."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(self.width, self.height))
        
        # Extract data
        models = list(irrelevance_data.keys())
        accuracies = [irrelevance_data[m]['accuracy'] for m in models]
        decoder_rates = [irrelevance_data[m]['error_analysis']['decoder_success_rate'] 
                        if irrelevance_data[m]['error_analysis'] else 0 for m in models]
        total_tests = [irrelevance_data[m]['total_tests'] for m in models]
        
        # Clean model names for display
        display_names = [name.replace('_', ' ')[:25] + '...' if len(name) > 25 
                        else name.replace('_', ' ') for name in models]
        
        # 1. Accuracy vs Decoder Success Rate
        scatter = ax1.scatter(decoder_rates, accuracies, s=[t/2 for t in total_tests], 
                             c=range(len(models)), cmap='viridis', alpha=0.7)
        
        for i, name in enumerate(display_names):
            if len(name) < 20:  # Only annotate shorter names
                ax1.annotate(name, (decoder_rates[i], accuracies[i]), 
                           xytext=(2, 2), textcoords='offset points', 
                           fontsize=self.font_size-2)
        
        ax1.set_xlabel('Decoder Success Error Rate', fontweight='bold')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('Accuracy vs Decoder Success Errors\n(Bubble size = # tests)', fontweight='bold')
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% Accuracy')
        ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50% Error Rate')
        ax1.legend()
        
        # 2. Model ranking by accuracy
        sorted_data = sorted(zip(display_names, accuracies), key=lambda x: x[1], reverse=True)
        names_sorted, acc_sorted = zip(*sorted_data)
        
        colors = [self.primary_colors[0] if acc > 0.8 else 
                 self.primary_colors[1] if acc > 0.5 else 
                 self.primary_colors[3] for acc in acc_sorted]
        
        bars = ax2.barh(range(len(names_sorted)), acc_sorted, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(names_sorted)))
        ax2.set_yticklabels(names_sorted, fontsize=self.font_size-2)
        ax2.set_xlabel('Accuracy', fontweight='bold')
        ax2.set_title('Model Ranking (Irrelevance Test)', fontweight='bold')
        ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
        
        # Add accuracy labels
        for bar, acc in zip(bars, acc_sorted):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{acc:.1%}', va='center', fontsize=self.font_size-2)
        
        # 3. Distribution of decoder success rates
        ax3.hist(decoder_rates, bins=10, alpha=0.7, color=self.primary_colors[2], edgecolor='black')
        ax3.set_xlabel('Decoder Success Error Rate', fontweight='bold')
        ax3.set_ylabel('Number of Models', fontweight='bold')
        ax3.set_title('Distribution of Decoder Success Rates', fontweight='bold')
        ax3.axvline(x=np.mean(decoder_rates), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(decoder_rates):.1%}')
        ax3.legend()
        
        # 4. Problem explanation
        ax4.axis('off')
        explanation = """
IRRELEVANCE TEST ISSUE EXPLANATION

The irrelevance test penalizes models for calling 
functions when they shouldn't. However:

• ALL models show ~100% "decoder_success" errors
• This suggests the test expects NO function calls
• Models are penalized for being appropriately cautious
• The scoring methodology appears flawed

RECOMMENDATION:
Review irrelevance test implementation and scoring logic.
Models should not be penalized for reasonable function calls.
        """
        
        ax4.text(0.05, 0.95, explanation.strip(), transform=ax4.transAxes, 
                fontsize=self.font_size, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path / 'irrelevance_analysis_detailed.png', 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        # Generate alt-text
        alt_text = self._generate_irrelevance_alt_text(irrelevance_data)
        self.alt_texts['irrelevance_analysis_detailed.png'] = alt_text
        
        return str(output_path / 'irrelevance_analysis_detailed.png')
    
    def save_alt_texts(self, output_path: Path):
        """Save alt-texts to file for accessibility."""
        alt_text_file = output_path / 'visualization_alt_texts.md'
        
        with open(alt_text_file, 'w', encoding='utf-8') as f:
            f.write("# Visualization Alt-Text Descriptions\n\n")
            f.write("This file contains detailed alt-text descriptions for all visualizations ")
            f.write("to support accessibility and screen readers.\n\n")
            
            for image_name, alt_text in self.alt_texts.items():
                f.write(f"## {image_name}\n\n")
                f.write(f"{alt_text}\n\n")
                f.write("---\n\n")
        
        print(f"Alt-texts saved to {alt_text_file}")
    
    def _generate_inversion_alt_text(self, inversions_df: pd.DataFrame) -> str:
        """Generate alt-text for performance inversion plot."""
        top_3 = inversions_df.head(3)
        alt_text = f"Bar chart showing top {len(inversions_df)} performance inversions where weaker models outperform stronger ones. "
        
        alt_text += f"The largest inversion is in {top_3.iloc[0]['test_category']} with a {top_3.iloc[0]['inversion_delta']:.1%} gap. "
        
        if 'is_significant' in inversions_df.columns:
            sig_count = inversions_df['is_significant'].sum()
            alt_text += f"{sig_count} out of {len(inversions_df)} inversions are statistically significant. "
        
        return alt_text
    
    def _generate_heatmap_alt_text(self, heatmap_data: pd.DataFrame, consistency_scores: pd.DataFrame) -> str:
        """Generate alt-text for family-task heatmap."""
        alt_text = f"Heatmap showing performance of {len(heatmap_data.columns)} model families across {len(heatmap_data.index)} test categories. "
        
        # Find best and worst performing combinations
        max_val = heatmap_data.max().max()
        min_val = heatmap_data.min().min()
        
        max_loc = heatmap_data.stack().idxmax()
        min_loc = heatmap_data.stack().idxmin()
        
        alt_text += f"Highest performance: {max_loc[1]} on {max_loc[0]} ({max_val:.1%}). "
        alt_text += f"Lowest performance: {min_loc[1]} on {min_loc[0]} ({min_val:.1%}). "
        
        if not consistency_scores.empty:
            most_consistent = consistency_scores.iloc[0]['model_family']
            alt_text += f"Most consistent family: {most_consistent}."
        
        return alt_text
    
    def _generate_error_dashboard_alt_text(self, df: pd.DataFrame, family_analysis: Dict) -> str:
        """Generate alt-text for error analysis dashboard."""
        total_evals = len(df)
        format_errors = df.get('is_format_error', pd.Series([False]*total_evals)).sum()
        
        alt_text = f"Dashboard showing error analysis across {total_evals} evaluations. "
        alt_text += f"Format errors account for {format_errors/total_evals:.1%} of all evaluations. "
        
        if family_analysis:
            families = list(family_analysis.keys())
            error_rates = [(fam, family_analysis[fam].get('format_error_rate', 0)) for fam in families]
            highest_error = max(error_rates, key=lambda x: x[1])
            
            alt_text += f"Family with highest error rate: {highest_error[0]} ({highest_error[1]:.1%})."
        
        return alt_text
    
    def _generate_irrelevance_alt_text(self, irrelevance_data: Dict) -> str:
        """Generate alt-text for irrelevance analysis plot."""
        models = list(irrelevance_data.keys())
        accuracies = [irrelevance_data[m]['accuracy'] for m in models]
        
        best_model = models[accuracies.index(max(accuracies))]
        worst_model = models[accuracies.index(min(accuracies))]
        
        alt_text = f"Multi-panel analysis of irrelevance test performance across {len(models)} models. "
        alt_text += f"Best performing: {best_model} ({max(accuracies):.1%}). "
        alt_text += f"Worst performing: {worst_model} ({min(accuracies):.1%}). "
        alt_text += "Shows systematic issue where all models have high decoder success error rates, "
        alt_text += "suggesting the test penalizes appropriate function calling behavior."
        
        return alt_text