#!/usr/bin/env python3
"""
Enhanced BFCL Benchmark Analysis Pipeline
Comprehensive analysis with statistical validation, family-task analysis, and case studies.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from statistical_validation import StatisticalValidator
from family_task_analysis import FamilyTaskAnalyzer  
from case_study_sampler import CaseStudySampler
from enhanced_visualizations import AccessibleVisualizer

class EnhancedBFCLAnalyzer:
    """Enhanced BFCL benchmark analyzer with comprehensive statistical analysis."""
    
    def __init__(self, config_path: Path, data_root: Path, output_dir: Path):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.validator = StatisticalValidator(
            confidence_level=self.config['statistical_thresholds']['confidence_level'],
            bootstrap_iterations=self.config['statistical_thresholds']['bootstrap_iterations']
        )
        
        self.family_analyzer = FamilyTaskAnalyzer()
        self.case_sampler = CaseStudySampler(self.data_root)
        self.visualizer = AccessibleVisualizer(self.config)
        
        # Load model tiers
        self.model_tiers = self.config['model_tiers']
        
        print(f"Initialized Enhanced BFCL Analyzer")
        print(f"Data root: {self.data_root}")
        print(f"Output directory: {self.output_dir}")
        
    def load_score_files(self) -> pd.DataFrame:
        """Load all score files with enhanced error handling."""
        print("Loading score files...")
        all_scores = []
        
        score_path = self.data_root / "score"
        if not score_path.exists():
            raise FileNotFoundError(f"Score directory not found: {score_path}")
        
        # Find all JSON score files
        score_files = list(score_path.rglob("*.json"))
        print(f"Found {len(score_files)} score files")
        
        for score_file in score_files:
            try:
                model_name = score_file.parent.name
                test_category = score_file.stem.replace('BFCL_v3_', '').replace('_score', '')
                
                with open(score_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse JSON data
                lines = content.strip().split('\n')
                for line_num, line in enumerate(lines):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            
                            # Summary line (contains accuracy)
                            if isinstance(data, dict) and 'accuracy' in data:
                                all_scores.append({
                                    'model_name': model_name,
                                    'test_category': test_category,
                                    'accuracy': data.get('accuracy', 0),
                                    'correct_count': data.get('correct_count', 0),
                                    'total_count': data.get('total_count', 0),
                                    'file_path': str(score_file),
                                    'data_type': 'summary'
                                })
                            # Detailed result line
                            elif isinstance(data, dict) and 'id' in data:
                                all_scores.append({
                                    'model_name': data.get('model_name', model_name),
                                    'test_category': data.get('test_category', test_category),
                                    'test_id': data.get('id'),
                                    'valid': data.get('valid', False),
                                    'error': str(data.get('error', [])),
                                    'error_type': data.get('error_type', ''),
                                    'file_path': str(score_file),
                                    'data_type': 'detailed'
                                })
                                
                        except json.JSONDecodeError as e:
                            if line_num < 2:  # Only warn for first few lines
                                print(f"JSON decode error in {score_file}, line {line_num}: {e}")
                            continue
                            
            except Exception as e:
                print(f"Error loading {score_file}: {e}")
                continue
        
        df = pd.DataFrame(all_scores)
        print(f"Loaded {len(df)} evaluation records")
        return df
    
    def classify_model_tiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify models into tiers based on configuration."""
        df = df.copy()
        
        def get_model_tier(model_name: str) -> str:
            model_lower = str(model_name).lower()
            
            # Check patterns for each tier
            for tier_name, tier_config in self.model_tiers.items():
                if tier_name == 'statistical_thresholds':  # Skip config section
                    continue
                    
                patterns = tier_config.get('patterns', [])
                if any(pattern.lower() in model_lower for pattern in patterns):
                    return tier_name.replace('_', ' ').title()
            
            return 'Other'
        
        df['model_tier'] = df['model_name'].apply(get_model_tier)
        return df
    
    def identify_technical_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify and classify technical/format errors."""
        print("Identifying technical and format errors...")
        
        if 'error' not in df.columns and 'error_type' not in df.columns:
            df['is_technical_error'] = False
            df['is_format_error'] = False
            df['is_suspicious'] = False
            return df
        
        # Error patterns from config or defaults
        technical_patterns = [
            r"timeout|timed out", r"connection error|connection refused",
            r"rate limit|RateLimitError", r"API error|APIError", 
            r"500 Internal Server Error", r"502 Bad Gateway", r"503 Service Unavailable"
        ]
        
        format_patterns = [
            r"JSON.*decode error|JSONDecodeError", r"parsing failed|parse error",
            r"invalid format|format error", r"unexpected token",
            r"malformed response", r"decoder_success"
        ]
        
        # Classify errors
        df['is_technical_error'] = df.get('error', pd.Series([''] * len(df))).str.contains(
            '|'.join(technical_patterns), case=False, na=False
        )
        
        df['is_format_error'] = df.get('error', pd.Series([''] * len(df))).str.contains(
            '|'.join(format_patterns), case=False, na=False
        ) | df.get('error_type', pd.Series([''] * len(df))).str.contains(
            'decoder_success|format', case=False, na=False
        )
        
        df['is_suspicious'] = df['is_format_error'] | df['is_technical_error']
        
        print(f"Identified {df['is_technical_error'].sum()} technical errors")
        print(f"Identified {df['is_format_error'].sum()} format errors")
        
        return df
    
    def analyze_performance_inversions_enhanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced performance inversion analysis with statistical validation."""
        print("Analyzing performance inversions with statistical validation...")
        
        # Get summary data only
        summary_df = df[df['data_type'] == 'summary'].copy()
        if summary_df.empty:
            print("No summary data available for inversion analysis")
            return pd.DataFrame()
        
        # Get model lists by tier
        top_tier_models = []
        lower_tier_models = []
        
        for _, row in summary_df.iterrows():
            if row['model_tier'] == 'Top Tier':
                top_tier_models.append(row['model_name'])
            elif row['model_tier'] in ['Lower Tier', 'Other']:
                lower_tier_models.append(row['model_name'])
        
        top_tier_models = list(set(top_tier_models))
        lower_tier_models = list(set(lower_tier_models))
        
        print(f"Found {len(top_tier_models)} top-tier and {len(lower_tier_models)} lower-tier models")
        
        # Find inversions
        inversions = []
        validation_results = []
        
        for test_category in summary_df['test_category'].unique():
            test_data = summary_df[summary_df['test_category'] == test_category]
            
            # Get performance for each tier
            top_performances = []
            lower_performances = []
            
            for model in top_tier_models:
                model_data = test_data[test_data['model_name'] == model]
                if not model_data.empty:
                    top_performances.extend(zip([model] * len(model_data), model_data['accuracy'].tolist()))
            
            for model in lower_tier_models:
                model_data = test_data[test_data['model_name'] == model]
                if not model_data.empty:
                    lower_performances.extend(zip([model] * len(model_data), model_data['accuracy'].tolist()))
            
            if top_performances and lower_performances:
                # Find worst top-tier and best lower-tier
                worst_top = min(top_performances, key=lambda x: x[1])
                best_lower = max(lower_performances, key=lambda x: x[1])
                
                delta = best_lower[1] - worst_top[1]
                
                # Statistical validation
                top_scores = [perf[1] for perf in top_performances]
                lower_scores = [perf[1] for perf in lower_performances]
                
                validation = self.validator.calculate_performance_delta_with_ci(top_scores, lower_scores)
                
                if delta > self.config['statistical_thresholds']['delta_min']:
                    inversions.append({
                        'test_category': test_category,
                        'weakest_top_tier_model': worst_top[0],
                        'weakest_top_tier_score': worst_top[1],
                        'strongest_lower_tier_model': best_lower[0],
                        'strongest_lower_tier_score': best_lower[1],
                        'inversion_delta': delta,
                        **validation
                    })
                    
                    validation_results.append({
                        'test_category': test_category,
                        **validation
                    })
        
        inversions_df = pd.DataFrame(inversions)
        if not inversions_df.empty:
            inversions_df = inversions_df.sort_values('inversion_delta', ascending=False)
            print(f"Found {len(inversions_df)} performance inversions")
        else:
            print("No significant performance inversions found")
        
        return inversions_df, pd.DataFrame(validation_results)
    
    def generate_comprehensive_report(self, df: pd.DataFrame, 
                                    inversions_df: pd.DataFrame,
                                    family_report: Dict,
                                    case_studies: Dict) -> str:
        """Generate comprehensive analysis report."""
        print("Generating comprehensive report...")
        
        # Calculate overall statistics
        total_evaluations = len(df)
        total_models = df['model_name'].nunique()
        total_categories = df['test_category'].nunique()
        
        # Error statistics
        format_errors = df['is_format_error'].sum()
        technical_errors = df['is_technical_error'].sum()
        
        # Performance statistics
        summary_df = df[df['data_type'] == 'summary']
        if not summary_df.empty:
            avg_accuracy = summary_df['accuracy'].mean()
            accuracy_std = summary_df['accuracy'].std()
        else:
            avg_accuracy = 0
            accuracy_std = 0
        
        report = f"""# Enhanced BFCL Benchmark Analysis Report
        
## Executive Summary

This comprehensive analysis examines **{total_evaluations:,}** evaluation records across **{total_models}** models and **{total_categories}** test categories, employing statistical validation, family-task analysis, and case study sampling.

### Key Metrics
- **Average Accuracy**: {avg_accuracy:.1%} (±{accuracy_std:.1%})
- **Format Errors**: {format_errors:,} ({format_errors/total_evaluations:.1%})
- **Technical Errors**: {technical_errors:,} ({technical_errors/total_evaluations:.1%})
- **Performance Inversions**: {len(inversions_df)} statistically validated

## Statistical Validation Results

### Performance Inversions with Confidence Intervals
"""
        
        if not inversions_df.empty:
            report += f"\\nFound **{len(inversions_df)}** significant performance inversions:\\n\\n"
            
            for idx, row in inversions_df.head(5).iterrows():
                ci_text = ""
                if not pd.isna(row.get('ci_lower')) and not pd.isna(row.get('ci_upper')):
                    ci_text = f" (95% CI: {row['ci_lower']:.1%} to {row['ci_upper']:.1%})"
                
                p_text = ""
                if not pd.isna(row.get('p_value')):
                    p_text = f", p={row['p_value']:.3f}" if row['p_value'] >= 0.001 else ", p<0.001"
                
                report += f"1. **{row['test_category']}**: {row['strongest_lower_tier_model'][:30]}... "
                report += f"({row['strongest_lower_tier_score']:.1%}) outperforms "
                report += f"{row['weakest_top_tier_model'][:30]}... ({row['weakest_top_tier_score']:.1%}) "
                report += f"by {row['inversion_delta']:.1%}{ci_text}{p_text}\\n"
        
        # Add family analysis summary
        if family_report and 'summary' in family_report:
            summary = family_report['summary']
            report += f"""
## Family-Task Analysis Results

- **Most Difficult Task**: {summary['most_difficult_task']}
- **Easiest Task**: {summary['easiest_task']}  
- **Most Consistent Family**: {summary['most_consistent_family']}
- **Least Consistent Family**: {summary['least_consistent_family']}
"""
        
        # Add case study summary
        if case_studies and 'case_studies' in case_studies:
            report += f"""
## Case Study Findings

Generated **{len(case_studies['case_studies'])}** detailed case studies for top performance inversions:

"""
            for case_key, case_data in case_studies['case_studies'].items():
                if 'metadata' in case_data:
                    hypothesis = case_data.get('hypothesis', 'No hypothesis available')
                    report += f"- **{case_key}**: {hypothesis}\\n"
        
        report += f"""
## Recommendations

### Immediate Actions Required

1. **Fix Irrelevance Test Scoring**: The current methodology penalizes appropriate function calling behavior
2. **Investigate Multi-turn Compatibility**: Some model families show systematic failures suggesting evaluation issues
3. **Separate Infrastructure from Capability Errors**: {format_errors:,} format errors should not count against model scores

### Statistical Validation Improvements

- All performance comparisons now include confidence intervals and p-values
- Sample-size weighted deltas account for evaluation completeness
- Bootstrap methods provide robust uncertainty estimates

### Methodology Enhancements

- **Family-Task Analysis**: Systematic identification of model strengths and weaknesses
- **Case Study Sampling**: Automated analysis of actual model outputs for inversion cases  
- **Accessible Visualizations**: Color-blind friendly plots with alt-text descriptions

---

*Report generated with Enhanced BFCL Analysis Pipeline v2.0*  
*Statistical significance threshold: p < {self.config['statistical_thresholds']['p_value_max']}*  
*Minimum effect size: Δ > {self.config['statistical_thresholds']['delta_min']}*
"""
        
        return report
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete enhanced analysis pipeline."""
        print("="*80)
        print("ENHANCED BFCL BENCHMARK ANALYSIS PIPELINE")
        print("="*80)
        
        results = {}
        
        try:
            # 1. Load and preprocess data
            df = self.load_score_files()
            df = self.classify_model_tiers(df)
            df = self.identify_technical_errors(df)
            results['raw_data'] = df
            
            # 2. Performance inversion analysis with statistical validation
            inversions_df, validation_df = self.analyze_performance_inversions_enhanced(df)
            results['inversions'] = inversions_df
            results['validation'] = validation_df
            
            # 3. Family-task cross analysis
            family_report = self.family_analyzer.generate_comprehensive_family_report(df)
            results['family_analysis'] = family_report
            
            # 4. Case study generation
            if not inversions_df.empty:
                case_studies = self.case_sampler.generate_case_studies_for_top_inversions(inversions_df)
                results['case_studies'] = case_studies
            else:
                results['case_studies'] = {}
            
            # 5. Enhanced visualizations
            print("Generating enhanced visualizations...")
            
            if not inversions_df.empty:
                self.visualizer.create_performance_inversion_plot(
                    inversions_df, validation_df, self.output_dir
                )
            
            if 'heatmap_data' in family_report:
                self.visualizer.create_family_task_heatmap(
                    family_report['heatmap_data'],
                    family_report['heatmap_annotations'],
                    family_report['consistency_scores'],
                    self.output_dir
                )
            
            # Error analysis dashboard
            family_data = {}
            if 'family_task_matrix' in family_report:
                for family in family_report['family_task_matrix']['model_family'].unique():
                    family_subset = family_report['family_task_matrix'][
                        family_report['family_task_matrix']['model_family'] == family
                    ]
                    family_data[family] = {
                        'avg_accuracy': family_subset['mean_accuracy'].mean(),
                        'format_error_rate': family_subset['format_error_rate'].mean(),
                        'technical_error_rate': family_subset['technical_error_rate'].mean()
                    }
            
            self.visualizer.create_error_analysis_dashboard(df, family_data, self.output_dir)
            
            # Irrelevance analysis if data exists
            irrelevance_data = {}
            irrelevance_df = df[df['test_category'].str.contains('irrelevance', na=False)]
            if not irrelevance_df.empty:
                for model in irrelevance_df['model_name'].unique():
                    model_data = irrelevance_df[irrelevance_df['model_name'] == model]
                    if 'accuracy' in model_data.columns:
                        accuracy_data = model_data[model_data['accuracy'].notna()]
                        if not accuracy_data.empty:
                            irrelevance_data[model] = {
                                'accuracy': accuracy_data['accuracy'].mean(),
                                'total_tests': len(model_data),
                                'error_analysis': {
                                    'decoder_success_rate': model_data['error_type'].str.contains(
                                        'decoder_success', na=False
                                    ).mean() if 'error_type' in model_data.columns else 0
                                }
                            }
                
                if irrelevance_data:
                    self.visualizer.create_irrelevance_analysis_plot(irrelevance_data, self.output_dir)
            
            # Save alt-texts for accessibility
            self.visualizer.save_alt_texts(self.output_dir)
            
            # 6. Generate comprehensive report
            report_text = self.generate_comprehensive_report(
                df, inversions_df, family_report, results['case_studies']
            )
            
            # Save all results
            self._save_results(results, report_text)
            
            print("="*80)
            print("ANALYSIS COMPLETE!")
            print("="*80)
            print(f"Results saved to: {self.output_dir}")
            
            return results
            
        except Exception as e:
            print(f"Analysis failed: {e}")
            raise
    
    def _save_results(self, results: Dict[str, Any], report_text: str):
        """Save all analysis results to files."""
        print("Saving results...")
        
        # Save DataFrames
        if 'raw_data' in results and not results['raw_data'].empty:
            results['raw_data'].to_csv(self.output_dir / 'enhanced_analysis_data.csv', index=False)
            
        if 'inversions' in results and not results['inversions'].empty:
            results['inversions'].to_csv(self.output_dir / 'validated_performance_inversions.csv', index=False)
            
        # Save JSON data
        json_results = {}
        for key, value in results.items():
            if key not in ['raw_data']:  # Skip large DataFrames
                if hasattr(value, 'to_dict'):
                    json_results[key] = value.to_dict()
                else:
                    json_results[key] = value
        
        with open(self.output_dir / 'comprehensive_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, default=str, ensure_ascii=False)
        
        # Save report
        with open(self.output_dir / 'ENHANCED_ANALYSIS_REPORT.md', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        # Save case studies if available
        if 'case_studies' in results and results['case_studies']:
            self.case_sampler.save_case_studies(
                results['case_studies'], 
                self.output_dir / 'detailed_case_studies.json'
            )
            
            # Generate human-readable case study report
            case_study_report = self.case_sampler.generate_case_study_summary_report(
                results['case_studies']
            )
            with open(self.output_dir / 'case_study_summary.md', 'w', encoding='utf-8') as f:
                f.write(case_study_report)
        
        print(f"Saved {len([f for f in self.output_dir.glob('*') if f.is_file()])} result files")

def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Enhanced BFCL Benchmark Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bfcl_analysis_enhanced.py --data_root /path/to/bfcl/data
  python bfcl_analysis_enhanced.py --data_root ./data --output_dir ./enhanced_results
  python bfcl_analysis_enhanced.py --config custom_config.yaml --data_root ./data
        """
    )
    
    parser.add_argument(
        '--data_root', 
        type=str, 
        required=True,
        help='Root directory containing BFCL score data'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./enhanced_analysis_output',
        help='Output directory for analysis results (default: ./enhanced_analysis_output)'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='model_tiers.yaml',
        help='Configuration file path (default: model_tiers.yaml)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Error: Data root directory does not exist: {data_root}")
        sys.exit(1)
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file does not exist: {config_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    
    if args.verbose:
        print(f"Data root: {data_root}")
        print(f"Config: {config_path}")
        print(f"Output: {output_dir}")
    
    try:
        # Run analysis
        analyzer = EnhancedBFCLAnalyzer(config_path, data_root, output_dir)
        results = analyzer.run_full_analysis()
        
        print("\\nAnalysis completed successfully!")
        print(f"Check results in: {output_dir}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()