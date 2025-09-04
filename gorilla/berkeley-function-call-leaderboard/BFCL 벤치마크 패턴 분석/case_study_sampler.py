"""
Case Study Sampling module for BFCL benchmark analysis.
Automatically samples and analyzes model outputs for performance inversions.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import random
import re
from collections import defaultdict

class CaseStudySampler:
    """Sample and analyze model outputs for case studies."""
    
    def __init__(self, data_root: Path, random_seed: int = 42):
        self.data_root = Path(data_root)
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize output storage
        self.case_studies = defaultdict(list)
        
    def load_detailed_results(self, model_name: str, test_category: str) -> Optional[List[Dict]]:
        """Load detailed results for a specific model and test category."""
        score_file = self.data_root / "score" / model_name / f"BFCL_v3_{test_category}_score.json"
        
        if not score_file.exists():
            return None
            
        detailed_results = []
        try:
            with open(score_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse line-by-line JSON
            lines = content.strip().split('\n')
            for line in lines[1:]:  # Skip first line (summary)
                if line.strip():
                    try:
                        data = json.loads(line)
                        if 'id' in data:  # Detailed result
                            detailed_results.append(data)
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            print(f"Error loading {score_file}: {e}")
            return None
            
        return detailed_results
    
    def analyze_output_patterns(self, model_outputs: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in model outputs."""
        if not model_outputs:
            return {}
            
        patterns = {
            'total_outputs': len(model_outputs),
            'valid_outputs': 0,
            'format_errors': 0,
            'decoder_success_errors': 0,
            'other_errors': 0,
            'error_types': defaultdict(int),
            'common_error_messages': defaultdict(int),
            'output_length_stats': [],
            'function_call_patterns': defaultdict(int)
        }
        
        for output in model_outputs:
            # Basic categorization
            is_valid = output.get('valid', False)
            error_type = output.get('error_type', '')
            error_messages = output.get('error', [])
            
            if is_valid:
                patterns['valid_outputs'] += 1
            else:
                if 'decoder_success' in error_type:
                    patterns['decoder_success_errors'] += 1
                elif 'format' in error_type.lower():
                    patterns['format_errors'] += 1
                else:
                    patterns['other_errors'] += 1
                    
                patterns['error_types'][error_type] += 1
                
                # Collect error messages
                if isinstance(error_messages, list):
                    for msg in error_messages:
                        patterns['common_error_messages'][str(msg)] += 1
                else:
                    patterns['common_error_messages'][str(error_messages)] += 1
            
            # Analyze model output
            model_result = output.get('model_result', '')
            if model_result:
                patterns['output_length_stats'].append(len(model_result))
                
                # Extract function call patterns
                function_calls = re.findall(r'(\w+)\(', model_result)
                for func in function_calls:
                    patterns['function_call_patterns'][func] += 1
        
        # Calculate statistics
        if patterns['output_length_stats']:
            length_stats = patterns['output_length_stats']
            patterns['output_length_mean'] = np.mean(length_stats)
            patterns['output_length_std'] = np.std(length_stats)
            patterns['output_length_median'] = np.median(length_stats)
        
        # Convert defaultdicts to regular dicts for JSON serialization
        patterns['error_types'] = dict(patterns['error_types'])
        patterns['common_error_messages'] = dict(patterns['common_error_messages'])
        patterns['function_call_patterns'] = dict(patterns['function_call_patterns'])
        
        return patterns
    
    def sample_outputs_for_comparison(self, 
                                    strong_model: str,
                                    weak_model: str, 
                                    test_category: str,
                                    n_samples: int = 5) -> Dict[str, Any]:
        """Sample outputs from strong and weak models for comparison."""
        
        strong_outputs = self.load_detailed_results(strong_model, test_category)
        weak_outputs = self.load_detailed_results(weak_model, test_category)
        
        if not strong_outputs or not weak_outputs:
            return {
                'error': f'Could not load results for {strong_model} or {weak_model} on {test_category}',
                'strong_model': strong_model,
                'weak_model': weak_model,
                'test_category': test_category
            }
        
        # Sample outputs
        strong_sample = random.sample(strong_outputs, min(n_samples, len(strong_outputs)))
        weak_sample = random.sample(weak_outputs, min(n_samples, len(weak_outputs)))
        
        # Analyze patterns
        strong_patterns = self.analyze_output_patterns(strong_outputs)
        weak_patterns = self.analyze_output_patterns(weak_outputs)
        
        # Create comparison
        comparison = {
            'metadata': {
                'strong_model': strong_model,
                'weak_model': weak_model,
                'test_category': test_category,
                'sampling_date': pd.Timestamp.now().isoformat(),
                'random_seed': self.random_seed,
                'n_samples': n_samples
            },
            'sample_outputs': {
                'strong_model_samples': self._clean_outputs_for_export(strong_sample),
                'weak_model_samples': self._clean_outputs_for_export(weak_sample)
            },
            'pattern_analysis': {
                'strong_model_patterns': strong_patterns,
                'weak_model_patterns': weak_patterns,
                'comparison_metrics': self._compare_patterns(strong_patterns, weak_patterns)
            },
            'key_differences': self._identify_key_differences(strong_sample, weak_sample),
            'hypothesis': self._generate_hypothesis(strong_patterns, weak_patterns, test_category)
        }
        
        return comparison
    
    def _clean_outputs_for_export(self, outputs: List[Dict]) -> List[Dict]:
        """Clean and format outputs for export."""
        cleaned = []
        for output in outputs:
            cleaned_output = {
                'test_id': output.get('id'),
                'question': self._extract_question(output.get('prompt', {})),
                'available_functions': self._extract_functions(output.get('prompt', {})),
                'model_output': output.get('model_result'),
                'decoded_result': output.get('decoded_result'),
                'is_valid': output.get('valid'),
                'error_type': output.get('error_type'),
                'error_messages': output.get('error', [])
            }
            cleaned.append(cleaned_output)
        return cleaned
    
    def _extract_question(self, prompt: Dict) -> str:
        """Extract question from prompt structure."""
        if 'question' in prompt:
            question = prompt['question']
            if isinstance(question, list) and len(question) > 0:
                if isinstance(question[0], list) and len(question[0]) > 0:
                    if isinstance(question[0][0], dict) and 'content' in question[0][0]:
                        return question[0][0]['content']
        return str(prompt.get('question', 'N/A'))
    
    def _extract_functions(self, prompt: Dict) -> List[Dict]:
        """Extract function definitions from prompt."""
        functions = prompt.get('function', [])
        if isinstance(functions, list):
            return [
                {
                    'name': func.get('name', 'unknown'),
                    'description': func.get('description', ''),
                    'parameters': func.get('parameters', {})
                }
                for func in functions
            ]
        return []
    
    def _compare_patterns(self, strong_patterns: Dict, weak_patterns: Dict) -> Dict:
        """Compare patterns between strong and weak models."""
        comparison = {}
        
        # Compare error rates
        strong_total = strong_patterns.get('total_outputs', 1)
        weak_total = weak_patterns.get('total_outputs', 1)
        
        comparison['valid_rate_difference'] = (
            weak_patterns.get('valid_outputs', 0) / weak_total - 
            strong_patterns.get('valid_outputs', 0) / strong_total
        )
        
        comparison['format_error_rate_difference'] = (
            weak_patterns.get('format_errors', 0) / weak_total -
            strong_patterns.get('format_errors', 0) / strong_total
        )
        
        comparison['decoder_success_error_difference'] = (
            weak_patterns.get('decoder_success_errors', 0) / weak_total -
            strong_patterns.get('decoder_success_errors', 0) / strong_total
        )
        
        # Compare output lengths
        strong_length = strong_patterns.get('output_length_mean', 0)
        weak_length = weak_patterns.get('output_length_mean', 0)
        comparison['output_length_difference'] = weak_length - strong_length
        
        return comparison
    
    def _identify_key_differences(self, strong_sample: List[Dict], weak_sample: List[Dict]) -> Dict:
        """Identify key qualitative differences between samples."""
        differences = {
            'output_style_differences': [],
            'error_pattern_differences': [],
            'function_usage_differences': []
        }
        
        # Analyze output styles
        strong_styles = [self._categorize_output_style(output) for output in strong_sample]
        weak_styles = [self._categorize_output_style(output) for output in weak_sample]
        
        strong_style_counts = pd.Series(strong_styles).value_counts()
        weak_style_counts = pd.Series(weak_styles).value_counts()
        
        differences['output_style_differences'] = {
            'strong_model_styles': strong_style_counts.to_dict(),
            'weak_model_styles': weak_style_counts.to_dict()
        }
        
        return differences
    
    def _categorize_output_style(self, output: Dict) -> str:
        """Categorize the style of model output."""
        model_result = output.get('model_result', '')
        
        if not model_result:
            return 'empty'
        elif model_result.startswith('[') and model_result.endswith(']'):
            return 'list_format'
        elif '(' in model_result and ')' in model_result:
            return 'function_call'
        elif '{' in model_result and '}' in model_result:
            return 'json_format'
        else:
            return 'text_format'
    
    def _generate_hypothesis(self, strong_patterns: Dict, weak_patterns: Dict, test_category: str) -> str:
        """Generate hypothesis for performance inversion."""
        hypotheses = []
        
        # Check decoder success patterns
        strong_decoder_rate = strong_patterns.get('decoder_success_errors', 0) / max(strong_patterns.get('total_outputs', 1), 1)
        weak_decoder_rate = weak_patterns.get('decoder_success_errors', 0) / max(weak_patterns.get('total_outputs', 1), 1)
        
        if 'irrelevance' in test_category.lower():
            if strong_decoder_rate > weak_decoder_rate:
                hypotheses.append("Strong model is more aggressive in function calling, leading to penalty in irrelevance test")
        
        # Check format error patterns
        strong_format_rate = strong_patterns.get('format_errors', 0) / max(strong_patterns.get('total_outputs', 1), 1)
        weak_format_rate = weak_patterns.get('format_errors', 0) / max(weak_patterns.get('total_outputs', 1), 1)
        
        if strong_format_rate > weak_format_rate:
            hypotheses.append("Strong model has format compatibility issues with evaluation framework")
        
        # Check multi-turn patterns
        if 'multi_turn' in test_category.lower():
            if strong_patterns.get('valid_outputs', 0) < weak_patterns.get('valid_outputs', 0):
                hypotheses.append("Strong model may have prompt format incompatibility in multi-turn scenarios")
        
        if not hypotheses:
            hypotheses.append("Performance inversion may be due to evaluation methodology issues rather than model capability")
        
        return "; ".join(hypotheses)
    
    def generate_case_studies_for_top_inversions(self, 
                                               inversions_df: pd.DataFrame,
                                               top_k: int = 3,
                                               samples_per_case: int = 5) -> Dict[str, Any]:
        """Generate case studies for top performance inversions."""
        case_studies = {}
        
        top_inversions = inversions_df.head(top_k)
        
        for idx, row in top_inversions.iterrows():
            case_key = f"{row['test_category']}_{row['weakest_top_tier_model']}_vs_{row['strongest_lower_tier_model']}"
            
            case_study = self.sample_outputs_for_comparison(
                strong_model=row['weakest_top_tier_model'],
                weak_model=row['strongest_lower_tier_model'],
                test_category=row['test_category'],
                n_samples=samples_per_case
            )
            
            # Add inversion metadata
            case_study['inversion_metadata'] = {
                'inversion_delta': row['inversion_delta'],
                'strong_model_score': row['weakest_top_tier_score'],
                'weak_model_score': row['strongest_lower_tier_score'],
                'rank': idx + 1
            }
            
            case_studies[case_key] = case_study
        
        return {
            'case_studies': case_studies,
            'generation_metadata': {
                'total_cases': len(case_studies),
                'samples_per_case': samples_per_case,
                'generation_date': pd.Timestamp.now().isoformat(),
                'random_seed': self.random_seed
            }
        }
    
    def save_case_studies(self, case_studies: Dict, output_path: Path):
        """Save case studies to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(case_studies, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Case studies saved to {output_path}")
    
    def generate_case_study_summary_report(self, case_studies: Dict) -> str:
        """Generate human-readable summary of case studies."""
        if 'case_studies' not in case_studies:
            return "No case studies available"
        
        report_lines = ["# Case Study Analysis Report\n"]
        
        for case_key, case_data in case_studies['case_studies'].items():
            if 'error' in case_data:
                continue
                
            metadata = case_data.get('metadata', {})
            inversion_meta = case_data.get('inversion_metadata', {})
            patterns = case_data.get('pattern_analysis', {})
            
            report_lines.extend([
                f"## Case {inversion_meta.get('rank', '?')}: {case_key}\n",
                f"**Performance Gap**: {inversion_meta.get('inversion_delta', 0):.1%}\n",
                f"**Strong Model**: {metadata.get('strong_model')} ({inversion_meta.get('strong_model_score', 0):.1%})\n",
                f"**Weak Model**: {metadata.get('weak_model')} ({inversion_meta.get('weak_model_score', 0):.1%})\n",
                f"**Test Category**: {metadata.get('test_category')}\n"
            ])
            
            # Add pattern analysis
            if patterns:
                comparison = patterns.get('comparison_metrics', {})
                report_lines.extend([
                    f"\n**Key Findings**:\n",
                    f"- Valid rate difference: {comparison.get('valid_rate_difference', 0):.1%}\n",
                    f"- Format error difference: {comparison.get('format_error_rate_difference', 0):.1%}\n",
                    f"- Output length difference: {comparison.get('output_length_difference', 0):.0f} chars\n"
                ])
            
            # Add hypothesis
            hypothesis = case_data.get('hypothesis', 'No hypothesis generated')
            report_lines.append(f"\n**Hypothesis**: {hypothesis}\n\n")
            report_lines.append("---\n\n")
        
        return "".join(report_lines)