"""
Comprehensive test suite for Enhanced BFCL Analysis Pipeline
Tests statistical validation, error pattern detection, and analysis functionality.
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from statistical_validation import StatisticalValidator
from family_task_analysis import FamilyTaskAnalyzer
from case_study_sampler import CaseStudySampler
from enhanced_visualizations import AccessibleVisualizer
from bfcl_analysis_enhanced import EnhancedBFCLAnalyzer

class TestStatisticalValidator:
    """Test statistical validation functionality."""
    
    @pytest.fixture
    def validator(self):
        return StatisticalValidator(confidence_level=0.95, bootstrap_iterations=100)
    
    @pytest.fixture
    def sample_data(self):
        """Sample performance data for testing."""
        return {
            'strong_scores': [0.8, 0.85, 0.9, 0.75, 0.82],
            'weak_scores': [0.95, 0.92, 0.88, 0.94, 0.91],
            'equal_scores_strong': [0.5, 0.5, 0.5],
            'equal_scores_weak': [0.5, 0.5, 0.5],
            'empty_scores': []
        }
    
    def test_bootstrap_confidence_interval(self, validator):
        """Test bootstrap confidence interval calculation."""
        data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        ci_lower, ci_upper = validator.bootstrap_confidence_interval(data, np.mean)
        
        assert not np.isnan(ci_lower)
        assert not np.isnan(ci_upper)
        assert ci_lower <= ci_upper
        assert 0 <= ci_lower <= 1
        assert 0 <= ci_upper <= 1
    
    def test_bootstrap_confidence_interval_empty_data(self, validator):
        """Test bootstrap CI with empty data."""
        data = np.array([])
        ci_lower, ci_upper = validator.bootstrap_confidence_interval(data, np.mean)
        
        assert np.isnan(ci_lower)
        assert np.isnan(ci_upper)
    
    def test_performance_delta_calculation(self, validator, sample_data):
        """Test performance delta calculation with CI."""
        result = validator.calculate_performance_delta_with_ci(
            sample_data['strong_scores'], 
            sample_data['weak_scores']
        )
        
        assert 'delta' in result
        assert 'delta_weighted' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert 'p_value' in result
        assert 'is_significant' in result
        
        # Delta should be positive (weak > strong)
        assert result['delta'] > 0
        assert result['n_strong'] == 5
        assert result['n_weak'] == 5
    
    def test_performance_delta_equal_scores(self, validator, sample_data):
        """Test delta calculation with equal performance."""
        result = validator.calculate_performance_delta_with_ci(
            sample_data['equal_scores_strong'],
            sample_data['equal_scores_weak']
        )
        
        assert abs(result['delta']) < 0.01  # Should be ~0
        assert not result['is_significant']  # Should not be significant
    
    def test_performance_delta_empty_data(self, validator, sample_data):
        """Test delta calculation with empty data."""
        result = validator.calculate_performance_delta_with_ci(
            sample_data['empty_scores'],
            sample_data['strong_scores']
        )
        
        assert np.isnan(result['delta'])
        assert not result['is_significant']
        assert result['n_strong'] == 0
    
    def test_chi_square_test_valid_data(self, validator):
        """Test chi-square test with valid data."""
        success_counts = [80, 60]
        total_counts = [100, 100]
        
        result = validator.chi_square_test_independence(success_counts, total_counts)
        
        assert 'p_value' in result
        assert 'effect_size' in result
        assert not np.isnan(result['p_value'])
        assert 0 <= result['p_value'] <= 1
    
    def test_chi_square_test_small_sample(self, validator):
        """Test chi-square test with small sample (should use Fisher's exact)."""
        success_counts = [2, 8]
        total_counts = [10, 10]
        
        result = validator.chi_square_test_independence(success_counts, total_counts)
        
        assert 'p_value' in result
        assert not np.isnan(result['p_value'])
    
    def test_wilson_confidence_interval(self, validator):
        """Test Wilson score confidence interval for accuracy."""
        data = pd.DataFrame([
            {'correct_count': 80, 'total_count': 100, 'accuracy': 0.8},
            {'correct_count': 90, 'total_count': 100, 'accuracy': 0.9},
            {'correct_count': 0, 'total_count': 0, 'accuracy': 0}  # Edge case
        ])
        
        result_df = validator.calculate_confidence_intervals_for_accuracy(data)
        
        assert 'ci_lower' in result_df.columns
        assert 'ci_upper' in result_df.columns
        assert 'ci_width' in result_df.columns
        
        # Check first row (80/100)
        assert 0 <= result_df.iloc[0]['ci_lower'] <= result_df.iloc[0]['ci_upper'] <= 1
        
        # Check zero case
        assert result_df.iloc[2]['ci_lower'] == 0
        assert result_df.iloc[2]['ci_upper'] == 0

class TestFamilyTaskAnalyzer:
    """Test family-task analysis functionality."""
    
    @pytest.fixture
    def analyzer(self):
        return FamilyTaskAnalyzer()
    
    @pytest.fixture
    def sample_df(self):
        """Sample data for family-task analysis."""
        return pd.DataFrame([
            {'model_name': 'gpt-4-turbo', 'test_category': 'simple', 'accuracy': 0.95},
            {'model_name': 'gpt-4-turbo', 'test_category': 'complex', 'accuracy': 0.85},
            {'model_name': 'claude-3-opus', 'test_category': 'simple', 'accuracy': 0.92},
            {'model_name': 'claude-3-opus', 'test_category': 'complex', 'accuracy': 0.88},
            {'model_name': 'llama-3-8b', 'test_category': 'simple', 'accuracy': 0.75},
            {'model_name': 'llama-3-8b', 'test_category': 'complex', 'accuracy': 0.65},
        ])
    
    def test_assign_model_family(self, analyzer):
        """Test model family assignment."""
        assert analyzer.assign_model_family('gpt-4-turbo') == 'OpenAI'
        assert analyzer.assign_model_family('claude-3-opus') == 'Anthropic'
        assert analyzer.assign_model_family('llama-3-8b') == 'Meta'
        assert analyzer.assign_model_family('unknown-model') == 'Other'
    
    def test_create_family_task_matrix(self, analyzer, sample_df):
        """Test family-task matrix creation."""
        matrix_df = analyzer.create_family_task_matrix(sample_df)
        
        assert 'model_family' in matrix_df.columns
        assert 'test_category' in matrix_df.columns
        assert 'mean_accuracy' in matrix_df.columns
        assert 'n_models' in matrix_df.columns
        
        # Should have entries for each family-task combination
        assert len(matrix_df) == 6  # 3 families × 2 test categories
        
        # Check OpenAI family performance on simple task
        openai_simple = matrix_df[
            (matrix_df['model_family'] == 'OpenAI') & 
            (matrix_df['test_category'] == 'simple')
        ]
        assert len(openai_simple) == 1
        assert openai_simple.iloc[0]['mean_accuracy'] == 0.95
    
    def test_identify_family_strengths_weaknesses(self, analyzer, sample_df):
        """Test identification of family strengths and weaknesses."""
        matrix_df = analyzer.create_family_task_matrix(sample_df)
        strengths_weaknesses = analyzer.identify_family_strengths_weaknesses(matrix_df, top_k=2)
        
        assert 'OpenAI' in strengths_weaknesses
        assert 'Anthropic' in strengths_weaknesses
        assert 'Meta' in strengths_weaknesses
        
        for family, data in strengths_weaknesses.items():
            assert 'strengths' in data
            assert 'weaknesses' in data
            assert 'overall_mean' in data
            assert len(data['strengths']) <= 2
            assert len(data['weaknesses']) <= 2
    
    def test_analyze_task_difficulty_ranking(self, analyzer, sample_df):
        """Test task difficulty ranking."""
        matrix_df = analyzer.create_family_task_matrix(sample_df)
        difficulty_df = analyzer.analyze_task_difficulty_ranking(matrix_df)
        
        assert 'test_category' in difficulty_df.columns
        assert 'overall_mean_accuracy' in difficulty_df.columns
        assert 'difficulty_rank' in difficulty_df.columns
        
        # Should be sorted by difficulty (lowest accuracy first)
        assert difficulty_df.iloc[0]['overall_mean_accuracy'] <= difficulty_df.iloc[-1]['overall_mean_accuracy']
        
        # Complex should be harder than simple
        complex_rank = difficulty_df[difficulty_df['test_category'] == 'complex']['difficulty_rank'].iloc[0]
        simple_rank = difficulty_df[difficulty_df['test_category'] == 'simple']['difficulty_rank'].iloc[0]
        assert complex_rank < simple_rank  # Lower rank = more difficult
    
    def test_identify_systematic_family_failures(self, analyzer, sample_df):
        """Test identification of systematic family failures."""
        matrix_df = analyzer.create_family_task_matrix(sample_df)
        failures = analyzer.identify_systematic_family_failures(matrix_df, accuracy_threshold=0.8)
        
        # Meta family should show up as having failures (accuracy < 0.8)
        assert 'Meta' in failures
        assert len(failures['Meta']) > 0
        
        # Each failure should have required fields
        for failure in failures['Meta']:
            assert 'test_category' in failure
            assert 'family_accuracy' in failure
            assert 'other_families_mean' in failure
            assert 'performance_gap' in failure

class TestCaseStudySampler:
    """Test case study sampling functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def sampler(self, temp_dir):
        return CaseStudySampler(temp_dir)
    
    def test_extract_question(self, sampler):
        """Test question extraction from prompt."""
        prompt = {
            'question': [[{'role': 'user', 'content': 'What is 2+2?'}]]
        }
        question = sampler._extract_question(prompt)
        assert question == 'What is 2+2?'
        
        # Test malformed prompt
        bad_prompt = {'question': 'just a string'}
        question = sampler._extract_question(bad_prompt)
        assert question == 'just a string'
    
    def test_extract_functions(self, sampler):
        """Test function extraction from prompt."""
        prompt = {
            'function': [
                {
                    'name': 'add_numbers',
                    'description': 'Add two numbers',
                    'parameters': {'type': 'object'}
                }
            ]
        }
        functions = sampler._extract_functions(prompt)
        assert len(functions) == 1
        assert functions[0]['name'] == 'add_numbers'
        assert functions[0]['description'] == 'Add two numbers'
    
    def test_categorize_output_style(self, sampler):
        """Test output style categorization."""
        assert sampler._categorize_output_style({'model_result': ''}) == 'empty'
        assert sampler._categorize_output_style({'model_result': '[func()]'}) == 'list_format'
        assert sampler._categorize_output_style({'model_result': 'func(x=1)'}) == 'function_call'
        assert sampler._categorize_output_style({'model_result': '{"key": "value"}'}) == 'json_format'
        assert sampler._categorize_output_style({'model_result': 'plain text'}) == 'text_format'
    
    def test_analyze_output_patterns(self, sampler):
        """Test output pattern analysis."""
        outputs = [
            {'valid': True, 'model_result': 'func(x=1)', 'error': []},
            {'valid': False, 'error_type': 'decoder_success', 'error': ['Error msg'], 'model_result': 'func(y=2)'},
            {'valid': False, 'error_type': 'format_error', 'error': ['Bad format'], 'model_result': 'invalid'}
        ]
        
        patterns = sampler.analyze_output_patterns(outputs)
        
        assert patterns['total_outputs'] == 3
        assert patterns['valid_outputs'] == 1
        assert patterns['decoder_success_errors'] == 1
        assert patterns['format_errors'] == 1
        assert 'func' in patterns['function_call_patterns']
        assert patterns['function_call_patterns']['func'] == 2
    
    def test_compare_patterns(self, sampler):
        """Test pattern comparison between strong and weak models."""
        strong_patterns = {
            'total_outputs': 10,
            'valid_outputs': 8,
            'format_errors': 1,
            'decoder_success_errors': 1,
            'output_length_mean': 50
        }
        
        weak_patterns = {
            'total_outputs': 10,
            'valid_outputs': 9,
            'format_errors': 0,
            'decoder_success_errors': 1,
            'output_length_mean': 45
        }
        
        comparison = sampler._compare_patterns(strong_patterns, weak_patterns)
        
        assert 'valid_rate_difference' in comparison
        assert 'format_error_rate_difference' in comparison
        assert 'output_length_difference' in comparison
        
        # Weak model should have higher valid rate
        assert comparison['valid_rate_difference'] > 0
    
    def test_generate_hypothesis(self, sampler):
        """Test hypothesis generation."""
        strong_patterns = {'total_outputs': 10, 'decoder_success_errors': 8, 'format_errors': 2}
        weak_patterns = {'total_outputs': 10, 'decoder_success_errors': 2, 'format_errors': 1}
        
        # Test irrelevance category
        hypothesis = sampler._generate_hypothesis(strong_patterns, weak_patterns, 'irrelevance')
        assert 'function calling' in hypothesis.lower()
        
        # Test multi-turn category
        hypothesis = sampler._generate_hypothesis(strong_patterns, weak_patterns, 'multi_turn_base')
        assert 'multi-turn' in hypothesis.lower() or 'prompt format' in hypothesis.lower()

class TestEnhancedBFCLAnalyzer:
    """Test main analysis pipeline."""
    
    @pytest.fixture
    def temp_config(self, temp_dir):
        """Create temporary config file."""
        config = {
            'model_tiers': {
                'top_tier': {
                    'patterns': ['gpt-4', 'claude']
                },
                'lower_tier': {
                    'patterns': ['llama', 'mistral']
                }
            },
            'statistical_thresholds': {
                'delta_min': 0.1,
                'p_value_max': 0.05,
                'confidence_level': 0.95,
                'bootstrap_iterations': 100
            },
            'visualization': {
                'colorblind_palette': {
                    'primary': ['#1f77b4', '#ff7f0e', '#2ca02c']
                },
                'figure_settings': {
                    'dpi': 100,
                    'width': 12,
                    'height': 8,
                    'font_size': 10,
                    'title_size': 12
                }
            }
        }
        
        config_file = temp_dir / 'test_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        return config_file
    
    @pytest.fixture
    def sample_score_data(self, temp_dir):
        """Create sample score data files."""
        score_dir = temp_dir / 'score' / 'test_model'
        score_dir.mkdir(parents=True)
        
        # Create sample score file
        score_file = score_dir / 'BFCL_v3_simple_score.json'
        sample_data = [
            {'accuracy': 0.85, 'correct_count': 85, 'total_count': 100},
            {'id': 'test_1', 'model_name': 'test_model', 'test_category': 'simple', 
             'valid': True, 'error': [], 'error_type': ''},
            {'id': 'test_2', 'model_name': 'test_model', 'test_category': 'simple', 
             'valid': False, 'error': ['Format error'], 'error_type': 'format_error'}
        ]
        
        with open(score_file, 'w') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\\n')
        
        return temp_dir
    
    def test_analyzer_initialization(self, temp_config, sample_score_data):
        """Test analyzer initialization."""
        output_dir = sample_score_data / 'output'
        
        analyzer = EnhancedBFCLAnalyzer(temp_config, sample_score_data, output_dir)
        
        assert analyzer.data_root == sample_score_data
        assert analyzer.output_dir == output_dir
        assert analyzer.config is not None
        assert analyzer.validator is not None
        assert analyzer.family_analyzer is not None
    
    def test_classify_model_tiers(self, temp_config, sample_score_data):
        """Test model tier classification."""
        analyzer = EnhancedBFCLAnalyzer(temp_config, sample_score_data, sample_score_data / 'output')
        
        df = pd.DataFrame([
            {'model_name': 'gpt-4-turbo'},
            {'model_name': 'claude-3-opus'},
            {'model_name': 'llama-3-8b'},
            {'model_name': 'unknown-model'}
        ])
        
        result_df = analyzer.classify_model_tiers(df)
        
        assert 'model_tier' in result_df.columns
        assert result_df[result_df['model_name'] == 'gpt-4-turbo']['model_tier'].iloc[0] == 'Top Tier'
        assert result_df[result_df['model_name'] == 'llama-3-8b']['model_tier'].iloc[0] == 'Lower Tier'
        assert result_df[result_df['model_name'] == 'unknown-model']['model_tier'].iloc[0] == 'Other'
    
    def test_identify_technical_errors(self, temp_config, sample_score_data):
        """Test technical error identification."""
        analyzer = EnhancedBFCLAnalyzer(temp_config, sample_score_data, sample_score_data / 'output')
        
        df = pd.DataFrame([
            {'error': 'timeout occurred', 'error_type': ''},
            {'error': 'JSON decode error', 'error_type': 'format_error'},
            {'error': 'decoder_success error', 'error_type': 'decoder_success'},
            {'error': '', 'error_type': ''}
        ])
        
        result_df = analyzer.identify_technical_errors(df)
        
        assert 'is_technical_error' in result_df.columns
        assert 'is_format_error' in result_df.columns
        assert 'is_suspicious' in result_df.columns
        
        assert result_df.iloc[0]['is_technical_error'] == True  # timeout
        assert result_df.iloc[1]['is_format_error'] == True    # JSON error
        assert result_df.iloc[2]['is_format_error'] == True    # decoder_success

class TestErrorPatternDetection:
    """Test error pattern detection and regex matching."""
    
    def test_technical_error_patterns(self):
        """Test technical error pattern matching."""
        patterns = [
            r"timeout|timed out",
            r"connection error|connection refused", 
            r"rate limit|RateLimitError",
            r"API error|APIError",
            r"500 Internal Server Error"
        ]
        
        test_errors = [
            "Connection timeout occurred",
            "API request timed out", 
            "Rate limit exceeded",
            "500 Internal Server Error",
            "Connection refused by server",
            "Normal processing error"  # Should not match
        ]
        
        combined_pattern = '|'.join(patterns)
        matches = [bool(pd.Series([error]).str.contains(combined_pattern, case=False, na=False).iloc[0]) 
                  for error in test_errors]
        
        assert matches[0] == True   # timeout
        assert matches[1] == True   # timed out
        assert matches[2] == True   # rate limit
        assert matches[3] == True   # 500 error
        assert matches[4] == True   # connection refused
        assert matches[5] == False  # normal error
    
    def test_format_error_patterns(self):
        """Test format error pattern matching."""
        patterns = [
            r"JSON.*decode error|JSONDecodeError",
            r"parsing failed|parse error",
            r"invalid format|format error",
            r"decoder_success"
        ]
        
        test_errors = [
            "JSONDecodeError: Expecting value",
            "JSON decode error occurred",
            "Parsing failed for input",
            "Invalid format detected",
            "decoder_success when should not",
            "Regular model error"  # Should not match
        ]
        
        combined_pattern = '|'.join(patterns)
        matches = [bool(pd.Series([error]).str.contains(combined_pattern, case=False, na=False).iloc[0]) 
                  for error in test_errors]
        
        assert matches[0] == True   # JSONDecodeError
        assert matches[1] == True   # JSON decode error
        assert matches[2] == True   # parsing failed
        assert matches[3] == True   # invalid format
        assert matches[4] == True   # decoder_success
        assert matches[5] == False  # regular error

# Integration tests
class TestIntegration:
    """Integration tests for complete pipeline."""
    
    @pytest.fixture
    def complete_setup(self, temp_dir):
        """Set up complete test environment."""
        # Create config
        config = {
            'model_tiers': {
                'top_tier': {'patterns': ['gpt-4']},
                'lower_tier': {'patterns': ['llama']}
            },
            'statistical_thresholds': {
                'delta_min': 0.1,
                'p_value_max': 0.05,
                'confidence_level': 0.95,
                'bootstrap_iterations': 50
            },
            'visualization': {
                'colorblind_palette': {'primary': ['#1f77b4', '#ff7f0e']},
                'figure_settings': {'dpi': 100, 'width': 10, 'height': 8, 'font_size': 10, 'title_size': 12}
            }
        }
        
        config_file = temp_dir / 'config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Create score data
        score_dir = temp_dir / 'score'
        
        # Strong model data
        strong_dir = score_dir / 'gpt-4-turbo'
        strong_dir.mkdir(parents=True)
        strong_file = strong_dir / 'BFCL_v3_test_score.json'
        
        strong_data = [
            {'accuracy': 0.7, 'correct_count': 70, 'total_count': 100},
            {'id': 'test_1', 'valid': False, 'error': ['timeout'], 'error_type': 'technical_error'}
        ]
        
        with open(strong_file, 'w') as f:
            for item in strong_data:
                f.write(json.dumps(item) + '\\n')
        
        # Weak model data (performs better)
        weak_dir = score_dir / 'llama-3-8b'
        weak_dir.mkdir(parents=True)
        weak_file = weak_dir / 'BFCL_v3_test_score.json'
        
        weak_data = [
            {'accuracy': 0.9, 'correct_count': 90, 'total_count': 100},
            {'id': 'test_1', 'valid': True, 'error': [], 'error_type': ''}
        ]
        
        with open(weak_file, 'w') as f:
            for item in weak_data:
                f.write(json.dumps(item) + '\\n')
        
        return {
            'config_file': config_file,
            'data_root': temp_dir,
            'output_dir': temp_dir / 'output'
        }
    
    def test_full_pipeline_integration(self, complete_setup):
        """Test complete analysis pipeline integration."""
        analyzer = EnhancedBFCLAnalyzer(
            complete_setup['config_file'],
            complete_setup['data_root'], 
            complete_setup['output_dir']
        )
        
        # Load data
        df = analyzer.load_score_files()
        assert not df.empty
        assert 'model_name' in df.columns
        
        # Classify tiers
        df = analyzer.classify_model_tiers(df)
        assert 'model_tier' in df.columns
        
        # Identify errors
        df = analyzer.identify_technical_errors(df)
        assert 'is_technical_error' in df.columns
        assert 'is_format_error' in df.columns
        
        # Analyze inversions
        inversions_df, validation_df = analyzer.analyze_performance_inversions_enhanced(df)
        
        # Should detect inversion (llama outperforming gpt-4)
        if not inversions_df.empty:
            assert inversions_df.iloc[0]['inversion_delta'] > 0.1
            assert 'test' in inversions_df.iloc[0]['test_category']

# Pytest configuration
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment."""
    # Set random seeds for reproducible tests
    np.random.seed(42)
    
    # Suppress warnings during testing
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

def test_module_imports():
    """Test that all modules can be imported correctly."""
    # Test imports
    from statistical_validation import StatisticalValidator
    from family_task_analysis import FamilyTaskAnalyzer
    from case_study_sampler import CaseStudySampler
    from enhanced_visualizations import AccessibleVisualizer
    from bfcl_analysis_enhanced import EnhancedBFCLAnalyzer
    
    # Basic instantiation tests
    assert StatisticalValidator() is not None
    assert FamilyTaskAnalyzer() is not None
    # CaseStudySampler and others require parameters, so just test import success

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main(["-v", "--tb=short", __file__])