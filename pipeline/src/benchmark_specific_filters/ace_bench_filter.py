"""
ACEBench-specific rule-based filtering.
Implements custom filtering logic for ACEBench evaluation data.
Based on the detailed analysis of 4 main issue categories.
"""

from typing import Dict, List, Tuple
from .base_filter import BaseBenchmarkFilter
import logging
import re
import json
import numpy as np

logger = logging.getLogger(__name__)

class ACEBenchFilter(BaseBenchmarkFilter):
    """ACEBench-specific filtering rules."""
    
    def __init__(self):
        super().__init__("ACEBench")
    
    def get_filter_name(self) -> str:
        return "ACEBench-Specific Filter"
    
    def is_applicable(self, sample: Dict) -> bool:
        """Check if sample is from ACEBench."""
        # ACEBench samples typically have function calls, APIs, or tool usage
        return (
            'function_calls' in sample or
            'tool_calls' in sample or
            'api' in sample or
            'task' in sample and any(task_type in sample['task'] for task_type in [
                'normal', 'special', 'agent', 'atom', 'single_turn', 'multi_turn'
            ])
        )
    
    def filter_samples(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Apply ACEBench-specific filtering rules.
        
        ACEBench-specific rules based on 4 main issue categories:
        1. Parameter Value Error - Wrong, non-canonical, inconsistent values
        2. Incorrect Parameter Value - Contradicts context or gold labels
        3. Addition of Unnecessary Parameter - Hallucinated extra parameters
        4. Value Error - Semantically correct but formatted incorrectly
        
        Final step: Apply variance-based discriminativeness filtering
        """
        logger.info(f"Applying ACEBench-specific filtering to {len(samples)} samples")
        
        # Filter 1: Basic structure validation
        structure_valid = self._filter_by_structure(samples)
        logger.info(f"Structure validation: {len(structure_valid)} samples passed")
        
        # Filter 2: Parameter value error detection
        param_value_valid = self._filter_by_parameter_value_error(structure_valid)
        logger.info(f"Parameter value error check: {len(param_value_valid)} samples passed")
        
        # Filter 3: Incorrect parameter value detection
        incorrect_param_valid = self._filter_by_incorrect_parameter_value(param_value_valid)
        logger.info(f"Incorrect parameter value check: {len(incorrect_param_valid)} samples passed")
        
        # Filter 4: Unnecessary parameter detection
        unnecessary_param_valid = self._filter_by_unnecessary_parameter(incorrect_param_valid)
        logger.info(f"Unnecessary parameter check: {len(unnecessary_param_valid)} samples passed")
        
        # Filter 5: Value error detection
        value_error_valid = self._filter_by_value_error(unnecessary_param_valid)
        logger.info(f"Value error check: {len(value_error_valid)} samples passed")
        
        # Filter 6: Variance-based discriminativeness (from comprehensive filtering)
        discriminative_valid = self._filter_by_discriminativeness(value_error_valid)
        logger.info(f"Discriminativeness check: {len(discriminative_valid)} samples passed")
        
        dropped = [s for s in samples if s not in discriminative_valid]
        
        self.log_filtering_stats(len(samples), len(discriminative_valid), len(dropped))
        
        return discriminative_valid, dropped
    
    def _filter_by_structure(self, samples: List[Dict]) -> List[Dict]:
        """Filter by ACEBench-specific structure requirements."""
        valid_samples = []
        
        for sample in samples:
            if not self.is_applicable(sample):
                continue
            
            # Must have evaluation result
            if 'eval_result' not in sample:
                continue
            
            # Must have task information
            if 'task' not in sample:
                continue
            
            valid_samples.append(sample)
        
        return valid_samples
    
    def _filter_by_parameter_value_error(self, samples: List[Dict]) -> List[Dict]:
        """
        Filter out samples with parameter value error issues.
        
        Detects:
        - Wrong, non-canonical values
        - Inconsistent with schema
        - Spacing/formatting errors
        - Invalid enums
        """
        valid_samples = []
        
        for sample in samples:
            # Check for function calls with parameter value errors
            if self._has_parameter_value_errors(sample):
                continue
            
            valid_samples.append(sample)
        
        return valid_samples
    
    def _filter_by_incorrect_parameter_value(self, samples: List[Dict]) -> List[Dict]:
        """
        Filter out samples with incorrect parameter value issues.
        
        Detects:
        - Values that contradict context
        - Values that contradict gold labels
        - Semantic mismatches
        """
        valid_samples = []
        
        for sample in samples:
            # Check for incorrect parameter values
            if self._has_incorrect_parameter_values(sample):
                continue
            
            valid_samples.append(sample)
        
        return valid_samples
    
    def _filter_by_unnecessary_parameter(self, samples: List[Dict]) -> List[Dict]:
        """
        Filter out samples with unnecessary parameter issues.
        
        Detects:
        - Hallucinated extra parameters
        - Parameters outside schema
        - Semantically neutral additions
        """
        valid_samples = []
        
        for sample in samples:
            # Check for unnecessary parameters
            if self._has_unnecessary_parameters(sample):
                continue
            
            valid_samples.append(sample)
        
        return valid_samples
    
    def _filter_by_value_error(self, samples: List[Dict]) -> List[Dict]:
        """
        Filter out samples with value error issues.
        
        Detects:
        - Semantically correct but formatted incorrectly
        - Concatenated tokens
        - Malformed ranges
        - Inconsistent casing
        """
        valid_samples = []
        
        for sample in samples:
            # Check for value formatting errors
            if self._has_value_formatting_errors(sample):
                continue
            
            valid_samples.append(sample)
        
        return valid_samples
    
    def _filter_by_discriminativeness(self, samples: List[Dict]) -> List[Dict]:
        """
        Filter out non-discriminative questions based on score variance.
        This ensures questions help distinguish between different LLM capabilities.
        """
        # Group samples by question/task
        question_groups = {}
        for sample in samples:
            # Extract question identifier (could be task_name, id, etc.)
            question_id = self._extract_question_id(sample)
            if question_id not in question_groups:
                question_groups[question_id] = []
            question_groups[question_id].append(sample)
        
        # Filter discriminative questions
        discriminative_samples = []
        for question_id, question_samples in question_groups.items():
            if self._is_question_discriminative(question_samples):
                discriminative_samples.extend(question_samples)
        
        return discriminative_samples
    
    def _extract_question_id(self, sample: Dict) -> str:
        """Extract question identifier from sample."""
        # Try different possible question ID fields
        if 'meta' in sample and 'id' in sample['meta']:
            return sample['meta']['id']
        elif 'task_name' in sample:
            return sample['task_name']
        elif 'id' in sample:
            return sample['id']
        else:
            # Fallback: use task content hash as identifier
            task_content = str(sample.get('task_name', '')) + str(sample.get('instruction', ''))
            return str(hash(task_content))
    
    def _is_question_discriminative(self, question_samples: List[Dict]) -> bool:
        """
        Determine if a question is discriminative based on model performance variation.
        Returns True if the question helps distinguish between different LLM capabilities.
        """
        if len(question_samples) < 2:
            return False  # Need at least 2 model responses to compare
        
        # Extract scores for this question
        scores = []
        for sample in question_samples:
            # Try different possible score locations
            if 'eval_result' in sample and 'score' in sample['eval_result']:
                scores.append(sample['eval_result']['score'])
            elif 'eval_result' in sample and 'scores' in sample['eval_result']:
                scores.extend(sample['eval_result']['scores'])
            elif 'score' in sample:
                scores.append(sample['score'])
            elif 'scores' in sample:
                scores.extend(sample['scores'])
        
        if not scores:
            return False
        
        # Convert to numeric scores
        numeric_scores = []
        for score in scores:
            if isinstance(score, (int, float)):
                numeric_scores.append(float(score))
            elif isinstance(score, dict) and 'score' in score:
                try:
                    numeric_scores.append(float(score['score']))
                except (ValueError, TypeError):
                    continue
        
        if len(numeric_scores) < 2:
            return False
        
        # Calculate variance to measure discriminativeness
        variance = np.var(numeric_scores)
        
        # Question is discriminative if there's sufficient variance in scores
        # This means different models perform differently on this question
        return variance > 0.01  # Threshold for meaningful variation
    
    def _has_parameter_value_errors(self, sample: Dict) -> bool:
        """Check if sample has parameter value errors."""
        # Check for function calls
        function_calls = sample.get('function_calls', [])
        if not function_calls:
            return False
        
        for call in function_calls:
            if isinstance(call, dict):
                # Check for malformed parameter values
                if self._check_malformed_parameters(call):
                    return True
        
        return False
    
    def _check_malformed_parameters(self, function_call: Dict) -> bool:
        """Check if function call has malformed parameters."""
        # Check for common malformed patterns
        malformed_patterns = [
            r'\$?\d+monthly',  # $2000monthly
            r'\d+months?',     # 24months
            r'\d+days?',       # 30days
            r'\d+years?',      # 2years
            r'\d+hours?',      # 24hours
            r'\d+minutes?',    # 60minutes
            r'[A-Za-z]+\d+',   # text123
            r'\d+[A-Za-z]+',   # 123text
        ]
        
        # Check function name
        function_name = function_call.get('name', '')
        if any(re.search(pattern, function_name, re.IGNORECASE) for pattern in malformed_patterns):
            return True
        
        # Check arguments
        arguments = function_call.get('arguments', {})
        if isinstance(arguments, dict):
            for param_name, param_value in arguments.items():
                param_str = str(param_value)
                
                # Check for malformed parameter names
                if any(re.search(pattern, param_name, re.IGNORECASE) for pattern in malformed_patterns):
                    return True
                
                # Check for malformed parameter values
                if any(re.search(pattern, param_str, re.IGNORECASE) for pattern in malformed_patterns):
                    return True
        
        return False
    
    def _has_incorrect_parameter_values(self, sample: Dict) -> bool:
        """Check if sample has incorrect parameter values."""
        # Check for function calls with incorrect values
        function_calls = sample.get('function_calls', [])
        if not function_calls:
            return False
        
        for call in function_calls:
            if isinstance(call, dict):
                # Check for values that contradict context
                if self._check_contradictory_values(call, sample):
                    return True
        
        return False
    
    def _check_contradictory_values(self, function_call: Dict, sample: Dict) -> bool:
        """Check if function call has values that contradict context."""
        # Check for date contradictions
        arguments = function_call.get('arguments', {})
        if isinstance(arguments, dict):
            dates = []
            for param_name, param_value in arguments.items():
                param_str = str(param_value)
                # Extract dates in various formats
                date_patterns = [
                    r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                    r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                    r'\d{1,2}/\d{1,2}/\d{2,4}'  # M/D/YY or M/D/YYYY
                ]
                
                for pattern in date_patterns:
                    found_dates = re.findall(pattern, param_str)
                    dates.extend(found_dates)
            
            # Check for chronological contradictions
            if len(dates) > 1:
                # Simple check: if dates are not in chronological order
                try:
                    sorted_dates = sorted(dates)
                    if dates != sorted_dates:
                        return True
                except:
                    pass
        
        return False
    
    def _has_unnecessary_parameters(self, sample: Dict) -> bool:
        """Check if sample has unnecessary parameters."""
        # Check for function calls with extra parameters
        function_calls = sample.get('function_calls', [])
        if not function_calls:
            return False
        
        for call in function_calls:
            if isinstance(call, dict):
                # Check for parameters outside expected schema
                if self._check_extra_parameters(call, sample):
                    return True
        
        return False
    
    def _check_extra_parameters(self, function_call: Dict, sample: Dict) -> bool:
        """Check if function call has extra parameters outside schema."""
        # This would require schema information to properly validate
        # For now, use heuristics based on common patterns
        
        arguments = function_call.get('arguments', {})
        if not isinstance(arguments, dict):
            return False
        
        # Check for common unnecessary parameter patterns
        unnecessary_patterns = [
            r'provide_.*_options',
            r'include_.*_details',
            r'add_.*_information',
            r'supply_.*_context',
            r'give_.*_examples'
        ]
        
        for param_name in arguments.keys():
            if any(re.search(pattern, param_name, re.IGNORECASE) for pattern in unnecessary_patterns):
                return True
        
        return False
    
    def _has_value_formatting_errors(self, sample: Dict) -> bool:
        """Check if sample has value formatting errors."""
        # Check for function calls with formatting issues
        function_calls = sample.get('function_calls', [])
        if not function_calls:
            return False
        
        for call in function_calls:
            if isinstance(call, dict):
                # Check for formatting errors
                if self._check_formatting_errors(call):
                    return True
        
        return False
    
    def _check_formatting_errors(self, function_call: Dict) -> bool:
        """Check if function call has formatting errors."""
        # Check for common formatting issues
        formatting_patterns = [
            r'[A-Z][a-z]*[A-Z][a-z]*',  # Inconsistent casing
            r'\d+[A-Za-z]+\d+',          # Mixed alphanumeric
            r'[A-Za-z]+\d+[A-Za-z]+',    # Mixed alphanumeric
            r'\s+',                       # Extra whitespace
            r'[^\w\s\-_\.]',             # Special characters
        ]
        
        # Check function name
        function_name = function_call.get('name', '')
        if any(re.search(pattern, function_name) for pattern in formatting_patterns):
            return True
        
        # Check arguments
        arguments = function_call.get('arguments', {})
        if isinstance(arguments, dict):
            for param_name, param_value in arguments.items():
                param_str = str(param_value)
                
                # Check for formatting issues in parameter names
                if any(re.search(pattern, param_name) for pattern in formatting_patterns):
                    return True
                
                # Check for formatting issues in parameter values
                if any(re.search(pattern, param_str) for pattern in formatting_patterns):
                    return True
        
        return False
    
    def _extract_task_category(self, sample: Dict) -> str:
        """Extract task category from sample."""
        task = sample.get('task', '')
        if 'normal' in task.lower():
            return 'normal'
        elif 'special' in task.lower():
            return 'special'
        elif 'agent' in task.lower():
            return 'agent'
        elif 'atom' in task.lower():
            return 'atom'
        elif 'single_turn' in task.lower():
            return 'single_turn'
        elif 'multi_turn' in task.lower():
            return 'multi_turn'
        else:
            return 'unknown'
