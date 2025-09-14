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
        # Check if this is an ACEBench sample based on the actual data structure
        benchmark_name = sample.get('benchmark_name', '')
        if hasattr(benchmark_name, 'value'):
            benchmark_name = benchmark_name.value
        benchmark_name = str(benchmark_name).lower()
        if benchmark_name == 'acebench':
            return True
        
        # Also check for ACEBench-specific patterns in the data
        task_name = sample.get('task_name', '')
        if task_name and any(task_type in task_name for task_type in [
            'normal', 'special', 'agent', 'atom', 'single_turn', 'multi_turn'
        ]):
            return True
        
        # Check for ACEBench-specific message structure with function calls
        messages = sample.get('messages', [])
        if messages and len(messages) >= 2:
            # Look for assistant message with function call pattern
            for msg in messages:
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    # ACEBench function calls are in format [FunctionName(params)]
                    if content and content.startswith('[') and content.endswith(']'):
                        return True
        
        return False
    
    def filter_samples(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Apply ACEBench-specific filtering rules at the QUESTION LEVEL.
        
        ACEBench-specific rules based on 4 main issue categories:
        1. Parameter Value Error - Wrong, non-canonical, inconsistent values
        2. Incorrect Parameter Value - Contradicts context or gold labels
        3. Addition of Unnecessary Parameter - Hallucinated extra parameters
        4. Value Error - Semantically correct but formatted incorrectly
        
        This filter works at the question level: if a question passes all checks,
        ALL samples for that question are kept.
        """
        logger.info(f"Applying ACEBench-specific filtering to {len(samples)} samples")
        
        # Group samples by question
        question_groups = self._group_samples_by_question(samples)
        logger.info(f"Grouped into {len(question_groups)} unique questions")
        
        # Apply filtering at question level
        passed_questions = []
        dropped_questions = []
        
        for question_id, question_samples in question_groups.items():
            if self._is_question_valid(question_samples):
                passed_questions.extend(question_samples)
            else:
                dropped_questions.extend(question_samples)
        
        logger.info(f"Questions passed: {len(passed_questions) // len(question_groups) if question_groups else 0} (all samples kept)")
        logger.info(f"Questions dropped: {len(dropped_questions) // len(question_groups) if question_groups else 0} (all samples dropped)")
        
        self.log_filtering_stats(len(samples), len(passed_questions), len(dropped_questions))
        
        return passed_questions, dropped_questions
    
    def _group_samples_by_question(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """Group samples by their question/task identifier."""
        question_groups = {}
        for sample in samples:
            question_id = self._extract_question_id(sample)
            if question_id not in question_groups:
                question_groups[question_id] = []
            question_groups[question_id].append(sample)
        return question_groups
    
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
    
    def _is_question_valid(self, question_samples: List[Dict]) -> bool:
        """
        Check if a question is valid by applying all ACEBench-specific filters at the QUESTION LEVEL.
        Returns True if the question passes all checks, False otherwise.
        
        This evaluates the question as a whole, not individual responses.
        """
        # Check if any sample in the question is applicable to ACEBench
        if not any(self.is_applicable(sample) for sample in question_samples):
            return False
        
        # Get a representative sample to evaluate the question
        # Use the first applicable sample to represent the question
        representative_sample = None
        for sample in question_samples:
            if self.is_applicable(sample):
                representative_sample = sample
                break
        
        if not representative_sample:
            return False
        
        # Apply all filtering checks to the QUESTION (using representative sample)
        # If the question has these issues, reject the entire question
        if (self._has_parameter_value_errors(representative_sample) or
            self._has_incorrect_parameter_values(representative_sample) or
            self._has_unnecessary_parameters(representative_sample) or
            self._has_value_formatting_errors(representative_sample)):
            return False
        
        # Question passes all checks
        return True
    
    def _filter_by_structure(self, samples: List[Dict]) -> List[Dict]:
        """Filter by ACEBench-specific structure requirements."""
        valid_samples = []
        
        for sample in samples:
            if not self.is_applicable(sample):
                continue
            
            # Must have evaluation result
            if 'eval_result' not in sample:
                continue
            
            # Must have task information (task_name in ACEBench)
            if 'task_name' not in sample:
                continue
            
            # Must have messages with function calls
            messages = sample.get('messages', [])
            if not messages or len(messages) < 2:
                continue
            
            # Must have assistant message with function call
            has_function_call = False
            for msg in messages:
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    if content and content.startswith('[') and content.endswith(']'):
                        has_function_call = True
                        break
            
            if not has_function_call:
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
    
    
    def _has_parameter_value_errors(self, sample: Dict) -> bool:
        """Check if sample has parameter value errors."""
        # Extract function call from ACEBench message format
        function_call_content = self._extract_function_call_content(sample)
        if not function_call_content:
            return False
        
        # Check for malformed parameter values in the function call string
        if self._check_malformed_parameters_string(function_call_content):
            return True
        
        return False
    
    def _extract_function_call_content(self, sample: Dict) -> str:
        """Extract function call content from ACEBench message format."""
        messages = sample.get('messages', [])
        for msg in messages:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                # Handle both string and list content
                if isinstance(content, list):
                    content = str(content)
                if content and content.startswith('[') and content.endswith(']'):
                    return content
        return ""
    
    def _check_malformed_parameters_string(self, function_call_content: str) -> bool:
        """Check if function call string has malformed parameters."""
        # Check for common malformed patterns in the function call string
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
        
        # Check for malformed patterns in the entire function call string
        if any(re.search(pattern, function_call_content, re.IGNORECASE) for pattern in malformed_patterns):
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
        # Extract function call from ACEBench message format
        function_call_content = self._extract_function_call_content(sample)
        if not function_call_content:
            return False
        
        # Check for values that contradict context
        if self._check_contradictory_values_string(function_call_content, sample):
            return True
        
        return False
    
    def _check_contradictory_values_string(self, function_call_content: str, sample: Dict) -> bool:
        """Check if function call string has values that contradict context."""
        # Check for date contradictions in the function call string
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{1,2}/\d{1,2}/\d{2,4}'  # M/D/YY or M/D/YYYY
        ]
        
        dates = []
        for pattern in date_patterns:
            found_dates = re.findall(pattern, function_call_content)
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
        # Extract function call from ACEBench message format
        function_call_content = self._extract_function_call_content(sample)
        if not function_call_content:
            return False
        
        # Check for parameters outside expected schema
        if self._check_extra_parameters_string(function_call_content, sample):
            return True
        
        return False
    
    def _check_extra_parameters_string(self, function_call_content: str, sample: Dict) -> bool:
        """Check if function call string has extra parameters outside schema."""
        # Check for common unnecessary parameter patterns in the function call string
        unnecessary_patterns = [
            r'provide_.*_options',
            r'include_.*_details',
            r'add_.*_information',
            r'supply_.*_context',
            r'give_.*_examples'
        ]
        
        # Check for unnecessary parameter patterns in the function call string
        if any(re.search(pattern, function_call_content, re.IGNORECASE) for pattern in unnecessary_patterns):
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
        # Extract function call from ACEBench message format
        function_call_content = self._extract_function_call_content(sample)
        if not function_call_content:
            return False
        
        # Check for formatting errors
        if self._check_formatting_errors_string(function_call_content):
            return True
        
        return False
    
    def _check_formatting_errors_string(self, function_call_content: str) -> bool:
        """Check if function call string has formatting errors."""
        # For ACEBench, we need to be more lenient since function calls have specific formatting
        # Only check for truly problematic formatting issues
        
        # Check for concatenated tokens without spaces (like "text123" or "123text")
        concatenated_patterns = [
            r'[A-Za-z]+\d+[A-Za-z]+',    # Mixed alphanumeric like "text123text"
            r'\d+[A-Za-z]+\d+',          # Mixed alphanumeric like "123text456"
        ]
        
        # Check for malformed function names (should be PascalCase)
        function_name_match = re.match(r'\[([A-Za-z_][A-Za-z0-9_]*)\(', function_call_content)
        if function_name_match:
            function_name = function_name_match.group(1)
            # Function names should be PascalCase, reject if they have numbers in the middle
            if re.search(r'[A-Za-z]\d+[A-Za-z]', function_name):
                return True
        
        # Check for concatenated patterns in the entire string
        if any(re.search(pattern, function_call_content) for pattern in concatenated_patterns):
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
