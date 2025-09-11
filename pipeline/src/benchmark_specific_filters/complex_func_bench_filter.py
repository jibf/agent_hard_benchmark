"""
ComplexFuncBench-specific rule-based filtering.
Implements custom filtering logic for ComplexFuncBench evaluation data.
"""

from typing import Dict, List, Tuple
from .base_filter import BaseBenchmarkFilter
import logging
from src.bench_loaders.complex_func_bench_loader import ComplexFuncBenchLoader
from src.utils.types import ComplexFuncBenchQuestion

logger = logging.getLogger(__name__)

class ComplexFuncBenchFilter(BaseBenchmarkFilter):
    """ComplexFuncBench-specific filtering rules."""
    
    def __init__(self):
        super().__init__("ComplexFuncBench")
    
    def get_filter_name(self) -> str:
        return "ComplexFuncBench-Specific Filter"
    
    def is_applicable(self, sample: Dict) -> bool:
        """Check if sample is from ComplexFuncBench."""
        return (
            'task_name' in sample and 
            any(task_type in sample['task_name'] for task_type in [
                'attraction', 'flights', 'hotels', 'car-rental', 'cross'
            ])
        )
    
    def filter_samples(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Apply ComplexFuncBench-specific filtering rules.
        Note: Comprehensive filtering has already been applied before this method is called.
        """
        logger.info(f"Applying ComplexFuncBench-specific filtering to {len(samples)} samples")
        # ['model_path', 'benchmark_name', 'sampling_params', 'messages', 'eval_result', 'user_model_path', 'task_name', 'user_sampling_params', 'meta', 'model_name'

        # Step 1: Filter out questions with invalid function calls
        qids_with_invalid_function_calls = self._get_qids_with_invalid_function_calls()
        passed_samples = []
        dropped_samples = []
        
        for sample in samples:
            if sample['meta']['id'] not in qids_with_invalid_function_calls:
                passed_samples.append(sample)
            else:
                dropped_samples.append(sample)
        
        logger.info(f"ComplexFuncBench filtering completed: {len(passed_samples)} passed, {len(dropped_samples)} dropped")
        return passed_samples, dropped_samples


    def _get_qids_with_invalid_function_calls(self) -> List[ComplexFuncBenchQuestion]:
        loader = ComplexFuncBenchLoader()
        questions = loader.load_questions()

        result = []
        for question in questions:
            for message in question.gt_conv_traj:
                if message['role'] == 'assistant' and 'function_call' in message.keys():
                    for function_call in message['function_call']:
                        schema = self._get_function_schema(function_call['name'], question.available_function_list)
                        if not self._validate_function_call(function_call, schema):
                            result.append(question.question_id)
                            break
        return result

    def _validate_function_call(self, function_call: Dict, schema: Dict) -> bool:
        """
        Checks:
        1. Required arguments are present
        2. Argument types match schema types
        """
        arguments = function_call.get('arguments', {})
        parameters = schema.get('parameters', {})
        properties = parameters.get('properties', {})
        required = parameters.get('required', [])
        
        for req_arg in required:
            if req_arg not in arguments:
                return False
        
        for arg_name, arg_value in arguments.items():
            if arg_name in properties:
                expected_type = properties[arg_name].get('type')
                if not self._check_type(arg_value, expected_type):
                    return False
        
        return True
    
    def _check_type(self, value, expected_type: str) -> bool:
        """Check if value matches expected JSON schema type."""
        if expected_type == 'string':
            return isinstance(value, str)
        elif expected_type == 'number':
            return isinstance(value, (int, float))
        elif expected_type == 'integer':
            return isinstance(value, int)
        elif expected_type == 'boolean':
            return isinstance(value, bool)
        elif expected_type == 'array':
            return isinstance(value, list)
        elif expected_type == 'object':
            return isinstance(value, dict)
        return True


    def _get_function_schema(self, func_name: str, available_function_list: List):
        for available_function in available_function_list:
            if available_function["name"] == func_name:
                return available_function
        raise ValueError(f"Function {func_name} does not exist in available_function_list")


