import json
import os
import sys
from typing import Dict, Any, List, Optional

# Add the src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.types import AceBenchQuestion, Benchmark
from .base_loader import BaseLoader


class AceBenchLoader(BaseLoader):
    """Loader for ACEBench benchmark data."""
    
    def __init__(self):
        self.benchmark_name = "acebench"
        self.evaluation_dir = "benchmark/ACEBench-evaluation"
    
    def load_questions(self) -> List[AceBenchQuestion]:
        """Load all questions from ACEBench evaluation files."""
        questions = []
        
        if not os.path.exists(self.evaluation_dir):
            print(f"Warning: ACEBench evaluation directory {self.evaluation_dir} not found")
            return questions
        
        # Look for .jsonl files in the evaluation directory
        for filename in os.listdir(self.evaluation_dir):
            if filename.endswith('.jsonl'):
                file_path = os.path.join(self.evaluation_dir, filename)
                file_questions = self._load_questions_from_file(file_path)
                questions.extend(file_questions)
        
        print(f"Loaded {len(questions)} questions from ACEBench")
        return questions
    
    def _load_questions_from_file(self, file_path: str) -> List[AceBenchQuestion]:
        """Load questions from a single .jsonl file."""
        questions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        question = self._format_sample(sample, line_num)
                        if question:
                            questions.append(question)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON at line {line_num}: {e}")
                        continue
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
        
        return questions
    
    def _format_sample(self, sample: Dict[str, Any], line_num: int) -> AceBenchQuestion:
        """Format a raw sample into an AceBenchQuestion."""
        try:
            question_id = sample.get('meta', {}).get('id', f"acebench_{line_num}")
            task_name = sample.get('task_name', 'unknown')
            benchmark_name = sample.get('benchmark_name', 'acebench')
            
            instruction = self._extract_instruction(sample)
            available_function_list = self._extract_functions(sample)
            gt_conv_traj = self._extract_ground_truth(sample)
            
            meta = {
                'acebench_context': {
                    'task_name': task_name,
                    'benchmark_name': benchmark_name,
                    'model_path': sample.get('model_path', ''),
                    'sampling_params': sample.get('sampling_params', {}),
                    'eval_result': sample.get('eval_result', {}),
                    'source_file': sample.get('meta', {}).get('source_file', ''),
                    'acebench_result': sample.get('meta', {}).get('acebench_result', ''),
                    'is_correct': sample.get('meta', {}).get('is_correct', False),
                    'error_type': sample.get('meta', {}).get('error_type', ''),
                    'possible_answer': sample.get('meta', {}).get('possible_answer', ''),
                    'finish_reason': sample.get('meta', {}).get('finish_reason', ''),
                    'turn_idx': sample.get('messages', [{}])[0].get('turn_idx', 0) if sample.get('messages') else 0
                }
            }
            
            # Convert complex data to JSON strings for string fields
            acebench_result = sample.get('meta', {}).get('acebench_result', '')
            if not isinstance(acebench_result, str):
                acebench_result = json.dumps(acebench_result) if acebench_result else ''
            
            possible_answer = sample.get('meta', {}).get('possible_answer', '')
            if not isinstance(possible_answer, str):
                possible_answer = json.dumps(possible_answer) if possible_answer else ''
            
            return AceBenchQuestion(
                question_id=question_id,
                instruction=instruction,
                available_function_list=available_function_list,
                gt_conv_traj=gt_conv_traj,
                benchmark=Benchmark.ACE_BENCH,
                task_name=task_name,
                benchmark_name=benchmark_name,
                model_path=sample.get('model_path', ''),
                sampling_params=sample.get('sampling_params', {}),
                eval_result=sample.get('eval_result', {}),
                source_file=sample.get('meta', {}).get('source_file', ''),
                acebench_result=acebench_result,
                is_correct=sample.get('meta', {}).get('is_correct', False),
                error_type=sample.get('meta', {}).get('error_type', ''),
                possible_answer=possible_answer,
                finish_reason=sample.get('meta', {}).get('finish_reason', ''),
                turn_idx=sample.get('messages', [{}])[0].get('turn_idx', 0) if sample.get('messages') else 0,
                meta=meta
            )
        except Exception as e:
            print(f"Error formatting sample {line_num}: {e}")
            return None
    
    def _extract_instruction(self, sample: Dict[str, Any]) -> str:
        """Extract instruction from sample messages."""
        messages = sample.get('messages', [])
        if messages:
            for msg in messages:
                if msg.get('role') == 'user':
                    return msg.get('content', '')
                elif msg.get('role') == 'system':
                    return msg.get('content', '')
        
        # Fallback: create instruction based on task name and category
        task_name = sample.get('task_name', 'unknown')
        task_category = self._extract_task_category(task_name)
        return f"Complete the ACEBench {task_category} task: {task_name}"
    
    def _extract_task_category(self, task_name: str) -> str:
        """Extract task category from task name."""
        if task_name.startswith('normal_'):
            return 'Normal'
        elif task_name.startswith('special_'):
            return 'Special'
        elif task_name.startswith('agent_'):
            return 'Agent'
        else:
            return 'Unknown'
    
    def _extract_functions(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract available functions from sample."""
        # ACEBench typically doesn't have explicit function schemas in evaluation files
        # However, we can try to extract function calls from the assistant messages
        messages = sample.get('messages', [])
        functions = []
        
        for msg in messages:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                # Try to extract function calls from the content
                extracted_functions = self._extract_function_calls_from_content(content)
                if extracted_functions:
                    functions.extend(extracted_functions)
        
        return functions
    
    def _extract_function_calls_from_content(self, content: str) -> List[Dict[str, Any]]:
        """Extract function calls from assistant message content."""
        functions = []
        
        try:
            # Check if content contains function calls (usually in list format)
            if content.startswith('[') and content.endswith(']'):
                # Try to parse as JSON
                try:
                    function_calls = json.loads(content)
                    if isinstance(function_calls, list):
                        for call in function_calls:
                            if isinstance(call, str):
                                # Extract function name from string representation
                                func_name = self._extract_function_name(call)
                                if func_name:
                                    functions.append({
                                        'name': func_name,
                                        'description': f'Function call: {call}',
                                        'parameters': {}
                                    })
                except json.JSONDecodeError:
                    # If not valid JSON, try to extract function names manually
                    func_names = self._extract_function_names_manually(content)
                    for func_name in func_names:
                        functions.append({
                            'name': func_name,
                            'description': f'Function call: {func_name}',
                            'parameters': {}
                        })
        except Exception:
            pass
        
        return functions
    
    def _extract_function_name(self, call_str: str) -> str:
        """Extract function name from function call string."""
        # Look for function name pattern: function_name(...)
        import re
        match = re.search(r'(\w+)\s*\(', call_str)
        if match:
            return match.group(1)
        return ""
    
    def _extract_function_names_manually(self, content: str) -> List[str]:
        """Extract function names manually from content."""
        import re
        # Look for patterns like function_name(...)
        function_patterns = re.findall(r'(\w+)\s*\([^)]*\)', content)
        return list(set(function_patterns))  # Remove duplicates
    
    def _extract_ground_truth(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract ground truth conversation trajectory."""
        # ACEBench evaluation files contain model responses, not ground truth
        # However, we can extract the conversation structure for context
        messages = sample.get('messages', [])
        if messages:
            # Convert messages to conversation format
            conversation = []
            for msg in messages:
                if msg.get('role') == 'user':
                    conversation.append({
                        'role': 'user',
                        'content': msg.get('content', '')
                    })
                elif msg.get('role') == 'assistant':
                    conversation.append({
                        'role': 'assistant', 
                        'content': msg.get('content', '')
                    })
            return conversation
        
        return []