import json
import os
import re
from typing import Dict, List, Tuple, Any
from collections import defaultdict

class PreciseTypeAnalyzer:
    def __init__(self):
        # Focus on actual type mismatches in function calls
        self.type_patterns = {
            # Boolean values as strings in function parameters
            'bool_as_string': [
                r'(\w+)=(["\'])(?:true|false|True|False)\2',
                r':\s*(["\'])(?:true|false|True|False)\1'
            ],
            # Numeric values as strings in function parameters  
            'number_as_string': [
                r'(\w+)=(["\'])(\d+(?:\.\d+)?)\2',
                r':\s*(["\'])\d+(?:\.\d+)?\1'
            ],
            # None/null as strings
            'null_as_string': [
                r'(["\'])(?:null|None|NULL)\1',
                r'(\w+)=(["\'])(?:null|None|NULL)\2'
            ]
        }
        
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single JSON file for actual type mismatches."""
        results = {
            'total_entries': 0,
            'function_calls_analyzed': 0,
            'type_errors': defaultdict(int),
            'examples': defaultdict(list),
            'file_path': file_path
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    results['total_entries'] += 1
                    
                    try:
                        data = json.loads(line.strip())
                        if 'result' in data and data['result']:
                            result_str = data['result']
                            
                            # Check if this looks like a function call
                            if self._is_function_call(result_str):
                                results['function_calls_analyzed'] += 1
                                self._analyze_function_call(result_str, results, line_num, data.get('id', 'unknown'))
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        continue
                        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
        return results
    
    def _is_function_call(self, text: str) -> bool:
        """Check if text contains function call patterns."""
        # Look for patterns like [function_name(param=value)] or function_name(param=value)
        function_pattern = r'[\[\s]*\w+\([^)]*\)'
        return bool(re.search(function_pattern, text))
    
    def _analyze_function_call(self, function_call: str, results: Dict, line_num: int, entry_id: str):
        """Analyze a function call for type mismatches."""
        
        # Boolean parameters that should be boolean but are strings
        bool_pattern = r'(\w+)\s*=\s*(["\'])(?:true|false|True|False)\2'
        bool_matches = re.findall(bool_pattern, function_call)
        for param_name, quote in bool_matches:
            results['type_errors']['bool_as_string'] += 1
            if len(results['examples']['bool_as_string']) < 10:
                example_match = re.search(rf'{param_name}\s*=\s*["\'][^"\']*["\']', function_call)
                if example_match:
                    results['examples']['bool_as_string'].append({
                        'entry_id': entry_id,
                        'line': line_num,
                        'param': param_name,
                        'value': example_match.group(),
                        'context': function_call[:200] + '...' if len(function_call) > 200 else function_call
                    })
        
        # Numeric parameters that should be numbers but are strings
        num_pattern = r'(\w+)\s*=\s*(["\'])(\d+(?:\.\d+)?)\2'
        num_matches = re.findall(num_pattern, function_call)
        for param_name, quote, number in num_matches:
            # Skip parameters that are likely meant to be strings (like IDs, passwords)
            if not self._should_be_string_param(param_name):
                results['type_errors']['number_as_string'] += 1
                if len(results['examples']['number_as_string']) < 10:
                    results['examples']['number_as_string'].append({
                        'entry_id': entry_id,
                        'line': line_num,
                        'param': param_name,
                        'value': f'{param_name}={quote}{number}{quote}',
                        'suggested': f'{param_name}={number}',
                        'context': function_call[:200] + '...' if len(function_call) > 200 else function_call
                    })
        
        # None/null as strings
        null_pattern = r'(\w+)\s*=\s*(["\'])(?:null|None|NULL)\2'
        null_matches = re.findall(null_pattern, function_call)
        for param_name, quote in null_matches:
            results['type_errors']['null_as_string'] += 1
            if len(results['examples']['null_as_string']) < 10:
                results['examples']['null_as_string'].append({
                    'entry_id': entry_id,
                    'line': line_num,
                    'param': param_name,
                    'value': f'{param_name}={quote}None{quote}',
                    'suggested': f'{param_name}=None',
                    'context': function_call[:200] + '...' if len(function_call) > 200 else function_call
                })
                
        # Array representation issues (simple check)
        if '[' in function_call and '"[' in function_call:
            results['type_errors']['array_as_string'] += 1
            if len(results['examples']['array_as_string']) < 5:
                array_match = re.search(r'(\w+)\s*=\s*"[^"]*\[', function_call)
                if array_match:
                    results['examples']['array_as_string'].append({
                        'entry_id': entry_id,
                        'line': line_num,
                        'context': function_call[:300] + '...' if len(function_call) > 300 else function_call
                    })
    
    def _should_be_string_param(self, param_name: str) -> bool:
        """Determine if a parameter should likely be a string based on its name."""
        string_param_patterns = [
            'id', 'uuid', 'token', 'key', 'password', 'username', 'email',
            'name', 'title', 'description', 'text', 'message', 'url', 'path',
            'address', 'phone', 'zip', 'code', 'hash', 'session'
        ]
        param_lower = param_name.lower()
        return any(pattern in param_lower for pattern in string_param_patterns)
    
    def compare_files(self, original_path: str, fixed_path: str) -> Dict[str, Any]:
        """Compare original and fixed files."""
        original_results = self.analyze_file(original_path)
        fixed_results = self.analyze_file(fixed_path)
        
        comparison = {
            'original': original_results,
            'fixed': fixed_results,
            'improvements': {}
        }
        
        # Calculate improvements
        for error_type in original_results['type_errors']:
            original_count = original_results['type_errors'][error_type]
            fixed_count = fixed_results['type_errors'].get(error_type, 0)
            improvement = original_count - fixed_count
            improvement_percentage = (improvement / original_count * 100) if original_count > 0 else 0
            
            comparison['improvements'][error_type] = {
                'original_count': original_count,
                'fixed_count': fixed_count,
                'improvement': improvement,
                'improvement_percentage': improvement_percentage
            }
        
        return comparison

def main():
    analyzer = PreciseTypeAnalyzer()
    
    base_paths = [
        "E:/Users/김현준/Downloads/agent_hard_benchmark_2/gorilla/berkeley-function-call-leaderboard/result/anthropic_claude-4-sonnet-thinking-off",
        "E:/Users/김현준/Downloads/agent_hard_benchmark_2/gorilla/berkeley-function-call-leaderboard/result/anthropic_claude-4-sonnet-thinking-on-10k"
    ]
    
    all_results = {}
    
    for base_path in base_paths:
        model_name = os.path.basename(base_path)
        all_results[model_name] = {}
        
        print(f"\nAnalyzing {model_name}...")
        print("-" * 60)
        
        # Focus on key files for detailed analysis
        key_files = [
            'BFCL_v3_simple_result.json',
            'BFCL_v3_multiple_result.json',
            'BFCL_v3_parallel_result.json',
            'BFCL_v3_multi_turn_base_result.json',
            'BFCL_v3_live_simple_result.json'
        ]
        
        for filename in key_files:
            file_path = os.path.join(base_path, filename)
            if os.path.exists(file_path):
                results = analyzer.analyze_file(file_path)
                all_results[model_name][filename] = results
                
                total_errors = sum(results['type_errors'].values())
                if results['function_calls_analyzed'] > 0:
                    error_rate = (total_errors / results['function_calls_analyzed']) * 100
                    print(f"  {filename}:")
                    print(f"    Function calls analyzed: {results['function_calls_analyzed']}")
                    print(f"    Type errors found: {total_errors}")
                    print(f"    Error rate: {error_rate:.2f}%")
                    
                    if total_errors > 0:
                        for error_type, count in results['type_errors'].items():
                            if count > 0:
                                print(f"      {error_type}: {count}")
    
    # Compare with fixed versions
    print("\n" + "="*80)
    print("COMPARISON WITH FIXED VERSIONS")
    print("="*80)
    
    for base_path in base_paths:
        model_name = os.path.basename(base_path)
        fixed_path = os.path.join(base_path, "fixed")
        
        if os.path.exists(fixed_path):
            print(f"\n{model_name.upper()}:")
            print("-" * 60)
            
            for filename in os.listdir(fixed_path):
                if filename.endswith('_fixed.json'):
                    original_name = filename.replace('_fixed', '')
                    original_file = os.path.join(base_path, original_name)
                    fixed_file = os.path.join(fixed_path, filename)
                    
                    if os.path.exists(original_file):
                        comparison = analyzer.compare_files(original_file, fixed_file)
                        
                        original_total = sum(comparison['original']['type_errors'].values())
                        fixed_total = sum(comparison['fixed']['type_errors'].values())
                        
                        if original_total > 0 or fixed_total > 0:
                            print(f"\n  {original_name}:")
                            print(f"    Original errors: {original_total}")
                            print(f"    Fixed errors: {fixed_total}")
                            
                            if original_total > 0:
                                overall_improvement = ((original_total - fixed_total) / original_total) * 100
                                print(f"    Overall improvement: {overall_improvement:.1f}%")
                                
                                for error_type, improvement in comparison['improvements'].items():
                                    if improvement['original_count'] > 0:
                                        print(f"      {error_type}: {improvement['original_count']} → {improvement['fixed_count']} ({improvement['improvement_percentage']:.1f}% improvement)")
    
    # Show detailed examples
    print("\n" + "="*80)
    print("DETAILED EXAMPLES OF TYPE MISMATCHES")
    print("="*80)
    
    for model_name, files in all_results.items():
        print(f"\n{model_name.upper()}:")
        
        for filename, file_data in files.items():
            if sum(file_data['type_errors'].values()) > 0:
                print(f"\n  {filename}:")
                
                for error_type, examples in file_data['examples'].items():
                    if examples:
                        print(f"    {error_type.upper()} Examples:")
                        for i, example in enumerate(examples[:3], 1):  # Show top 3 examples
                            print(f"      {i}. Entry ID: {example['entry_id']}")
                            if 'param' in example:
                                print(f"         Parameter: {example['param']}")
                                print(f"         Found: {example['value']}")
                                if 'suggested' in example:
                                    print(f"         Should be: {example['suggested']}")
                            print(f"         Context: {example['context'][:150]}...")
                            print()

if __name__ == "__main__":
    main()