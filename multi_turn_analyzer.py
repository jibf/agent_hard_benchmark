import json
import os
import re
from typing import Dict, List, Tuple, Any
from collections import defaultdict

class MultiTurnAnalyzer:
    def __init__(self):
        pass
    
    def sample_analyze_large_file(self, file_path: str, sample_size: int = 100) -> Dict[str, Any]:
        """Analyze a large file by sampling entries."""
        results = {
            'total_entries': 0,
            'sampled_entries': 0,
            'function_calls_analyzed': 0,
            'type_errors': defaultdict(int),
            'examples': defaultdict(list),
            'file_path': file_path
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                results['total_entries'] = len([l for l in lines if l.strip()])
                
                # Sample every nth line to get representative sample
                step = max(1, len(lines) // sample_size)
                sampled_lines = lines[::step][:sample_size]
                
                for line_num, line in enumerate(sampled_lines):
                    if not line.strip():
                        continue
                    
                    results['sampled_entries'] += 1
                    
                    try:
                        data = json.loads(line.strip())
                        if 'result' in data and data['result']:
                            result_str = data['result']
                            
                            # Check if this looks like function calls
                            if self._contains_function_calls(result_str):
                                results['function_calls_analyzed'] += 1
                                self._analyze_function_calls(result_str, results, line_num, data.get('id', 'unknown'))
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        continue
                        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
        return results
    
    def _contains_function_calls(self, text: str) -> bool:
        """Check if text contains function call patterns."""
        # Look for patterns like [function_name(param=value)] or function_name(param=value)
        patterns = [
            r'\[[\w\.]+\([^)]*\)',  # [function(params)]
            r'[\w\.]+\([^)]*\)',    # function(params)
        ]
        return any(re.search(pattern, text) for pattern in patterns)
    
    def _analyze_function_calls(self, text: str, results: Dict, line_num: int, entry_id: str):
        """Analyze function calls for type mismatches."""
        
        # Boolean parameters that should be boolean but are strings
        bool_patterns = [
            r'(\w+)\s*=\s*(["\'])(true|false|True|False)\2',
            r'(\w+)\s*:\s*(["\'])(true|false|True|False)\2'
        ]
        
        for pattern in bool_patterns:
            matches = re.findall(pattern, text)
            for param_name, quote, bool_val in matches:
                results['type_errors']['bool_as_string'] += 1
                if len(results['examples']['bool_as_string']) < 15:
                    results['examples']['bool_as_string'].append({
                        'entry_id': entry_id,
                        'line': line_num,
                        'param': param_name,
                        'value': f'{param_name}={quote}{bool_val}{quote}',
                        'suggested': f'{param_name}={bool_val.lower()}',
                        'context': text[:300] + '...' if len(text) > 300 else text
                    })
        
        # Numeric parameters that should be numbers but are strings
        num_patterns = [
            r'(\w+)\s*=\s*(["\'])(\d+(?:\.\d+)?)\2',
            r'(\w+)\s*:\s*(["\'])(\d+(?:\.\d+)?)\2'
        ]
        
        for pattern in num_patterns:
            matches = re.findall(pattern, text)
            for param_name, quote, number in matches:
                # Skip parameters that should be strings
                if not self._should_be_string_param(param_name):
                    results['type_errors']['number_as_string'] += 1
                    if len(results['examples']['number_as_string']) < 15:
                        results['examples']['number_as_string'].append({
                            'entry_id': entry_id,
                            'line': line_num,
                            'param': param_name,
                            'value': f'{param_name}={quote}{number}{quote}',
                            'suggested': f'{param_name}={number}',
                            'context': text[:300] + '...' if len(text) > 300 else text
                        })
        
        # None/null as strings
        null_patterns = [
            r'(\w+)\s*=\s*(["\'])(null|None|NULL)\2',
            r'(\w+)\s*:\s*(["\'])(null|None|NULL)\2'
        ]
        
        for pattern in null_patterns:
            matches = re.findall(pattern, text)
            for param_name, quote, null_val in matches:
                results['type_errors']['null_as_string'] += 1
                if len(results['examples']['null_as_string']) < 10:
                    results['examples']['null_as_string'].append({
                        'entry_id': entry_id,
                        'line': line_num,
                        'param': param_name,
                        'value': f'{param_name}={quote}{null_val}{quote}',
                        'suggested': f'{param_name}=None',
                        'context': text[:300] + '...' if len(text) > 300 else text
                    })
    
    def _should_be_string_param(self, param_name: str) -> bool:
        """Determine if a parameter should likely be a string based on its name."""
        string_param_patterns = [
            'id', 'uuid', 'token', 'key', 'password', 'username', 'email',
            'name', 'title', 'description', 'text', 'message', 'url', 'path',
            'address', 'phone', 'zip', 'code', 'hash', 'session', 'query',
            'content', 'command', 'file', 'dir'
        ]
        param_lower = param_name.lower()
        return any(pattern in param_lower for pattern in string_param_patterns)
    
    def compare_files_sampled(self, original_path: str, fixed_path: str, sample_size: int = 100) -> Dict[str, Any]:
        """Compare original and fixed files using sampling."""
        original_results = self.sample_analyze_large_file(original_path, sample_size)
        fixed_results = self.sample_analyze_large_file(fixed_path, sample_size)
        
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
    analyzer = MultiTurnAnalyzer()
    
    base_paths = [
        "E:/Users/김현준/Downloads/agent_hard_benchmark_2/gorilla/berkeley-function-call-leaderboard/result/anthropic_claude-4-sonnet-thinking-off",
        "E:/Users/김현준/Downloads/agent_hard_benchmark_2/gorilla/berkeley-function-call-leaderboard/result/anthropic_claude-4-sonnet-thinking-on-10k"
    ]
    
    # Focus on multi-turn files which are large
    target_files = [
        'BFCL_v3_multi_turn_base_result.json',
        'BFCL_v3_multi_turn_long_context_result.json',
        'BFCL_v3_multi_turn_miss_func_result.json',
        'BFCL_v3_multi_turn_miss_param_result.json'
    ]
    
    all_results = {}
    sample_size = 200  # Sample 200 entries from each large file
    
    for base_path in base_paths:
        model_name = os.path.basename(base_path)
        all_results[model_name] = {}
        
        print(f"\nAnalyzing {model_name} (sampling {sample_size} entries from each large file)...")
        print("-" * 80)
        
        for filename in target_files:
            file_path = os.path.join(base_path, filename)
            if os.path.exists(file_path):
                print(f"  Analyzing {filename}...")
                results = analyzer.sample_analyze_large_file(file_path, sample_size)
                all_results[model_name][filename] = results
                
                total_errors = sum(results['type_errors'].values())
                if results['function_calls_analyzed'] > 0:
                    error_rate = (total_errors / results['function_calls_analyzed']) * 100
                    print(f"    Total entries in file: {results['total_entries']}")
                    print(f"    Sampled entries: {results['sampled_entries']}")
                    print(f"    Function calls analyzed: {results['function_calls_analyzed']}")
                    print(f"    Type errors found: {total_errors}")
                    print(f"    Error rate: {error_rate:.2f}%")
                    
                    if total_errors > 0:
                        for error_type, count in results['type_errors'].items():
                            if count > 0:
                                print(f"      {error_type}: {count}")
    
    # Compare with fixed versions
    print("\n" + "="*100)
    print("COMPARISON WITH FIXED VERSIONS (Sampled Analysis)")
    print("="*100)
    
    for base_path in base_paths:
        model_name = os.path.basename(base_path)
        fixed_path = os.path.join(base_path, "fixed")
        
        if os.path.exists(fixed_path):
            print(f"\n{model_name.upper()}:")
            print("-" * 80)
            
            for filename in target_files:
                fixed_filename = filename.replace('.json', '_fixed.json')
                original_file = os.path.join(base_path, filename)
                fixed_file = os.path.join(fixed_path, fixed_filename)
                
                if os.path.exists(original_file) and os.path.exists(fixed_file):
                    print(f"\n  {filename}:")
                    comparison = analyzer.compare_files_sampled(original_file, fixed_file, sample_size)
                    
                    original_total = sum(comparison['original']['type_errors'].values())
                    fixed_total = sum(comparison['fixed']['type_errors'].values())
                    
                    print(f"    Original errors (sampled): {original_total}")
                    print(f"    Fixed errors (sampled): {fixed_total}")
                    
                    if original_total > 0:
                        overall_improvement = ((original_total - fixed_total) / original_total) * 100
                        print(f"    Overall improvement: {overall_improvement:.1f}%")
                        
                        for error_type, improvement in comparison['improvements'].items():
                            if improvement['original_count'] > 0:
                                print(f"      {error_type}: {improvement['original_count']} → {improvement['fixed_count']} ({improvement['improvement_percentage']:.1f}% improvement)")
    
    # Calculate overall statistics
    print("\n" + "="*100)
    print("OVERALL STATISTICS (Based on Sampled Data)")
    print("="*100)
    
    for model_name, files in all_results.items():
        print(f"\n{model_name.upper()}:")
        print("-" * 60)
        
        total_function_calls = sum(file_data['function_calls_analyzed'] for file_data in files.values())
        total_type_errors = sum(sum(file_data['type_errors'].values()) for file_data in files.values())
        
        if total_function_calls > 0:
            error_percentage = (total_type_errors / total_function_calls) * 100
            print(f"  Total function calls analyzed (sampled): {total_function_calls}")
            print(f"  Total type errors found: {total_type_errors}")
            print(f"  Type error percentage: {error_percentage:.2f}%")
            
            # Show error type breakdown
            error_breakdown = defaultdict(int)
            for file_data in files.values():
                for error_type, count in file_data['type_errors'].items():
                    error_breakdown[error_type] += count
            
            if error_breakdown:
                print("  Error type breakdown:")
                for error_type, count in sorted(error_breakdown.items()):
                    percentage = (count / total_type_errors) * 100 if total_type_errors > 0 else 0
                    print(f"    {error_type}: {count} ({percentage:.1f}%)")
    
    # Show detailed examples
    print("\n" + "="*100)
    print("DETAILED EXAMPLES OF TYPE MISMATCHES")
    print("="*100)
    
    for model_name, files in all_results.items():
        print(f"\n{model_name.upper()}:")
        
        for filename, file_data in files.items():
            if sum(file_data['type_errors'].values()) > 0:
                print(f"\n  {filename}:")
                
                for error_type, examples in file_data['examples'].items():
                    if examples:
                        print(f"    {error_type.upper()} Examples:")
                        for i, example in enumerate(examples[:5], 1):  # Show top 5 examples
                            print(f"      {i}. Entry ID: {example['entry_id']}")
                            print(f"         Parameter: {example['param']}")
                            print(f"         Found: {example['value']}")
                            print(f"         Should be: {example['suggested']}")
                            print(f"         Context: {example['context'][:200]}...")
                            print()

if __name__ == "__main__":
    main()