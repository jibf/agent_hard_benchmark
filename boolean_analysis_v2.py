#!/usr/bin/env python3
"""
Enhanced Boolean Parameter Type Strictness Analysis for BFCL Claude Results
Focuses specifically on parameters that should be boolean but are passed as strings.
"""

import json
import re
from collections import Counter
from pathlib import Path


class EnhancedBooleanAnalyzer:
    def __init__(self):
        # Focus on clear boolean parameter patterns
        self.boolean_param_patterns = [
            # Common boolean parameter names
            r'\b(is_\w+|has_\w+|enable_?\w*|disable_?\w*|should_\w+|can_\w+|will_\w+)',
            r'\b(include_?\w*|exclude_?\w*|allow_?\w*|deny_?\w*|require_?\w*)',
            r'\b(formatted?|debug|verbose|strict|force|recursive)',
            r'\b(ascending|descending|reverse|sorted)',
            r'\b(live_\w+|real_time|immediate)',
            r'\b(\w*enabled?|\w*disabled?)',
            r'\b(case_?insensitive|ignore_case)',
            r'\b(show_\w+|hide_\w+|display_\w+)',
            r'\b(use_\w+|apply_\w+)',
            r'\b(aligned?|inlined?|compressed?)',
        ]
        
        # String values that clearly indicate boolean intent
        self.string_boolean_values = ['true', 'false', 'yes', 'no', 'on', 'off', '1', '0']
        
        self.stats = {
            'total_function_calls': 0,
            'boolean_issues': [],
            'param_name_counts': Counter(),
            'boolean_value_counts': Counter(),
            'files_analyzed': 0,
        }

    def extract_function_calls(self, result_text):
        """Extract function calls from result text."""
        # Match function calls in brackets
        pattern = r'\[([^\]]+)\]'
        return re.findall(pattern, result_text)

    def parse_parameters(self, function_call):
        """Parse parameters from a function call string."""
        # Match parameter=value patterns
        param_pattern = r'(\w+)\s*=\s*(["\']?)([^,\]]+?)\2(?=\s*[,\)]|\s*$)'
        matches = re.findall(param_pattern, function_call)
        
        parameters = []
        for param_name, quote, value in matches:
            # Clean up the value
            value = value.strip()
            is_quoted = bool(quote)
            parameters.append((param_name, value, is_quoted))
        
        return parameters

    def is_boolean_parameter(self, param_name, value, is_quoted):
        """Determine if a parameter should be boolean based on name and value."""
        param_name_lower = param_name.lower()
        value_lower = value.lower()
        
        # Check if parameter name suggests boolean
        name_is_boolean = any(
            re.search(pattern, param_name_lower) 
            for pattern in self.boolean_param_patterns
        )
        
        # Check if value suggests boolean
        value_is_boolean = value_lower in self.string_boolean_values
        
        # It's a boolean issue if:
        # 1. Parameter name suggests boolean AND value is string boolean
        # 2. OR value is clearly boolean string (true/false/yes/no) regardless of param name
        return (
            (name_is_boolean and value_is_boolean and is_quoted) or
            (value_lower in ['true', 'false', 'yes', 'no'] and is_quoted)
        )

    def analyze_function_call(self, function_call):
        """Analyze a single function call for boolean issues."""
        self.stats['total_function_calls'] += 1
        
        parameters = self.parse_parameters(function_call)
        
        for param_name, value, is_quoted in parameters:
            if self.is_boolean_parameter(param_name, value, is_quoted):
                self.stats['boolean_issues'].append({
                    'function_call': function_call,
                    'param_name': param_name,
                    'value': value,
                    'is_quoted': is_quoted,
                    'suggested_fix': self.suggest_boolean_fix(value)
                })
                
                self.stats['param_name_counts'][param_name] += 1
                self.stats['boolean_value_counts'][value.lower()] += 1

    def suggest_boolean_fix(self, string_value):
        """Suggest the proper boolean value for a string."""
        value_lower = string_value.lower()
        if value_lower in ['true', '1', 'yes', 'on']:
            return 'True'
        elif value_lower in ['false', '0', 'no', 'off']:
            return 'False'
        return f'convert("{string_value}") to boolean'

    def analyze_result_file(self, file_path):
        """Analyze a single result file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'result' in data and isinstance(data['result'], str):
                            function_calls = self.extract_function_calls(data['result'])
                            for call in function_calls:
                                self.analyze_function_call(call)
                    except json.JSONDecodeError:
                        continue
            self.stats['files_analyzed'] += 1
        except Exception as e:
            print(f"Error analyzing file {file_path}: {e}")

    def analyze_directories(self, directories):
        """Analyze multiple result directories."""
        for directory in directories:
            directory_path = Path(directory)
            if not directory_path.exists():
                print(f"Directory not found: {directory}")
                continue
                
            result_files = list(directory_path.glob("BFCL_v3_*.json"))
            print(f"Found {len(result_files)} files in {directory_path.name}")
            
            for file_path in result_files:
                self.analyze_result_file(file_path)

    def generate_report(self):
        """Generate comprehensive analysis report."""
        total_issues = len(self.stats['boolean_issues'])
        total_calls = self.stats['total_function_calls']
        
        report = f"""
=== Enhanced Boolean Parameter Type Strictness Analysis ===

Files Analyzed: {self.stats['files_analyzed']}
Total Function Calls: {total_calls:,}
Boolean Parameter Issues: {total_issues:,}
Issue Rate: {(total_issues/total_calls*100):.2f}% of all function calls

=== Top Boolean Parameters with Type Issues ===
"""
        for param, count in self.stats['param_name_counts'].most_common(20):
            percentage = (count/total_issues*100) if total_issues > 0 else 0
            report += f"{param}: {count} issues ({percentage:.1f}%)\n"

        report += f"""
=== String Boolean Values Distribution ===
"""
        for value, count in self.stats['boolean_value_counts'].most_common():
            percentage = (count/total_issues*100) if total_issues > 0 else 0
            report += f'"{value}": {count} occurrences ({percentage:.1f}%)\n'

        report += f"""
=== Sample Boolean Type Issues ===
"""
        # Show representative examples
        for i, issue in enumerate(self.stats['boolean_issues'][:15]):
            report += f"""
Issue #{i+1}:
  Function: {issue['function_call'][:120]}{'...' if len(issue['function_call']) > 120 else ''}
  Parameter: {issue['param_name']}
  Current Value: "{issue['value']}" (quoted string)
  Should Be: {issue['suggested_fix']} (boolean)
"""

        # Analysis by category
        report += f"""
=== Analysis by Boolean Parameter Categories ===
"""
        categories = {
            'is_/has_': [p for p in self.stats['param_name_counts'].keys() if p.startswith(('is_', 'has_'))],
            'enable/disable': [p for p in self.stats['param_name_counts'].keys() if 'enable' in p.lower() or 'disable' in p.lower()],
            'include/exclude': [p for p in self.stats['param_name_counts'].keys() if 'include' in p.lower() or 'exclude' in p.lower()],
            'show/hide': [p for p in self.stats['param_name_counts'].keys() if any(word in p.lower() for word in ['show', 'hide', 'display'])],
            'flags': [p for p in self.stats['param_name_counts'].keys() if p.lower() in ['formatted', 'debug', 'verbose', 'strict', 'force', 'aligned', 'ascending']],
        }
        
        for category, params in categories.items():
            if params:
                total_count = sum(self.stats['param_name_counts'][p] for p in params)
                report += f"{category}: {total_count} issues across {len(params)} parameters\n"
                for param in sorted(params, key=lambda p: self.stats['param_name_counts'][p], reverse=True)[:5]:
                    report += f"  - {param}: {self.stats['param_name_counts'][param]} issues\n"

        return report


def main():
    analyzer = EnhancedBooleanAnalyzer()
    
    directories = [
        "E:/Users/김현준/Downloads/agent_hard_benchmark_2/gorilla/berkeley-function-call-leaderboard/result/anthropic_claude-4-sonnet-thinking-off",
        "E:/Users/김현준/Downloads/agent_hard_benchmark_2/gorilla/berkeley-function-call-leaderboard/result/anthropic_claude-4-sonnet-thinking-on-10k"
    ]
    
    print("Starting Enhanced Boolean Parameter Type Strictness Analysis")
    print("=" * 60)
    
    analyzer.analyze_directories(directories)
    
    report = analyzer.generate_report()
    print(report)
    
    # Save detailed report
    report_file = "E:/Users/김현준/Downloads/agent_hard_benchmark_2/enhanced_boolean_analysis.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
        
        # Add all issues for detailed analysis
        f.write(f"\n\n=== Complete Issue List ===\n")
        for i, issue in enumerate(analyzer.stats['boolean_issues']):
            f.write(f"\nIssue #{i+1}:\n")
            f.write(f"  Function: {issue['function_call']}\n")
            f.write(f"  Parameter: {issue['param_name']}\n")
            f.write(f"  Current: \"{issue['value']}\"\n")
            f.write(f"  Should Be: {issue['suggested_fix']}\n")
    
    print(f"\nDetailed report saved to: {report_file}")


if __name__ == "__main__":
    main()