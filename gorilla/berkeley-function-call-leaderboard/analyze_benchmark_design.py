import json
import os
from pathlib import Path
from dotenv import load_dotenv
import requests
import time
from typing import Dict, Any, List

# Load environment variables
load_dotenv()

class BenchmarkAnalyzer:
    def __init__(self):
        self.api_key = os.getenv('API_KEY')
        self.base_url = os.getenv('BASE_URL')
        self.model = "openai/gpt-4.1"
        
        if not self.api_key or not self.base_url:
            raise ValueError("API_KEY and BASE_URL must be set in .env file")
    
    def call_gpt4(self, prompt: str) -> str:
        """Call GPT-4.1 API to analyze a case"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(
                f"{self.base_url}chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        
        except Exception as e:
            print(f"API call failed: {e}")
            return "ERROR: Failed to analyze case"
    
    def create_analysis_prompt(self, case: Dict[str, Any]) -> str:
        """Create analysis prompt for GPT-4.1"""
        
        # Extract key information
        case_id = case.get('id', 'unknown')
        task_name = case.get('task_name', 'unknown')
        error = case.get('error', [])
        error_type = case.get('error_type', '')
        
        prompt_data = case.get('prompt', {})
        question = prompt_data.get('question', [[]])
        functions = prompt_data.get('function', [])
        
        # Get user question text
        user_question = ""
        if question and len(question) > 0 and len(question[0]) > 0:
            user_question = question[0][-1].get('content', 'No question found')
        
        # Get function descriptions
        function_descriptions = []
        for func in functions:
            func_name = func.get('name', 'unknown')
            func_desc = func.get('description', 'No description')
            func_params = func.get('parameters', {})
            function_descriptions.append(f"- {func_name}: {func_desc} | Parameters: {func_params}")
        
        # Get model's actual result if available
        model_result = case.get('model_result', case.get('model_result_raw', 'No result'))
        possible_answers = case.get('possible_answer', [])
        
        prompt = f"""
You are an expert evaluator analyzing whether a failed test case in a function calling benchmark represents a fundamental design flaw in the benchmark itself, or if it's a case where models should theoretically be able to succeed.

## Case Information:
- **Case ID**: {case_id}
- **Task Type**: {task_name}
- **Error Type**: {error_type}
- **Error Details**: {error}

## Test Setup:
**User Question**: {user_question}

**Available Functions**:
{chr(10).join(function_descriptions) if function_descriptions else "No functions provided"}

**Model's Response**: {model_result}

**Expected Answers**: {possible_answers}

## Analysis Task:
Analyze this failed case and determine:

1. **Is this a benchmark design flaw?** 
   - Are the instructions unclear or ambiguous?
   - Is the question-function mismatch fundamental?
   - Are the available functions insufficient for the task?
   - Is the expected behavior unreasonable?

2. **Could a perfect model theoretically succeed?**
   - Given the system prompt, question, and available functions
   - Is there a clear, logical path to the correct answer?
   - Are there fundamental impossibilities in the task design?

3. **What is the root cause of failure?**
   - Benchmark design issue
   - Model capability limitation  
   - Edge case that's reasonable to fail
   - Ambiguous specification

## Your Response Format:
Provide a structured analysis in this format:

**VERDICT**: [DESIGN_FLAW | LEGITIMATE_TEST | EDGE_CASE]

**REASONING**: 
[2-3 sentences explaining your judgment]

**DETAILED_ANALYSIS**:
- Issue Type: [describe the main problem]
- Solvability: [whether a perfect model could solve this]
- Recommendation: [keep as-is, modify, or remove from benchmark]

Be direct and decisive in your analysis. Focus on whether this represents a fair test of function calling capabilities.
"""
        
        return prompt.strip()
    
    def analyze_case(self, case: Dict[str, Any]) -> Dict[str, str]:
        """Analyze a single failed case"""
        prompt = self.create_analysis_prompt(case)
        analysis = self.call_gpt4(prompt)
        
        return {
            'case_id': case.get('id', 'unknown'),
            'task_name': case.get('task_name', 'unknown'),
            'analysis': analysis
        }
    
    def analyze_all_cases(self, input_file: str, output_file: str, max_cases: int = None):
        """Analyze all failed cases from the input file"""
        
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"Input file not found: {input_file}")
            return
        
        cases = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        case = json.loads(line.strip())
                        cases.append(case)
                    except json.JSONDecodeError:
                        continue
        
        print(f"Found {len(cases)} failed cases to analyze")
        
        if max_cases:
            cases = cases[:max_cases]
            print(f"Limiting analysis to first {max_cases} cases")
        
        results = []
        
        for i, case in enumerate(cases, 1):
            print(f"Analyzing case {i}/{len(cases)}: {case.get('id', 'unknown')}")
            
            try:
                result = self.analyze_case(case)
                results.append(result)
                
                # Save progress periodically
                if i % 10 == 0:
                    self.save_results(results, output_file)
                    print(f"Progress saved: {i} cases analyzed")
                
                # Rate limiting - wait 1 second between requests
                time.sleep(1)
                
            except Exception as e:
                print(f"Failed to analyze case {case.get('id', 'unknown')}: {e}")
                results.append({
                    'case_id': case.get('id', 'unknown'),
                    'task_name': case.get('task_name', 'unknown'),
                    'analysis': f"ERROR: {str(e)}"
                })
        
        # Final save
        self.save_results(results, output_file)
        print(f"Analysis complete! Results saved to {output_file}")
        
        # Print summary
        self.print_summary(results)
    
    def save_results(self, results: List[Dict], output_file: str):
        """Save results to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def print_summary(self, results: List[Dict]):
        """Print analysis summary"""
        print(f"\n=== ANALYSIS SUMMARY ===")
        print(f"Total cases analyzed: {len(results)}")
        
        # Count verdicts
        verdicts = {}
        errors = 0
        
        for result in results:
            analysis = result['analysis']
            if analysis.startswith('ERROR:'):
                errors += 1
                continue
                
            # Extract verdict from analysis
            lines = analysis.split('\n')
            verdict_line = None
            for line in lines:
                if line.startswith('**VERDICT**:'):
                    verdict_line = line.replace('**VERDICT**:', '').strip()
                    break
            
            if verdict_line:
                # Extract the verdict (first word in brackets or first word)
                if '[' in verdict_line and ']' in verdict_line:
                    verdict = verdict_line.split('[')[1].split(']')[0]
                else:
                    verdict = verdict_line.split()[0] if verdict_line.split() else 'UNKNOWN'
            else:
                verdict = 'UNKNOWN'
            
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
        
        print(f"Errors: {errors}")
        print(f"Verdict distribution:")
        for verdict, count in sorted(verdicts.items()):
            percentage = (count / len(results)) * 100
            print(f"  {verdict}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    analyzer = BenchmarkAnalyzer()
    
    input_file = r"E:\Users\김현준\Downloads\agent_hard_benchmark_2\gorilla\berkeley-function-call-leaderboard\score\all_fail_case.json"
    output_file = r"E:\Users\김현준\Downloads\agent_hard_benchmark_2\gorilla\berkeley-function-call-leaderboard\score\benchmark_design_analysis.json"
    
    # Analyze all cases (or set max_cases=20 for testing)
    analyzer.analyze_all_cases(input_file, output_file, max_cases=None)