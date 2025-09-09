import json
import os
from pathlib import Path
from dotenv import load_dotenv
import requests
import time
from typing import Dict, Any, List
from collections import defaultdict
from datetime import datetime
import concurrent.futures
from multiprocessing import Pool, cpu_count
import threading

# Load environment variables
load_dotenv()

class FunctionalityMismatchAnalyzerMP:
    def __init__(self, max_workers: int = 16):
        self.api_key = os.getenv('API_KEY')
        self.base_url = os.getenv('BASE_URL')
        self.model = "openai/gpt-4.1"
        self.max_workers = max_workers
        
        if not self.api_key or not self.base_url:
            raise ValueError("API_KEY and BASE_URL must be set in .env file")
        
        # Load system prompt
        self.system_prompt = self.load_system_prompt()
        
        # Target task types
        self.target_tasks = [
            'live_multiple',
            'multi_turn_long_context', 
            'multi_turn_miss_func',
            'live_irrelevance',
            'multi_turn_miss_param',
            'multi_turn_base',
            'irrelevance',
            'live_simple'
        ]
        
        # Thread lock for file operations
        self.file_lock = threading.Lock()
    
    def load_system_prompt(self) -> str:
        """Load the default system prompt from BFCL"""
        prompt_path = Path(r"E:\Users\김현준\Downloads\agent_hard_benchmark_2\gorilla\berkeley-function-call-leaderboard\bfcl_eval\constants\default_prompts.py")
        
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract DEFAULT_SYSTEM_PROMPT from the file
                if 'DEFAULT_SYSTEM_PROMPT_WITHOUT_FUNC_DOC = """' in content:
                    start = content.find('DEFAULT_SYSTEM_PROMPT_WITHOUT_FUNC_DOC = """') + len('DEFAULT_SYSTEM_PROMPT_WITHOUT_FUNC_DOC = """')
                    end = content.find('"""', start)
                    return content[start:end].strip()
        
        return "System prompt not found"
    
    def load_test_data(self, task_type: str) -> List[Dict]:
        """Load all test cases for a specific task type"""
        data_dir = Path(r"E:\Users\김현준\Downloads\agent_hard_benchmark_2\gorilla\berkeley-function-call-leaderboard\bfcl_eval\data")
        
        # Use BFCL_v3_ prefix for file names
        file_path = data_dir / f"BFCL_v3_{task_type}.json"
        
        test_cases = []
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                # Handle both JSON and JSONL formats
                if content.startswith('['):
                    # JSON array format
                    test_cases = json.loads(content)
                else:
                    # JSONL format
                    for line in content.split('\n'):
                        if line.strip():
                            try:
                                test_cases.append(json.loads(line.strip()))
                            except json.JSONDecodeError:
                                continue
        
        return test_cases
    
    def create_analysis_prompt(self, case: Dict[str, Any], system_prompt: str) -> str:
        """Create prompt for functionality mismatch analysis"""
        
        case_id = case.get('id', 'unknown')
        question = case.get('question', [[]])
        functions = case.get('function', [])
        
        # Extract user question
        user_question = ""
        if isinstance(question, list) and len(question) > 0:
            if isinstance(question[0], list) and len(question[0]) > 0:
                for msg in question[0]:
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        user_question = msg.get('content', '')
                        break
        
        # Format functions
        function_info = []
        if functions:
            for func in functions:
                func_name = func.get('name', 'unknown')
                func_desc = func.get('description', 'No description')
                func_params = func.get('parameters', {})
                function_info.append(f"- Name: {func_name}\n  Description: {func_desc}\n  Parameters: {json.dumps(func_params, indent=2)}")
        
        prompt = f"""
You are analyzing a function-calling benchmark test case for functionality mismatches.

## System Prompt Used in Benchmark:
{system_prompt}

## Test Case Analysis

**Case ID**: {case_id}

**User Question/Request**: 
{user_question}

**Available Functions**: 
{chr(10).join(function_info) if function_info else "NO FUNCTIONS PROVIDED (Empty function list)"}

## Your Task:
Determine if this test case has a **functionality mismatch** problem. A functionality mismatch occurs when:

1. **Empty Function List**: No functions are provided but the test expects function calls
2. **Domain Mismatch**: The provided functions cannot fulfill the user's intent (e.g., user asks for bakery, only grocery shop function available)
3. **Missing Core Functionality**: Essential functions for completing the task are absent
4. **Incompatible Functions**: Functions exist but their parameters/capabilities don't match the requirements

## Analysis Required:

Please analyze and respond in this exact format:

**VERDICT**: [MISMATCH | NO_MISMATCH | UNCERTAIN]

**MISMATCH_TYPE**: [EMPTY_FUNCTIONS | DOMAIN_MISMATCH | MISSING_FUNCTIONALITY | INCOMPATIBLE | N/A]

**REASONING**: 
[2-3 sentences explaining your judgment]

**DETAILS**:
- Can user intent be fulfilled?: [YES/NO/PARTIALLY]
- Missing capabilities: [List what's missing if applicable]
- Severity: [CRITICAL/HIGH/MEDIUM/LOW/N/A]

Focus only on whether the provided functions can theoretically fulfill the user's request. Do not consider implementation difficulty or model capabilities.
"""
        
        return prompt.strip()
    
    def call_gpt4(self, prompt: str) -> str:
        """Call GPT-4.1 API with retry logic"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.1
        }
        
        max_retries = 3
        for attempt in range(max_retries):
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
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    print(f"API call failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"API call failed after {max_retries} attempts: {e}")
                    return f"ERROR: {str(e)}"
    
    def analyze_case_worker(self, case_data: tuple) -> Dict:
        """Worker function for multiprocessing - analyze a single case"""
        case, system_prompt = case_data
        
        # Create analysis prompt
        prompt = self.create_analysis_prompt(case, system_prompt)
        
        # Call API
        analysis = self.call_gpt4(prompt)
        
        # Parse result
        verdict = "UNKNOWN"
        mismatch_type = "N/A"
        
        if "**VERDICT**:" in analysis:
            verdict_line = analysis.split("**VERDICT**:")[1].split("\n")[0].strip()
            verdict = verdict_line.strip('[]').strip()
        
        if "**MISMATCH_TYPE**:" in analysis:
            type_line = analysis.split("**MISMATCH_TYPE**:")[1].split("\n")[0].strip()
            mismatch_type = type_line.strip('[]').strip()
        
        return {
            'case_id': case.get('id', 'unknown'),
            'verdict': verdict,
            'mismatch_type': mismatch_type,
            'analysis': analysis
        }
    
    def analyze_task_type_mp(self, task_type: str) -> Dict:
        """Analyze all cases for a specific task type using multiprocessing"""
        print(f"\n{'='*60}")
        print(f"Analyzing task type: {task_type}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using {self.max_workers} parallel workers")
        print(f"{'='*60}")
        
        # Load test cases
        test_cases = self.load_test_data(task_type)
        
        if not test_cases:
            print(f"Warning: No test cases found for {task_type}")
            return {
                'task_type': task_type,
                'total_cases': 0,
                'error': 'No test cases found'
            }
        
        print(f"Found {len(test_cases)} test cases")
        
        # Prepare data for workers (case, system_prompt)
        worker_data = [(case, self.system_prompt) for case in test_cases]
        
        results = []
        
        # Use ThreadPoolExecutor for I/O bound API calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_case = {executor.submit(self.analyze_case_worker, case_data): case_data[0]['id'] 
                             for case_data in worker_data}
            
            # Process completed tasks
            completed = 0
            for future in concurrent.futures.as_completed(future_to_case):
                case_id = future_to_case[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if completed % 50 == 0:
                        print(f"Progress: {completed}/{len(test_cases)} cases completed ({completed/len(test_cases)*100:.1f}%)")
                        self.save_intermediate_results(task_type, results)
                    
                except Exception as e:
                    print(f"Case {case_id} failed: {e}")
                    # Add error result
                    results.append({
                        'case_id': case_id,
                        'verdict': 'ERROR',
                        'mismatch_type': 'N/A',
                        'analysis': f'Error: {str(e)}'
                    })
        
        # Sort results by case_id for consistency
        results.sort(key=lambda x: x['case_id'])
        
        # Calculate summary
        mismatch_count = sum(1 for r in results if r['verdict'] == 'MISMATCH')
        error_count = sum(1 for r in results if r['verdict'] == 'ERROR')
        mismatch_types = defaultdict(int)
        
        for r in results:
            if r['verdict'] == 'MISMATCH':
                mismatch_types[r['mismatch_type']] += 1
        
        summary = {
            'task_type': task_type,
            'total_cases': len(test_cases),
            'mismatch_cases': mismatch_count,
            'error_cases': error_count,
            'mismatch_rate': (mismatch_count / len(test_cases) * 100) if test_cases else 0,
            'error_rate': (error_count / len(test_cases) * 100) if test_cases else 0,
            'mismatch_types': dict(mismatch_types),
            'cases': results,
            'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return summary
    
    def save_intermediate_results(self, task_type: str, results: List[Dict]):
        """Save intermediate results during processing"""
        output_dir = Path(r"E:\Users\김현준\Downloads\agent_hard_benchmark_2\gorilla\berkeley-function-call-leaderboard\score")
        temp_file = output_dir / f"temp_functionality_analysis_{task_type}.json"
        
        with self.file_lock:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    
    def run_multiprocess_analysis(self):
        """Run analysis on all target task types using multiprocessing"""
        output_dir = Path(r"E:\Users\김현준\Downloads\agent_hard_benchmark_2\gorilla\berkeley-function-call-leaderboard\score")
        output_dir.mkdir(exist_ok=True)
        
        all_results = {}
        start_time = datetime.now()
        
        print(f"Analysis started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using {self.max_workers} parallel workers")
        print(f"Expected speedup: ~{self.max_workers}x faster than sequential")
        
        # Calculate total cases
        total_cases = 0
        for task_type in self.target_tasks:
            test_cases = self.load_test_data(task_type)
            total_cases += len(test_cases)
        
        print(f"Total cases to process: {total_cases}")
        print(f"Estimated completion time: ~{total_cases / (self.max_workers * 2)} minutes")
        
        for idx, task_type in enumerate(self.target_tasks, 1):
            print(f"\n[{idx}/{len(self.target_tasks)}] Processing {task_type}...")
            
            result = self.analyze_task_type_mp(task_type)
            all_results[task_type] = result
            
            # Save individual task results
            task_output = output_dir / f"functionality_analysis_{task_type}.json"
            with open(task_output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"Results saved to: {task_output}")
            print(f"Mismatch rate: {result.get('mismatch_rate', 0):.1f}%")
            print(f"Error rate: {result.get('error_rate', 0):.1f}%")
            
            # Clean up temp files
            temp_file = output_dir / f"temp_functionality_analysis_{task_type}.json"
            if temp_file.exists():
                temp_file.unlink()
        
        # Save combined results
        combined_output = output_dir / "functionality_analysis_all_tasks.json"
        with open(combined_output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        # Calculate total time
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Print final summary
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        
        total_cases = 0
        total_mismatches = 0
        total_errors = 0
        
        for task_type, result in all_results.items():
            cases = result.get('total_cases', 0)
            mismatches = result.get('mismatch_cases', 0)
            errors = result.get('error_cases', 0)
            mismatch_rate = result.get('mismatch_rate', 0)
            error_rate = result.get('error_rate', 0)
            
            total_cases += cases
            total_mismatches += mismatches
            total_errors += errors
            
            print(f"{task_type:25} - Cases: {cases:4}, Mismatches: {mismatches:4} ({mismatch_rate:.1f}%), Errors: {errors:3} ({error_rate:.1f}%)")
        
        if total_cases > 0:
            overall_mismatch_rate = (total_mismatches / total_cases) * 100
            overall_error_rate = (total_errors / total_cases) * 100
            print(f"\n{'OVERALL':25} - Cases: {total_cases:4}, Mismatches: {total_mismatches:4} ({overall_mismatch_rate:.1f}%), Errors: {total_errors:3} ({overall_error_rate:.1f}%)")
        
        print(f"\nAll results saved to: {combined_output}")
        print(f"Total analysis time: {duration}")
        print(f"Average time per case: {duration.total_seconds() / total_cases:.2f} seconds")
        print(f"Analysis completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    # Use 16 parallel workers as requested
    analyzer = FunctionalityMismatchAnalyzerMP(max_workers=16)
    analyzer.run_multiprocess_analysis()