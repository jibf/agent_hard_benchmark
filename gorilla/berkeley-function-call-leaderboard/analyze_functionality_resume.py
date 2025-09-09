import json
import os
from pathlib import Path
from dotenv import load_dotenv
import requests
import time
from typing import Dict, Any, List, Set
from collections import defaultdict
from datetime import datetime

# Load environment variables
load_dotenv()

class FunctionalityMismatchAnalyzerResume:
    def __init__(self):
        self.api_key = os.getenv('API_KEY')
        self.base_url = os.getenv('BASE_URL')
        self.model = "openai/gpt-4.1"
        
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
        
        # Track completed tasks
        self.completed_tasks = self.get_completed_tasks()
    
    def get_completed_tasks(self) -> Set[str]:
        """Check which tasks have already been completed"""
        output_dir = Path(r"E:\Users\김현준\Downloads\agent_hard_benchmark_2\gorilla\berkeley-function-call-leaderboard\score")
        completed = set()
        
        for task_type in self.target_tasks:
            result_file = output_dir / f"functionality_analysis_{task_type}.json"
            if result_file.exists():
                # Check if the file contains complete results
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'completed_at' in data:  # Check for completion marker
                            completed.add(task_type)
                            print(f"✓ {task_type}: Already completed")
                        else:
                            print(f"⚠ {task_type}: Partial results found, will resume")
                except:
                    print(f"⚠ {task_type}: Error reading file, will redo")
            else:
                print(f"○ {task_type}: Not started")
        
        return completed
    
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
    
    def get_processed_cases(self, task_type: str) -> Set[str]:
        """Get list of already processed case IDs for a task"""
        processed = set()
        output_dir = Path(r"E:\Users\김현준\Downloads\agent_hard_benchmark_2\gorilla\berkeley-function-call-leaderboard\score")
        
        # Check temp file first
        temp_file = output_dir / f"temp_functionality_analysis_{task_type}.json"
        if temp_file.exists():
            try:
                with open(temp_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for case in data:
                        processed.add(case.get('case_id', ''))
                print(f"  Found {len(processed)} cases in temp file")
            except:
                pass
        
        # Also check main result file
        result_file = output_dir / f"functionality_analysis_{task_type}.json"
        if result_file.exists():
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'cases' in data:
                        for case in data['cases']:
                            processed.add(case.get('case_id', ''))
                print(f"  Found {len(processed)} cases in result file")
            except:
                pass
        
        return processed
    
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
        """Call GPT-4.1 API"""
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
            return f"ERROR: {str(e)}"
    
    def analyze_case(self, case: Dict, system_prompt: str) -> Dict:
        """Analyze a single case"""
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
    
    def analyze_task_type_resume(self, task_type: str, batch_size: int = 10) -> Dict:
        """Analyze all cases for a specific task type with resume capability"""
        print(f"\n{'='*60}")
        print(f"Analyzing task type: {task_type}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # Get already processed cases
        processed_cases = self.get_processed_cases(task_type)
        
        # Load test cases
        test_cases = self.load_test_data(task_type)
        
        if not test_cases:
            print(f"Warning: No test cases found for {task_type}")
            return {
                'task_type': task_type,
                'total_cases': 0,
                'error': 'No test cases found'
            }
        
        # Filter out already processed cases
        remaining_cases = [case for case in test_cases if case.get('id', '') not in processed_cases]
        
        print(f"Total cases: {len(test_cases)}")
        print(f"Already processed: {len(processed_cases)}")
        print(f"Remaining to process: {len(remaining_cases)}")
        
        if len(remaining_cases) == 0:
            print(f"All cases already processed!")
            # Load and return existing results
            output_dir = Path(r"E:\Users\김현준\Downloads\agent_hard_benchmark_2\gorilla\berkeley-function-call-leaderboard\score")
            result_file = output_dir / f"functionality_analysis_{task_type}.json"
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        # Load existing results if any
        existing_results = []
        if len(processed_cases) > 0:
            output_dir = Path(r"E:\Users\김현준\Downloads\agent_hard_benchmark_2\gorilla\berkeley-function-call-leaderboard\score")
            temp_file = output_dir / f"temp_functionality_analysis_{task_type}.json"
            if temp_file.exists():
                with open(temp_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
        
        results = existing_results.copy()
        
        # Process remaining cases
        for i, case in enumerate(remaining_cases):
            current_idx = len(processed_cases) + i + 1
            print(f"Processing case {current_idx}/{len(test_cases)}: {case.get('id', 'unknown')}")
            
            result = self.analyze_case(case, self.system_prompt)
            results.append(result)
            
            # Save intermediate results every 10 cases
            if (i + 1) % 10 == 0:
                self.save_intermediate_results(task_type, results)
                print(f"  Intermediate save: {len(results)} cases processed")
            
            # Rate limiting
            time.sleep(0.3)
        
        # Final save of intermediate results
        self.save_intermediate_results(task_type, results)
        
        # Calculate summary
        mismatch_count = sum(1 for r in results if r['verdict'] == 'MISMATCH')
        mismatch_types = defaultdict(int)
        for r in results:
            if r['verdict'] == 'MISMATCH':
                mismatch_types[r['mismatch_type']] += 1
        
        summary = {
            'task_type': task_type,
            'total_cases': len(test_cases),
            'mismatch_cases': mismatch_count,
            'mismatch_rate': (mismatch_count / len(test_cases) * 100) if test_cases else 0,
            'mismatch_types': dict(mismatch_types),
            'cases': results,
            'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return summary
    
    def save_intermediate_results(self, task_type: str, results: List[Dict]):
        """Save intermediate results during processing"""
        output_dir = Path(r"E:\Users\김현준\Downloads\agent_hard_benchmark_2\gorilla\berkeley-function-call-leaderboard\score")
        temp_file = output_dir / f"temp_functionality_analysis_{task_type}.json"
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def run_resume_analysis(self):
        """Run analysis on remaining task types only"""
        output_dir = Path(r"E:\Users\김현준\Downloads\agent_hard_benchmark_2\gorilla\berkeley-function-call-leaderboard\score")
        output_dir.mkdir(exist_ok=True)
        
        # Get remaining tasks
        remaining_tasks = [task for task in self.target_tasks if task not in self.completed_tasks]
        
        if not remaining_tasks:
            print("All tasks already completed!")
            return
        
        print(f"\n{'='*60}")
        print(f"RESUMING ANALYSIS")
        print(f"{'='*60}")
        print(f"Completed tasks: {list(self.completed_tasks)}")
        print(f"Remaining tasks: {remaining_tasks}")
        print(f"Estimated time: ~{len(remaining_tasks) * 5} minutes")
        
        all_results = {}
        
        # Load existing completed results
        for completed_task in self.completed_tasks:
            result_file = output_dir / f"functionality_analysis_{completed_task}.json"
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    all_results[completed_task] = json.load(f)
        
        # Process remaining tasks
        for idx, task_type in enumerate(remaining_tasks, 1):
            print(f"\n[{idx}/{len(remaining_tasks)}] Processing {task_type}...")
            
            result = self.analyze_task_type_resume(task_type)
            all_results[task_type] = result
            
            # Save individual task results
            task_output = output_dir / f"functionality_analysis_{task_type}.json"
            with open(task_output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"Results saved to: {task_output}")
            print(f"Mismatch rate: {result.get('mismatch_rate', 0):.1f}%")
            
            # Clean up temp files
            temp_file = output_dir / f"temp_functionality_analysis_{task_type}.json"
            if temp_file.exists():
                temp_file.unlink()
        
        # Save combined results
        combined_output = output_dir / "functionality_analysis_all_tasks.json"
        with open(combined_output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        # Print final summary
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        
        total_cases = 0
        total_mismatches = 0
        
        for task_type, result in all_results.items():
            cases = result.get('total_cases', 0)
            mismatches = result.get('mismatch_cases', 0)
            rate = result.get('mismatch_rate', 0)
            
            total_cases += cases
            total_mismatches += mismatches
            
            print(f"{task_type:25} - Cases: {cases:4}, Mismatches: {mismatches:4} ({rate:.1f}%)")
        
        if total_cases > 0:
            overall_rate = (total_mismatches / total_cases) * 100
            print(f"\n{'OVERALL':25} - Cases: {total_cases:4}, Mismatches: {total_mismatches:4} ({overall_rate:.1f}%)")
        
        print(f"\nAll results saved to: {combined_output}")
        print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    analyzer = FunctionalityMismatchAnalyzerResume()
    analyzer.run_resume_analysis()