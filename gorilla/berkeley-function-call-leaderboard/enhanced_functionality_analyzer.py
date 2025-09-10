import json
import os
import sys
import re
from pathlib import Path
from dotenv import load_dotenv
import requests
import time
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import concurrent.futures
import multiprocessing as mp
from functools import partial
import threading
import queue

# Add BFCL path to system path for imports
sys.path.append(str(Path(__file__).parent / "bfcl_eval"))

try:
    # Import BFCL modules for proper processing
    from bfcl_eval.constants.default_prompts import DEFAULT_SYSTEM_PROMPT_WITHOUT_FUNC_DOC, DEFAULT_SYSTEM_PROMPT
    from bfcl_eval.model_handler.utils import (
        system_prompt_pre_processing_chat_model,
        convert_system_prompt_into_user_prompt,
        combine_consecutive_user_prompts
    )
except ImportError:
    # Fallback definitions if imports fail
    DEFAULT_SYSTEM_PROMPT_WITHOUT_FUNC_DOC = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.
"""

# Load environment variables
load_dotenv()

class EnhancedFunctionalityAnalyzer:
    def __init__(self, num_workers: int = 8):
        self.api_key = os.getenv('API_KEY')
        self.base_url = os.getenv('BASE_URL')
        self.model = "openai/gpt-4.1"
        self.num_workers = num_workers
        
        if not self.api_key or not self.base_url:
            raise ValueError("API_KEY and BASE_URL must be set in .env file")
        
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
        
        # Model-specific configurations
        self.model_configs = {
            'openai': {
                'supports_function_calling': True,
                'temperature': 0.0,
                'max_tokens': 500,
                'system_prompt_processing': 'standard'
            },
            'anthropic': {
                'supports_function_calling': True,
                'temperature': 0.0, 
                'max_tokens': 500,
                'system_prompt_processing': 'xml_tools'
            },
            'default': {
                'supports_function_calling': False,
                'temperature': 0.1,
                'max_tokens': 500,
                'system_prompt_processing': 'prompt_based'
            }
        }
    
    def get_model_config(self, model_name: str) -> Dict:
        """Get configuration for specific model"""
        if 'openai' in model_name.lower():
            return self.model_configs['openai']
        elif 'anthropic' in model_name.lower() or 'claude' in model_name.lower():
            return self.model_configs['anthropic']
        else:
            return self.model_configs['default']
    
    def format_function_docs(self, functions: List[Dict], model_config: Dict) -> str:
        """Format function documentation according to model requirements"""
        if not functions:
            return "NO FUNCTIONS PROVIDED"
        
        processing_type = model_config.get('system_prompt_processing', 'standard')
        
        if processing_type == 'xml_tools':
            # Claude/Anthropic XML format
            return self._format_functions_xml(functions)
        elif processing_type == 'standard' and model_config.get('supports_function_calling'):
            # OpenAI function calling format
            return self._format_functions_openai(functions)
        else:
            # Standard prompt-based format
            return self._format_functions_standard(functions)
    
    def _format_functions_xml(self, functions: List[Dict]) -> str:
        """Format functions in XML format for Claude"""
        xml_parts = []
        for func in functions:
            name = func.get('name', 'unknown')
            desc = func.get('description', 'No description')
            params = func.get('parameters', {})
            
            xml_parts.append(f'<tool name="{name}">')
            xml_parts.append(f'<description>{desc}</description>')
            
            if params.get('properties'):
                xml_parts.append('<parameters>')
                for param_name, param_info in params['properties'].items():
                    param_type = param_info.get('type', 'string')
                    param_desc = param_info.get('description', '')
                    required = param_name in params.get('required', [])
                    xml_parts.append(f'<parameter name="{param_name}" type="{param_type}" required="{required}">{param_desc}</parameter>')
                xml_parts.append('</parameters>')
            
            xml_parts.append('</tool>')
        
        return '\n'.join(xml_parts)
    
    def _format_functions_openai(self, functions: List[Dict]) -> str:
        """Format functions in OpenAI tool calling format"""
        tools = []
        for func in functions:
            tool = {
                "type": "function",
                "function": {
                    "name": func.get('name', 'unknown'),
                    "description": func.get('description', 'No description'),
                    "parameters": func.get('parameters', {})
                }
            }
            tools.append(json.dumps(tool, indent=2))
        
        return "Available Tools:\n" + "\n\n".join(tools)
    
    def _format_functions_standard(self, functions: List[Dict]) -> str:
        """Format functions in standard prompt format"""
        func_docs = []
        for func in functions:
            name = func.get('name', 'unknown')
            desc = func.get('description', 'No description')
            params = func.get('parameters', {})
            
            doc = f"Function: {name}\n"
            doc += f"Description: {desc}\n"
            
            if params.get('properties'):
                doc += "Parameters:\n"
                for param_name, param_info in params['properties'].items():
                    param_type = param_info.get('type', 'any')
                    param_desc = param_info.get('description', '')
                    required = param_name in params.get('required', [])
                    req_str = " (required)" if required else " (optional)"
                    doc += f"  - {param_name} ({param_type}){req_str}: {param_desc}\n"
            
            func_docs.append(doc)
        
        return "\n".join(func_docs)
    
    def extract_multi_turn_context(self, case: Dict) -> Dict:
        """Extract multi-turn specific context information"""
        context = {}
        
        # Extract initial configuration
        initial_config = case.get('initial_config', {})
        if initial_config:
            context['initial_config'] = initial_config
            context['environment_setup'] = self._describe_environment(initial_config)
        
        # Extract execution path
        path = case.get('path', [])
        if path:
            context['expected_execution_path'] = path
            context['function_sequence'] = " -> ".join(path)
        
        # Extract involved classes/modules
        involved_classes = case.get('involved_classes', [])
        if involved_classes:
            context['involved_systems'] = involved_classes
        
        return context
    
    def _describe_environment(self, initial_config: Dict) -> str:
        """Describe the environment setup from initial_config"""
        descriptions = []
        
        for system, config in initial_config.items():
            if isinstance(config, dict):
                if 'working_dir' in config:
                    descriptions.append(f"{system}: Working directory at {config['working_dir']}")
                if 'files' in config:
                    file_count = len(config['files']) if isinstance(config['files'], list) else len(config['files'])
                    descriptions.append(f"{system}: {file_count} files available")
                if 'directories' in config:
                    dir_count = len(config['directories']) if isinstance(config['directories'], list) else len(config['directories'])
                    descriptions.append(f"{system}: {dir_count} directories available")
        
        return "; ".join(descriptions) if descriptions else "No specific environment setup"
    
    def extract_conversation_turns(self, question: List) -> List[str]:
        """Extract conversation turns from question structure"""
        turns = []
        
        if not question:
            return ["No question provided"]
        
        for turn in question:
            if isinstance(turn, list):
                turn_content = []
                for msg in turn:
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        turn_content.append(msg.get('content', ''))
                if turn_content:
                    turns.append(" ".join(turn_content))
        
        return turns if turns else ["No valid turns found"]
    
    def create_enhanced_analysis_prompt(self, case: Dict[str, Any], model_name: str = "gpt-4.1") -> str:
        """Create enhanced analysis prompt with all BFCL context"""
        
        case_id = case.get('id', 'unknown')
        question = case.get('question', [[]])
        functions = case.get('function', [])
        
        # Get model configuration
        model_config = self.get_model_config(model_name)
        
        # Extract conversation turns
        conversation_turns = self.extract_conversation_turns(question)
        
        # Extract multi-turn context
        multi_turn_context = self.extract_multi_turn_context(case)
        
        # Format functions according to model requirements
        formatted_functions = self.format_function_docs(functions, model_config)
        
        # Create base system prompt
        base_system_prompt = DEFAULT_SYSTEM_PROMPT_WITHOUT_FUNC_DOC
        
        # Create full system prompt with functions
        if functions:
            full_system_prompt = f"{base_system_prompt}\n\nHere is a list of functions in JSON format that you can invoke.\n{formatted_functions}\n"
        else:
            full_system_prompt = base_system_prompt
        
        # Build conversation context
        conversation_context = ""
        if len(conversation_turns) > 1:
            conversation_context = f"\n**Multi-turn Conversation:**\n"
            for i, turn in enumerate(conversation_turns, 1):
                conversation_context += f"Turn {i}: {turn}\n"
        else:
            conversation_context = f"\n**User Request:** {conversation_turns[0]}\n"
        
        # Build multi-turn context information
        context_info = ""
        if multi_turn_context:
            context_info = f"\n**Multi-turn Context Information:**\n"
            if 'environment_setup' in multi_turn_context:
                context_info += f"- Environment: {multi_turn_context['environment_setup']}\n"
            if 'function_sequence' in multi_turn_context:
                context_info += f"- Expected sequence: {multi_turn_context['function_sequence']}\n"
            if 'involved_systems' in multi_turn_context:
                context_info += f"- Systems involved: {', '.join(multi_turn_context['involved_systems'])}\n"
        
        prompt = f"""
You are analyzing a function-calling benchmark test case for functionality mismatches with complete BFCL context.

## BFCL System Prompt (Used in actual benchmark):
{full_system_prompt}

## Test Case Analysis

**Case ID**: {case_id}
**Task Type**: {case.get('task_name', 'unknown')}
**Model Configuration**: {model_config}

{conversation_context}

{context_info}

**Available Functions**: 
{formatted_functions}

## Enhanced Analysis Task:

Determine if this test case has a **functionality mismatch** problem, considering the complete BFCL execution context.

A functionality mismatch occurs when:

1. **Empty Function List**: No functions are provided but the test expects function calls
2. **Domain Mismatch**: The provided functions cannot fulfill the user's intent 
3. **Missing Core Functionality**: Essential functions for completing the task are absent
4. **Incompatible Parameters**: Functions exist but their parameters don't match requirements
5. **Multi-turn Context Issues**: Multi-turn scenarios lack proper state management or function availability
6. **Environment-Function Mismatch**: Available functions don't match the described environment

## Required Analysis Format:

**VERDICT**: [MISMATCH | NO_MISMATCH | UNCERTAIN]

**MISMATCH_TYPE**: [EMPTY_FUNCTIONS | DOMAIN_MISMATCH | MISSING_FUNCTIONALITY | INCOMPATIBLE | CONTEXT_MISMATCH | N/A]

**REASONING**: 
[2-3 sentences explaining your judgment, considering the complete BFCL context]

**DETAILED_ANALYSIS**:
- Can user intent be fulfilled with available functions?: [YES/NO/PARTIALLY]
- Multi-turn context alignment: [ALIGNED/MISALIGNED/N/A]
- Environment-function compatibility: [COMPATIBLE/INCOMPATIBLE/N/A]
- Missing critical capabilities: [List specific gaps]
- Severity: [CRITICAL/HIGH/MEDIUM/LOW/N/A]

**BFCL_SPECIFIC_ISSUES**:
- System prompt adequacy: [ADEQUATE/INADEQUATE]
- Function documentation quality: [GOOD/POOR/MISSING]
- Test case design problems: [List any issues]

Focus on whether the test case can be solved by ANY model with perfect function-calling capabilities, considering the actual BFCL execution environment.
"""
        
        return prompt.strip()
    
    def load_test_data_with_context(self, task_type: str) -> List[Dict]:
        """Load test cases with additional context information"""
        # Use relative path from current working directory
        data_dir = Path("bfcl_eval/data")
        
        # Load main test cases
        main_file = data_dir / f"BFCL_v3_{task_type}.json"
        test_cases = []
        
        if main_file.exists():
            with open(main_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                # Try JSON array format first
                if content.startswith('['):
                    try:
                        test_cases = json.loads(content)
                    except json.JSONDecodeError:
                        # Fallback to JSONL parsing
                        for line in content.split('\n'):
                            if line.strip():
                                try:
                                    test_cases.append(json.loads(line.strip()))
                                except json.JSONDecodeError:
                                    continue
                else:
                    # JSONL format - each line is a separate JSON object
                    for line in content.split('\n'):
                        if line.strip():
                            try:
                                test_cases.append(json.loads(line.strip()))
                            except json.JSONDecodeError as e:
                                print(f"Warning: Failed to parse line in {main_file}: {e}")
                                continue
        
        # Load possible answers for additional context
        possible_answer_file = data_dir / "possible_answer" / f"{task_type}_possible_answer.json"
        possible_answers = {}
        
        if possible_answer_file.exists():
            try:
                with open(possible_answer_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                    if content.startswith('['):
                        try:
                            answers_list = json.loads(content)
                            for item in answers_list:
                                possible_answers[item.get('id', '')] = item.get('ground_truth', [])
                        except json.JSONDecodeError:
                            # Fallback to JSONL
                            for line in content.split('\n'):
                                if line.strip():
                                    try:
                                        item = json.loads(line.strip())
                                        possible_answers[item.get('id', '')] = item.get('ground_truth', [])
                                    except json.JSONDecodeError:
                                        continue
                    else:
                        # JSONL format
                        for line in content.split('\n'):
                            if line.strip():
                                try:
                                    item = json.loads(line.strip())
                                    possible_answers[item.get('id', '')] = item.get('ground_truth', [])
                                except json.JSONDecodeError:
                                    continue
            except Exception as e:
                print(f"Warning: Could not load possible answers for {task_type}: {e}")
        
        # Merge possible answers into test cases
        for case in test_cases:
            case_id = case.get('id', '')
            if case_id in possible_answers:
                case['ground_truth'] = possible_answers[case_id]
        
        # Apply BFCL's multi-turn function loading logic
        test_cases = self._apply_multi_turn_function_loading(test_cases)
        
        return test_cases
    
    def _apply_multi_turn_function_loading(self, test_cases: List[Dict]) -> List[Dict]:
        """Apply BFCL's multi-turn function loading logic"""
        # Function mapping from BFCL
        MULTI_TURN_FUNC_DOC_FILE_MAPPING = {
            "GorillaFileSystem": "gorilla_file_system.json",
            "MathAPI": "math_api.json", 
            "MessageAPI": "message_api.json",
            "TwitterAPI": "posting_api.json",
            "TicketAPI": "ticket_api.json",
            "TradingBot": "trading_bot.json",
            "TravelAPI": "travel_booking.json",
            "VehicleControlAPI": "vehicle_control.json",
        }
        
        multi_turn_func_doc_path = Path("bfcl_eval/data/multi_turn_func_doc")
        
        for entry in test_cases:
            # Check if this is a multi-turn case (based on BFCL logic)
            if not self._is_multi_turn(entry.get("id", "")):
                continue
                
            # Multi-turn cases need function docs loaded dynamically
            involved_classes = entry.get("involved_classes", [])
            entry["function"] = []
            
            for func_collection in involved_classes:
                if func_collection in MULTI_TURN_FUNC_DOC_FILE_MAPPING:
                    func_doc_file = multi_turn_func_doc_path / MULTI_TURN_FUNC_DOC_FILE_MAPPING[func_collection]
                    if func_doc_file.exists():
                        try:
                            with open(func_doc_file, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                func_doc = []
                                
                                # Try JSON array format first
                                if content.startswith('['):
                                    try:
                                        func_doc = json.loads(content)
                                    except json.JSONDecodeError:
                                        # Fallback to JSONL parsing
                                        for line in content.split('\n'):
                                            if line.strip():
                                                try:
                                                    func_doc.append(json.loads(line.strip()))
                                                except json.JSONDecodeError:
                                                    continue
                                else:
                                    # JSONL format - each line is a separate JSON object
                                    for line in content.split('\n'):
                                        if line.strip():
                                            try:
                                                func_doc.append(json.loads(line.strip()))
                                            except json.JSONDecodeError:
                                                continue
                                
                                entry["function"].extend(func_doc)
                        except Exception as e:
                            print(f"Warning: Failed to load {func_doc_file}: {e}")
        
        return test_cases
    
    def _is_multi_turn(self, test_id: str) -> bool:
        """Check if test case is multi-turn based on ID"""
        return "multi_turn" in test_id
    
    def call_gpt4(self, prompt: str, model_config: Dict) -> str:
        """Call GPT-4.1 API with model-specific configuration"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": model_config.get('max_tokens', 500),
            "temperature": model_config.get('temperature', 0.1)
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
            return f"ERROR: {str(e)}"
    
    def analyze_single_case(self, case: Dict, model_config: Dict) -> Dict:
        """Analyze a single test case with enhanced context"""
        try:
            # Create enhanced analysis prompt
            prompt = self.create_enhanced_analysis_prompt(case, self.model)
            
            # Call API
            analysis = self.call_gpt4(prompt, model_config)
            
            # Parse result
            verdict = "UNKNOWN"
            mismatch_type = "N/A"
            
            # Extract verdict
            if "**VERDICT**:" in analysis:
                verdict_line = analysis.split("**VERDICT**:")[1].split("\n")[0].strip()
                verdict = verdict_line.strip('[]').strip()
            
            # Extract mismatch type
            if "**MISMATCH_TYPE**:" in analysis:
                type_line = analysis.split("**MISMATCH_TYPE**:")[1].split("\n")[0].strip()
                mismatch_type = type_line.strip('[]').strip()
            
            return {
                'case_id': case.get('id', 'unknown'),
                'verdict': verdict,
                'mismatch_type': mismatch_type,
                'analysis': analysis,
                'processing_time': time.time()
            }
            
        except Exception as e:
            return {
                'case_id': case.get('id', 'unknown'),
                'verdict': 'ERROR',
                'mismatch_type': 'ERROR',
                'analysis': f"Error during analysis: {str(e)}",
                'processing_time': time.time()
            }
    
    def analyze_task_parallel(self, task_type: str, batch_size: int = 20) -> Dict:
        """Analyze task type using parallel processing"""
        print(f"\n{'='*80}")
        print(f"Enhanced Analysis: {task_type}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # Load test cases with enhanced context
        test_cases = self.load_test_data_with_context(task_type)
        
        if not test_cases:
            print(f"Warning: No test cases found for {task_type}")
            return {
                'task_type': task_type,
                'total_cases': 0,
                'error': 'No test cases found'
            }
        
        print(f"Found {len(test_cases)} test cases")
        print(f"Using {self.num_workers} parallel workers")
        
        # Get model configuration
        model_config = self.get_model_config(self.model)
        
        # Prepare worker function
        analyze_func = partial(self.analyze_single_case, model_config=model_config)
        
        results = []
        processed = 0
        
        # Process in batches with multiprocessing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all cases
            future_to_case = {}
            
            for i in range(0, len(test_cases), batch_size):
                batch = test_cases[i:i+batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(test_cases) + batch_size - 1) // batch_size
                
                print(f"Submitting batch {batch_num}/{total_batches} (cases {i+1}-{min(i+batch_size, len(test_cases))})")
                
                for case in batch:
                    future = executor.submit(analyze_func, case)
                    future_to_case[future] = case
                
                # Add delay between batch submissions to avoid overwhelming API
                if i + batch_size < len(test_cases):
                    time.sleep(0.5)
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_case):
                case = future_to_case[future]
                try:
                    result = future.result(timeout=60)
                    results.append(result)
                    processed += 1
                    
                    if processed % 50 == 0:
                        print(f"Processed {processed}/{len(test_cases)} cases")
                        
                except Exception as e:
                    print(f"Error processing case {case.get('id', 'unknown')}: {e}")
                    results.append({
                        'case_id': case.get('id', 'unknown'),
                        'verdict': 'ERROR',
                        'mismatch_type': 'ERROR',
                        'analysis': f"Processing error: {str(e)}",
                        'processing_time': time.time()
                    })
                    processed += 1
        
        # Calculate statistics
        mismatch_count = sum(1 for r in results if r['verdict'] == 'MISMATCH')
        mismatch_types = defaultdict(int)
        
        for result in results:
            if result['verdict'] == 'MISMATCH':
                mismatch_types[result['mismatch_type']] += 1
        
        # Create summary
        summary = {
            'task_type': task_type,
            'total_cases': len(test_cases),
            'processed_cases': len(results),
            'mismatch_cases': mismatch_count,
            'mismatch_rate': (mismatch_count / len(results) * 100) if results else 0,
            'mismatch_types': dict(mismatch_types),
            'cases': results,
            'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'processing_config': {
                'model': self.model,
                'workers': self.num_workers,
                'batch_size': batch_size
            }
        }
        
        print(f"Analysis completed: {len(results)} cases processed")
        print(f"Mismatch rate: {summary['mismatch_rate']:.1f}%")
        
        return summary
    
    def run_enhanced_analysis(self, resume_from: Optional[str] = None):
        """Run enhanced analysis on all task types"""
        output_dir = Path("score")
        output_dir.mkdir(exist_ok=True)
        
        # Determine which tasks to run
        if resume_from:
            try:
                resume_index = self.target_tasks.index(resume_from)
                tasks_to_run = self.target_tasks[resume_index:]
                print(f"Resuming from {resume_from} (tasks: {tasks_to_run})")
            except ValueError:
                print(f"Task {resume_from} not found, running all tasks")
                tasks_to_run = self.target_tasks
        else:
            tasks_to_run = self.target_tasks
        
        start_time = datetime.now()
        print(f"Enhanced analysis started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Tasks to analyze: {len(tasks_to_run)}")
        print(f"Multiprocessing workers: {self.num_workers}")
        
        all_results = {}
        
        for idx, task_type in enumerate(tasks_to_run, 1):
            print(f"\n[{idx}/{len(tasks_to_run)}] Processing {task_type}...")
            
            # Run analysis
            result = self.analyze_task_parallel(task_type)
            all_results[task_type] = result
            
            # Save individual results
            task_output = output_dir / f"enhanced_functionality_analysis_{task_type}.json"
            with open(task_output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"Results saved to: {task_output}")
        
        # Save combined results
        combined_output = output_dir / "enhanced_functionality_analysis_all_tasks.json"
        with open(combined_output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        # Calculate and display final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n{'='*80}")
        print("ENHANCED ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        total_cases = 0
        total_mismatches = 0
        
        for task_type, result in all_results.items():
            cases = result.get('processed_cases', 0)
            mismatches = result.get('mismatch_cases', 0)
            rate = result.get('mismatch_rate', 0)
            
            total_cases += cases
            total_mismatches += mismatches
            
            print(f"{task_type:25} - Cases: {cases:4}, Mismatches: {mismatches:4} ({rate:.1f}%)")
        
        if total_cases > 0:
            overall_rate = (total_mismatches / total_cases) * 100
            print(f"\n{'OVERALL':25} - Cases: {total_cases:4}, Mismatches: {total_mismatches:4} ({overall_rate:.1f}%)")
        
        print(f"\nTotal analysis time: {duration}")
        print(f"Analysis completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Combined results: {combined_output}")

if __name__ == "__main__":
    # Configuration
    NUM_WORKERS = min(16, mp.cpu_count())  # Use up to 16 workers or CPU count
    
    print("Enhanced BFCL Functionality Analyzer")
    print(f"Workers: {NUM_WORKERS}")
    print(f"CPU cores: {mp.cpu_count()}")
    
    analyzer = EnhancedFunctionalityAnalyzer(num_workers=NUM_WORKERS)
    
    # Optional: Resume from specific task
    import sys
    resume_task = sys.argv[1] if len(sys.argv) > 1 else None
    
    analyzer.run_enhanced_analysis(resume_from=resume_task)