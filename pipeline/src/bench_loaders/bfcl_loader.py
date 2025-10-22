import json
import os
import sys
from tkinter import N
from typing import Dict, Any, List, Optional
import glob
import importlib.util
import copy
import re
import inspect
import ast

try:
    from src.bench_loaders import BaseLoader  # absolute import for standalone usage
except ImportError:  # pragma: no cover - fallback when package context available
    from . import BaseLoader
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add the BFCL data directory to Python path so bfcl_eval can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'BFCL'))
from src.utils.types import BFCLQuestion, Benchmark

# BFCL Constants (from BFCL evaluation system)
DEFAULT_SYSTEM_PROMPT_WITHOUT_FUNC_DOC = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.
"""

DEFAULT_SYSTEM_PROMPT = (
    DEFAULT_SYSTEM_PROMPT_WITHOUT_FUNC_DOC
    + """
Here is a list of functions in JSON format that you can invoke.\n{functions}\n
"""
)

MAXIMUM_STEP_LIMIT = 20
EXPECTATION_ONLY_CATEGORIES = {"irrelevance", "live_irrelevance", "live_relevance"}

# Class mapping for multi-turn function execution
CLASS_FILE_PATH_MAPPING = {
    "TwitterAPI": "posting_api",
    "GorillaFileSystem": "gorilla_file_system",
    "MathAPI": "math_api",
    "MessageAPI": "message_api",
    "TicketAPI": "ticket_api",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "VehicleControlAPI": "vehicle_control",
}

FILE_NAME_TO_CLASS_NAME_DICT = {
    'gorilla_file_system': 'GorillaFileSystem',
    'math_api': 'MathAPI',
    'message_api': 'MessageAPI',
    'posting_api': 'TwitterAPI',
    'ticket_api': 'TicketAPI',
    'trading_bot': 'TradingBot',
    'travel_booking': 'TravelAPI',
    'vehicle_control': 'VehicleControlAPI'
}

# These classes are stateless and do not require any initial configuration
STATELESS_CLASSES = [
    "MathAPI",
]


class BfclLoader(BaseLoader):
    """
    Berkeley Function Calling Leaderboard (BFCL) data loader
    
    Supports comprehensive BFCL evaluation including:
    - Single-turn and multi-turn function calling
    - Language-specific processing (Python, Java, JavaScript)
    - Model-specific input formatting (FC vs prompting)
    - Multi-turn state management and missing function handling
    """
    def __init__(self):
        self.data_path = "data/BFCL/"
        self.func_doc_path = "data/BFCL/multi_turn_func_doc/"
        self.possible_answer_path = "data/BFCL/possible_answer/"
        
        # Cache for function documentation
        self._func_docs_cache = {}
        # Cache for possible answers
        self._possible_answers_cache = {}
        # Cache for loaded classes and instances
        self._class_cache = {}
        self._instances_cache = {}

    @staticmethod
    def _ordinal(number: int) -> str:
        """Return the ordinal representation of a positive integer (1 -> 1st)."""
        if 10 <= number % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
        return f"{number}{suffix}"

    def _load_function_docs(self) -> None:
        """Load multi-turn function documentation files"""
        if not os.path.exists(self.func_doc_path):
            print(f"Warning: Multi-turn function doc path not found: {self.func_doc_path}")
            return
            
        func_doc_files = glob.glob(os.path.join(self.func_doc_path, "*.json"))
        for file_path in func_doc_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    func_list = []
                    for line in f:
                        if line.strip():
                            func_list.append(json.loads(line.strip()))
                    
                    # Extract class name from filename (e.g., gorilla_file_system.json -> GorillaFileSystem)
                    file_name = os.path.basename(file_path).replace('.json', '')
                    
                    class_name = FILE_NAME_TO_CLASS_NAME_DICT[file_name] 
                    self._func_docs_cache[class_name] = func_list
            except Exception as e:
                print(f"Error loading function docs from {file_path}: {e}")
    
    def _load_possible_answers(self) -> None:
        """Load possible answer files"""
        if not os.path.exists(self.possible_answer_path):
            print(f"Warning: Possible answer path not found: {self.possible_answer_path}")
            return
            
        answer_files = glob.glob(os.path.join(self.possible_answer_path, "*.json"))
        for file_path in answer_files:
            try:
                filename = os.path.basename(file_path)
                with open(file_path, 'r', encoding='utf-8') as f:
                    answers = {}
                    for line in f:
                        if line.strip():
                            answer_data = json.loads(line.strip())
                            answers[answer_data['id']] = answer_data.get('ground_truth', [])
                    self._possible_answers_cache[filename] = answers
            except Exception as e:
                print(f"Error loading possible answers from {file_path}: {e}")

    def _load_class(self, class_name: str):
        """Load a class dynamically from the data/BFCL/multi_turn_eval/func_source_code directory"""
        if class_name in self._class_cache:
            return self._class_cache[class_name]

        if class_name not in CLASS_FILE_PATH_MAPPING:
            raise ValueError(f"Unknown class: {class_name}")

        module_name = CLASS_FILE_PATH_MAPPING[class_name]
        module_path = os.path.join(self.data_path, "bfcl_eval", "eval_checker", "multi_turn_eval", "func_source_code", f"{module_name}.py")

        if not os.path.exists(module_path):
            raise FileNotFoundError(f"Module file not found: {module_path}")

        # Load module dynamically
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec for {module_name}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the class
        class_obj = getattr(module, class_name)
        self._class_cache[class_name] = class_obj
        return class_obj

    def _initialize_instances(self, involved_classes: List[str], initial_config: Dict, test_entry_id: str) -> Dict:
        """Initialize class instances with initial configuration"""
        instances = {}

        for class_name in involved_classes:
            instance_key = f"{test_entry_id}_{class_name}"

            if instance_key in self._instances_cache:
                instances[class_name] = self._instances_cache[instance_key]
                continue

            # Load and instantiate the class
            class_obj = self._load_class(class_name)
            instance = class_obj()

            # Load scenario if not stateless
            if class_name not in STATELESS_CLASSES:
                class_initial_config = initial_config.get(class_name, {})
                if hasattr(instance, '_load_scenario'):
                    instance._load_scenario(copy.deepcopy(class_initial_config))

            instances[class_name] = instance
            self._instances_cache[instance_key] = instance

        return instances

    def _load_default_state(self, class_name: str):
        if class_name not in CLASS_FILE_PATH_MAPPING:
            return None
        if class_name == "GorillaFileSystem":
            return None

        module_name = CLASS_FILE_PATH_MAPPING[class_name]
        module_path = os.path.join(self.data_path, "bfcl_eval", "eval_checker", "multi_turn_eval", "func_source_code", f"{module_name}.py")

        if not os.path.exists(module_path):
            return None

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            return None

        return getattr(module, "DEFAULT_STATE", None)

    def _format_state_lines(self, class_name: str, state) -> List[str]:
        lines: List[str] = []

        lines.append(f"#### {class_name}\n")

        assert isinstance(state, dict)
        for key, value in state.items():
            sanitized = self._serialize_tool_result(value)
            lines.append(f"* {key}: {sanitized}")
        lines.append("")

        return lines

    def _serialize_tool_result(self, value):
        if hasattr(value, "_mpf_"):
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

        if isinstance(value, dict):
            return {k: self._serialize_tool_result(v) for k, v in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [self._serialize_tool_result(v) for v in value]

        if isinstance(value, str):
            stripped = value.strip()
            if stripped and ((stripped[0] == '{' and stripped[-1] == '}') or (stripped[0] == '[' and stripped[-1] == ']')):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    return value
                else:
                    return self._serialize_tool_result(parsed)
            return value

        return value

    def _get_method_mapping(self, instances: Dict) -> Dict[str, str]:
        """Create mapping of method names to instance names"""
        method_mapping = {}

        for class_name, instance in instances.items():
            for method_name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
                if not method_name.startswith("_"):
                    method_mapping[method_name] = class_name

        return method_mapping

    def _parse_function_call(self, func_call: str) -> tuple[str, Dict]:
        """Parse function call string using AST parsing"""
        try:
            # Parse the function call as a Python expression using AST
            tree = ast.parse(func_call.strip(), mode='eval')
            call_node = tree.body

            if not isinstance(call_node, ast.Call):
                raise ValueError(f"Not a function call: {func_call}")

            # Extract function name
            if isinstance(call_node.func, ast.Name):
                func_name = call_node.func.id
            else:
                raise ValueError(f"Complex function names not supported: {func_call}")

            # Extract arguments
            args = {}
            positional_count = 0

            # Handle positional arguments
            for arg in call_node.args:
                value = ast.literal_eval(arg)
                args[f'arg_{positional_count}'] = value
                positional_count += 1

            # Handle keyword arguments
            for keyword in call_node.keywords:
                if keyword.arg is None:  # **kwargs
                    raise ValueError("**kwargs not supported")
                value = ast.literal_eval(keyword.value)
                args[keyword.arg] = value

            return func_name, args

        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Failed to parse function call '{func_call}': {e}")

    def _call_tool_with_state(self, func_call: str, instances: Dict, method_mapping: Dict) -> str:
        """Execute a function call with current state and return the result"""
        func_name, args = self._parse_function_call(func_call)

        if func_name not in method_mapping:
            raise ValueError(f"Error: Function {func_name} not found")

        class_name = method_mapping[func_name]
        instance = instances[class_name]
        method = getattr(instance, func_name)

        # Handle positional arguments by mapping to parameter names
        if any(key.startswith('arg_') for key in args.keys()):
            # Get function signature to map positional args to parameter names
            sig = inspect.signature(method)
            param_names = list(sig.parameters.keys())

            # Convert arg_0, arg_1, etc. to actual parameter names
            converted_args = {}

            # First, add all keyword arguments
            for key, value in args.items():
                if not key.startswith('arg_'):
                    converted_args[key] = value

            # Then, add positional arguments
            for key, value in args.items():
                if key.startswith('arg_'):
                    arg_index = int(key.split('_')[1])
                    if arg_index < len(param_names):
                        param_name = param_names[arg_index]
                        # Don't override if keyword argument already provided
                        if param_name not in converted_args:
                            converted_args[param_name] = value
                    else:
                        # Too many arguments
                        raise ValueError(f"Too many arguments for function {func_name}")
            args = converted_args

        # Call the method with arguments
        result = method(**args)

        if result is None:
            return "null"

        sanitized = self._serialize_tool_result(result)
        return sanitized



    def _get_language_specific_hint(self, test_category: str) -> str:
        """Get language-specific hint based on test category"""
        if test_category == "java":
            return " Note that the provided function is in Java 8 SDK syntax."
        elif test_category == "javascript":
            return " Note that the provided function is in JavaScript syntax."
        else:
            return " Note that the provided function is in Python 3 syntax."
    
    def _add_language_specific_hints(self, function_list: List[Dict], test_category: str) -> List[Dict]:
        """Add language-specific hints to function descriptions"""
        if not function_list or not test_category:
            return function_list
        
        # Create a deep copy to avoid modifying original
        processed_functions = []
        language_hint = self._get_language_specific_hint(test_category)
        
        for func in function_list:
            func_copy = func.copy()
            if 'description' in func_copy:
                func_copy['description'] = func_copy['description'] + language_hint
            processed_functions.append(func_copy)
        
        return processed_functions
    
    def get_system_prompt(self, function_list: List[Dict], test_category: str = "", include_functions: bool = True) -> str:
        """Generate system prompt for BFCL evaluation with language-specific hints"""
        if not include_functions or not function_list:
            return DEFAULT_SYSTEM_PROMPT_WITHOUT_FUNC_DOC
        
        # Apply language-specific processing to function list
        processed_functions = self._add_language_specific_hints(function_list, test_category)
        functions_str = json.dumps(processed_functions, indent=2)
        return DEFAULT_SYSTEM_PROMPT.format(functions=functions_str)
    
    def get_model_specific_inputs(self, question: BFCLQuestion, model_type: str = "prompting") -> Dict[str, Any]:
        """Get model-specific inputs for BFCL evaluation"""
        test_category = self._determine_category_and_subcategory(question.question_id)[0]
        
        if model_type == "function_calling":
            # For models with native function calling support
            processed_functions = self._add_language_specific_hints(question.available_function_list, test_category)
            
            inputs = {
                "messages": question.gt_conv_traj[0] if question.gt_conv_traj else [],
                "functions": processed_functions,
                "function_call": "auto",
                "temperature": 0.001  # BFCL default temperature
            }
            
            # Multi-turn specific fields
            if question.is_multi_turn:
                inputs["initial_config"] = question.initial_config
                inputs["involved_classes"] = question.involved_classes
                inputs["max_turn_limit"] = question.max_turn_limit
                if question.missed_function:
                    inputs["missed_function"] = question.missed_function
            
            return inputs
        else:
            # For prompting models - apply language-specific processing
            system_prompt = question.system_prompt or self.get_system_prompt(
                question.available_function_list, test_category
            )
            
            system_message = {
                "role": "system", 
                "content": system_prompt
            }
            user_messages = question.gt_conv_traj[0] if question.gt_conv_traj else []
            
            inputs = {
                "messages": [system_message] + user_messages,
                "temperature": 0.001
            }
            
            # Multi-turn specific fields
            if question.is_multi_turn:
                inputs["initial_config"] = question.initial_config
                inputs["involved_classes"] = question.involved_classes
                inputs["max_turn_limit"] = question.max_turn_limit
                inputs["exclude_state_log"] = question.exclude_state_log
                if question.missed_function:
                    inputs["missed_function"] = question.missed_function
            
            return inputs
    
    def convert_to_openai_tools(self, function_list: List[Dict]) -> List[Dict]:
        """Convert BFCL function format to OpenAI tools format"""
        tools = []
        for func in function_list:
            tool = {
                "type": "function",
                "function": {
                    "name": func["name"],
                    "description": func["description"],
                    "parameters": func["parameters"]
                }
            }
            tools.append(tool)
        return tools
    
    def get_evaluation_config(self, question: BFCLQuestion) -> Dict[str, Any]:
        """Get evaluation configuration for BFCL"""
        return {
            "max_step_limit": question.max_turn_limit,
            "exclude_state_log": question.exclude_state_log,
            "is_multi_turn": question.is_multi_turn,
            "category": question.category,
            "subcategory": question.subcategory,
            "is_live_data": question.is_live_data,
            "ground_truth": question.ground_truth,
            "expected_path": question.expected_path,
            "evaluation_ready": question.meta.get("evaluation_ready", True) if question.meta else True
        }
    
    def _get_missing_function_info(self, line: dict) -> Optional[Dict[str, List[str]]]:
        """Extract missing function information for multi-turn miss scenarios"""
        return line.get('missed_function', None)

    def _extract_instruction_from_question(self, question_data: List[List[Dict]]) -> str:
        """Extract user instruction from question data structure"""
        if not question_data or not question_data[0]:
            return ""

        # Single-turn case: question[0][0]['content']
        if len(question_data) == 1:
            first_turn = question_data[0]
            if first_turn and isinstance(first_turn[0], dict):
                return first_turn[0].get('content', '')

        # Multi-turn case: combine all user messages
        instructions = []
        for turn_idx, turn in enumerate(question_data):
            if turn and isinstance(turn[0], dict):
                content = turn[0].get('content', '')
                if content:
                    # Add turn number for clarity in multi-turn scenarios
                    instructions.append(f"Turn {turn_idx + 1}: {content}")

        return " | ".join(instructions) if instructions else ""

    def _align_ground_truth_to_turns(self, question_data: List[List[Dict]], ground_truth: List[Any]) -> List[List[Any]]:
        """Align ground truth tool call groups with the number of user turns.

        Some BFCL entries (especially multi-turn stateful ones) provide tool call
        sequences where the grouping does not perfectly align with the number of
        user turns—for example, multiple consecutive tool call batches for a
        single user request. This helper flattens and repartitions the tool call
        groups so we always have one list per user turn while preserving order.
        """
        if not question_data or not ground_truth:
            return ground_truth or []

        num_turns = len(question_data)

        # Normalise each ground truth entry into a list for easier merging
        normalised = []
        for entry in ground_truth:
            if isinstance(entry, list):
                normalised.append(entry)
            else:
                normalised.append([entry])

        if len(normalised) == num_turns:
            return normalised

        # If there are fewer tool call groups than turns, pad with empty lists
        if len(normalised) < num_turns:
            normalised.extend([[] for _ in range(num_turns - len(normalised))])
            return normalised

        aligned: List[List[Any]] = []
        idx = 0
        for turn_idx in range(num_turns):
            remaining_turns = num_turns - turn_idx
            remaining_groups = len(normalised) - idx
            # Ensure we leave at least one group for each remaining turn
            take = max(1, remaining_groups - (remaining_turns - 1))

            merged_group: List[Any] = []
            for _ in range(take):
                merged_group.extend(normalised[idx])
                idx += 1
            aligned.append(merged_group)

        return aligned

    def _create_interleaved_trajectory(self, question_data: List[List[Dict]], ground_truth: List[Any], missed_function_dict: Optional[Dict], initial_config: Optional[Dict] = None, involved_classes: Optional[List[str]] = None, test_entry_id: str = "unknown") -> List[Dict]:
        """Create interleaved trajectory with user questions and agent tool calls, including function execution results"""
        if not question_data or not ground_truth:
            return []

        if len(question_data) != len(ground_truth):
            ground_truth = self._align_ground_truth_to_turns(question_data, ground_truth)

        if len(question_data) != len(ground_truth):
            raise ValueError(
                f"Ground truth length mismatch for {test_entry_id}: "
                f"{len(question_data)} turns vs {len(ground_truth)} tool groups"
            )
        trajectory = []

        # Initialize instances if we have multi-turn data with state
        instances = None
        method_mapping = None
        if initial_config and involved_classes:
            instances = self._initialize_instances(involved_classes, initial_config, test_entry_id)
            method_mapping = self._get_method_mapping(instances)

        for i in range(len(question_data)):
            # Add user message
            if missed_function_dict and str(i) in missed_function_dict and question_data[i] == []:
                # user provides missed function.
                trajectory.append({
                    "role": "user",
                    "content": f"Now you can use previously missed function(s) {missed_function_dict[str(i)]}"
                })
            else:
                trajectory.append({
                    "role": "user",
                    "content": question_data[i]
                })

            # Normalize current turn tool calls to a list
            current_turn_calls = ground_truth[i]
            if not isinstance(current_turn_calls, list):
                current_turn_calls = [current_turn_calls]

            if not current_turn_calls:
                trajectory.append({"role": "assistant", "tool_calls": []})
                continue

            for func_call in current_turn_calls:
                trajectory.append({
                    "role": "assistant",
                    "tool_calls": [func_call]
                })

                if instances and method_mapping and isinstance(func_call, str) and func_call.strip():
                    try:
                        response = self._call_tool_with_state(func_call, instances, method_mapping)
                    except Exception as exec_error:
                        response = f"Execution error: {exec_error}"

                    trajectory.append({
                        "role": "tool",
                        "tool_responses": [{
                            "function_call": func_call,
                            "response": response
                        }]
                    })

        return trajectory
    
    def _determine_category_and_subcategory(self, question_id: str, filename: str = "") -> tuple[str, str]:
        """Determine category and subcategory from question_id and filename"""
        # Handle live data
        if 'live_' in question_id or 'live' in filename:
            is_live = True
            if question_id.startswith('live_simple_'):
                return 'live_simple', 'simple'
            elif question_id.startswith('live_multiple_'):
                return 'live_multiple', 'multiple'
            elif question_id.startswith('live_parallel_'):
                if 'multiple' in question_id:
                    return 'live_parallel_multiple', 'parallel_multiple'
                else:
                    return 'live_parallel', 'parallel'
            elif question_id.startswith('live_irrelevance_'):
                return 'live_irrelevance', 'irrelevance'
            elif question_id.startswith('live_relevance_'):
                return 'live_relevance', 'relevance'
        
        # Handle regular data
        if question_id.startswith('simple_'):
            return 'simple', 'single_turn'
        elif question_id.startswith('multiple_'):
            return 'multiple', 'single_turn'
        elif question_id.startswith('parallel_'):
            if 'multiple' in question_id:
                return 'parallel_multiple', 'single_turn'
            else:
                return 'parallel', 'single_turn'
        elif question_id.startswith('multi_turn_'):
            if 'base' in question_id:
                return 'multi_turn_base', 'multi_turn'
            elif 'long_context' in question_id:
                return 'multi_turn_long_context', 'multi_turn'
            elif 'miss_func' in question_id:
                return 'multi_turn_miss_func', 'multi_turn'
            elif 'miss_param' in question_id:
                return 'multi_turn_miss_param', 'multi_turn'
            else:
                return 'multi_turn', 'multi_turn'
        elif question_id.startswith('irrelevance_'):
            return 'irrelevance', 'single_turn'
        elif question_id.startswith('java_'):
            return 'java', 'language_specific'
        elif question_id.startswith('javascript_'):
            return 'javascript', 'language_specific'
        else:
            return 'unknown', 'unknown'
    
    def _get_functions_for_multi_turn(self, involved_classes: List[str]) -> List[Dict]:
        """Get function definitions for multi-turn questions based on involved classes"""
        if not involved_classes or not self._func_docs_cache:
            return []
        
        all_functions = []
        for class_name in involved_classes:
            if class_name not in self._func_docs_cache:
                print(f"{class_name} not in doc")
            all_functions.extend(self._func_docs_cache[class_name])
        
        return all_functions
    
    def _get_ground_truth(self, question_id: str, filename: str) -> Optional[Any]:
        """Get possible answer for a question"""
        # Try to find in possible answers cache
        for answer_filename, answers in self._possible_answers_cache.items():
            # Direct filename match (e.g., BFCL_v3_simple.json matches BFCL_v3_simple.json)
            base_filename = filename.replace('.json', '')
            answer_base = answer_filename.replace('.json', '')
            
            if base_filename == answer_base:
                return answers.get(question_id, None)
            
            # Partial matching for similar categories
            if base_filename in answer_filename or answer_base in filename:
                return answers.get(question_id, None)
                
        return None
    
    def _format_line(self, line: dict, filename: str) -> BFCLQuestion:
        """Format a single line from BFCL dataset into BFCLQuestion"""
        question_id = line.get('id', 'unknown')
        question_data = line.get("question", [])

        # Determine category and subcategory early so we can infer multi-turn status reliably
        category, subcategory = self._determine_category_and_subcategory(question_id, filename)

        # Extract instruction
        instruction = self._extract_instruction_from_question(question_data)

        # Determine if multi-turn; some multi-turn files contain a single user turn in the question list,
        # but still require multi-turn handling (stateful execution, initial config, etc.).
        is_multi_turn = (subcategory == 'multi_turn') or (len(question_data) > 1)
        
        # Multi-turn specific fields
        initial_config = line.get('initial_config', None) if is_multi_turn else None
        expected_path = line.get('path', None) if is_multi_turn else None
        involved_classes = line.get('involved_classes', None) if is_multi_turn else None
        
        # Get function list
        function_list = line.get("function", [])
        
        # For multi-turn questions, if no functions are provided, try to get from func docs
        if is_multi_turn and not function_list and involved_classes:
            function_list = self._get_functions_for_multi_turn(involved_classes)
        
        is_live_data = 'live' in category
        
        # Get possible answer
        ground_truth = self._get_ground_truth(question_id, filename)
        
        # Expectation policy where there is no ground truth for these categories
        expect_call = None
        expectation_text = None
        if category in EXPECTATION_ONLY_CATEGORIES and ground_truth is None:
            if category == "live_relevance":
                expect_call = True
                expectation_text = (
                    "BFCL relevance policy: Expect at least one valid function call for the task in this turn."
                )
            else:  # irrelevance or live_irrelevance
                expect_call = False
                expectation_text = (
                    "BFCL irrelevance policy: Expect no valid or relevant function calls for the task in this turn."
                )

            # Option B (prompt-visible): prepend to the instruction so prompts see it
            if instruction:
                instruction = expectation_text + "\n\n" + instruction
            else:
                instruction = expectation_text

        # Handle missing functions for multi-turn scenarios
        missed_function_dict = self._get_missing_function_info(line)

        missed_function = ""
        if missed_function_dict:
            missed_lines = ["### Missed Functions"]
            for turn_key, funcs in missed_function_dict.items():
                try:
                    human_turn = int(turn_key) + 1
                    descriptor = f"{self._ordinal(human_turn)} user message"
                except (TypeError, ValueError):
                    descriptor = f"turn {turn_key}"

                missed_lines.append(
                    f"The following function(s) will be provided with the {descriptor}; the agent should not use them before it:\n{funcs}"
                )
            missed_function = "\n".join(missed_lines)

        # Prepare default state summary for prompt readability
        default_state_lines = []

        # if question_id == "multi_turn_miss_func_62":
            # import pdb; pdb.set_trace()
        if involved_classes:
            for class_name in involved_classes:
                raw_state = self._load_default_state(class_name)
                if raw_state is None:
                    continue

                default_state_lines.extend(
                    self._format_state_lines(class_name, raw_state)
                )

        default_states = "\n".join(default_state_lines) if default_state_lines else "* (no tool default states provided)"

        # Create interleaved trajectory combining user questions and agent tool calls
        interleaved_trajectory = self._create_interleaved_trajectory(
            question_data,
            ground_truth or [],
            missed_function_dict,
            initial_config,
            involved_classes,
            question_id
        )

        # Generate system prompt with language-specific hints
        system_prompt = self.get_system_prompt(function_list, category)

        # Set exclude state log for certain categories
        exclude_state_log = category in ['irrelevance', 'live_irrelevance']

        # Construct a description of the initial pwd
        initial_pwd_description = ""
        if initial_config:
            gorilla_fs_config = initial_config.get('GorillaFileSystem', None)
            if gorilla_fs_config:
                # assert len(gorilla_fs_config["root"]) == 1
                initial_pwd = list(gorilla_fs_config["root"].keys())[0]
                initial_pwd_description = f"Note that root directory in the file system is `{initial_pwd}`, not `root`. This is the initial working directory of the system."


        # Build meta (attach expect_call if set)
        meta_payload = {
            'file_source': filename,
            'original_category': self._determine_category_and_subcategory(question_id, filename)[0],
            'has_ground_truth': ground_truth is not None,
            'num_functions': len(function_list),
            'num_turns': len(question_data) if question_data else 0,
            'has_missing_functions': missed_function_dict is not None,
            'evaluation_ready': True
        }
        if expect_call is not None:
            meta_payload['expect_call'] = expect_call

        return BFCLQuestion(
            question_id=question_id,
            task_name=category,
            instruction=instruction,
            gt_conv_traj=interleaved_trajectory,
            available_function_list=function_list,
            benchmark=Benchmark.BFCL,
            initial_config=initial_config,
            initial_pwd_description=initial_pwd_description,
            expected_path=expected_path,
            involved_classes=involved_classes,
            is_multi_turn=is_multi_turn,
            ground_truth=ground_truth,
            category=category,
            subcategory=subcategory,
            is_live_data=is_live_data,
            missed_function=missed_function,
            system_prompt=system_prompt,
            max_turn_limit=MAXIMUM_STEP_LIMIT,
            exclude_state_log=exclude_state_log,
            meta=meta_payload,
            default_states=default_states)
    
    def load_questions(self) -> List[BFCLQuestion]:
        """Load questions from all BFCL dataset files"""
        questions = []
        
        # Load function documentation and possible answers first
        self._load_function_docs()
        self._load_possible_answers()
        
        # Find all BFCL JSON files in the data directory
        json_files = glob.glob(os.path.join(self.data_path, "BFCL_v3_*.json"))
        
        if not json_files:
            print(f"Warning: No BFCL data files found in {self.data_path}")
            return []
        
        print(f"Found {len(json_files)} BFCL data files")
        print(f"Loaded function docs for {len(self._func_docs_cache)} classes")
        print(f"Loaded possible answers for {len(self._possible_answers_cache)} files")
        
        for file_path in json_files:
            filename = os.path.basename(file_path)
            file_questions = 0
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Each line is a JSON object
                    for line_num, line in enumerate(f, 1):
                        try:
                            if line.strip():  # Skip empty lines
                                line_data = json.loads(line.strip())
                                questions.append(self._format_line(line_data, filename))
                                file_questions += 1
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line {line_num} in {filename}: {e}")
                        except Exception as e:
                            print(f"Error processing line {line_num} in {filename}: {e}")
                            
                print(f"Loaded {file_questions} questions from {filename}")
                            
            except FileNotFoundError:
                print(f"Warning: File not found: {file_path}")
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
        
        # Print statistics
        categories = {}
        multi_turn_count = 0
        with_ground_truth = 0
        
        for q in questions:
            categories[q.category] = categories.get(q.category, 0) + 1
            if q.is_multi_turn:
                multi_turn_count += 1
            if q.ground_truth:
                with_ground_truth += 1
        
        print(f"\n=== BFCL Loader Statistics ===")
        print(f"Total questions loaded: {len(questions)}")
        print(f"Multi-turn questions: {multi_turn_count}")
        print(f"Questions with ground truth: {with_ground_truth}")
        print(f"Categories: {dict(sorted(categories.items()))}")
        
        return questions
