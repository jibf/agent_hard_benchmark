import json
import os
import sys
from typing import Dict, Any, List
from . import BaseLoader
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.types import TauBenchQuestion, Benchmark



class TauBenchLoader(BaseLoader):
    """Formatter for tau-bench dataset"""
    
    def __init__(self):
        # Add the project root to path for imports
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    def load_tau_bench_data(self, domain: str) -> Dict[str, Any]:
        """Load tau-bench environment data (users, flights/products, reservations)"""
        domain_path = f"data/tau-bench-envs/{domain}"
        data_path = os.path.join(domain_path, "data")
        
        env_data = {}
        for file_name in ["users.json", "flights.json", "products.json", "reservations.json", "orders.json"]:
            file_path = os.path.join(data_path, file_name)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    env_data[file_name.replace('.json', '')] = json.load(f)
        
        return env_data
    
    def load_tau_bench_tools(self, domain: str) -> List[Dict[str, Any]]:
        """Load tool schemas for tau-bench domain using get_info() methods"""
        try:
            return self._extract_tool_schemas_from_domain(domain)
        except Exception as e:
            print(f"Failed to extract schemas using get_info(), falling back to manual schemas: {e}")
            manual_schemas = self._create_manual_tool_schemas()
            return manual_schemas.get(domain, [])
    
    def format_sample(self, sample: Dict[str, Any], domain: str = None, env_data: Dict[str, Any] = None, sample_id: str = None) -> TauBenchQuestion:
        """Format tau-bench task to standard evaluation format"""
        if domain is None:
            raise ValueError("domain parameter is required for tau-bench formatting")
        
        if env_data is None:
            env_data = self.load_tau_bench_data(domain)
        
        # Extract task components
        user_id = sample.get('user_id')
        instruction = sample.get('instruction', '')
        actions = sample.get('actions', [])
        outputs = sample.get('outputs', [])
        
        # Build user prompt with context
        user_prompt = self._build_contextual_user_prompt(instruction, user_id, env_data, domain)
        
        # Convert actions to conversation format with real tool execution
        conversations = self._convert_actions_to_conversations(instruction, actions, domain, env_data)
        
        # Get function schemas
        functions = self.load_tau_bench_tools(domain)
        
        return TauBenchQuestion(
            question_id=sample_id or f"{domain}-{user_id}",
            instruction=user_prompt,
            gt_conv_traj=conversations,
            available_function_list=functions,
            benchmark=Benchmark.TAU_BENCH,
            agent_system_prompt=instruction,
            meta={
                'tau_bench_context': {
                    'user_id': user_id,
                    'domain': domain,
                    'expected_outputs': outputs,
                    'env_data': env_data
                },
            }
        )
    
    def extract_conversation(self, question_sample: dict) -> tuple:
        # For already formatted tau-bench data, extract components directly
        if 'meta' in question_sample and 'tau_bench_context' in question_sample['meta']:
            tau_context = question_sample['meta']['tau_bench_context']
            domain = tau_context.get('domain')
            user_prompt = question_sample['meta'].get('user_prompt', '')
            conversations = question_sample.get('conversations', [])
            available_function_list = question_sample.get('available_function_list', [])
            return user_prompt, conversations, available_function_list
        
        # For raw task data, use format_sample
        domain = question_sample.get('domain')  # Domain might be passed separately
        formatted_sample = self.format_sample(question_sample, domain=domain)
        return formatted_sample.meta['user_prompt'], formatted_sample.conversations, formatted_sample.available_function_list

    def _build_contextual_user_prompt(self, instruction: str, user_id: str, env_data: Dict[str, Any], domain: str = None) -> str:
        """Build a contextual user prompt that includes system rules and relevant background information"""
        
        context_parts = []
        
        # Add system prompt/rules for tau-bench
        if domain:
            # Add wiki.md content if it exists
            try:
                wiki_file = f"data/tau-bench-envs/{domain}/wiki.md"
                if os.path.exists(wiki_file):
                    with open(wiki_file, 'r', encoding='utf-8') as f:
                        wiki_content = f.read().strip()
                    if wiki_content:
                        context_parts.append("[System Policy and Rules]")
                        context_parts.append(wiki_content)
                        context_parts.append("")
            except Exception as e:
                print(f"Could not load wiki for {domain}: {e}")
            
            # Add rules.py content if it exists (for backward compatibility)
            try:
                rules_file = f"data/tau-bench-envs/{domain}/rules.py"
                if os.path.exists(rules_file):
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(f"{domain}_rules", rules_file)
                    rules_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(rules_module)
                    
                    if hasattr(rules_module, 'RULES'):
                        context_parts.append("[System Rules for AI Model]")
                        for rule in rules_module.RULES:
                            context_parts.append(f"- {rule}")
                        context_parts.append("")
            except Exception as e:
                print(f"Could not load rules for {domain}: {e}")
        
        # Add the scenario instruction
        context_parts.append("[AI Model Instruction]")
        context_parts.append(instruction)
        
        # Add user context if available
        if user_id and 'users' in env_data:
            user_data = env_data['users'].get(user_id, {})
            if user_data:
                context_parts.append(f"\n[User Context]")
                name = user_data.get('name', {})
                if name:
                    context_parts.append(f"Name: {name.get('first_name', '')} {name.get('last_name', '')}")
                
                if 'membership' in user_data:
                    context_parts.append(f"Membership Status: {user_data['membership']}")
                
                if 'reservations' in user_data and user_data['reservations']:
                    context_parts.append(f"Existing Reservations: {', '.join(user_data['reservations'])}")
                
                if 'payment_methods' in user_data:
                    payment_count = len(user_data['payment_methods'])
                    context_parts.append(f"Available Payment Methods: {payment_count}")
        
        return '\n'.join(context_parts)
    
    def _convert_actions_to_conversations(self, instruction: str, actions: List[Dict[str, Any]], domain: str = None, env_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Convert tau-bench actions to conversation format with real tool execution results"""
        
        conversations = []
        # Create a mutable copy of env_data to track state changes
        current_env_data = json.loads(json.dumps(env_data)) if env_data else {}
        
        # Convert each action to assistant message with tool call + real observation
        for i, action in enumerate(actions):
            # Assistant message with tool call
            conversations.append({
                "role": "assistant", 
                "function_call": [
                    {
                        "name": action["name"],
                        "arguments": action.get("arguments", {})
                    }
                ]
            })
            
            # Execute the actual tool to get real observation with updated env_data
            try:
                result, updated_env_data = self._execute_tool_with_state(action["name"], action.get("arguments", {}), domain, current_env_data)
                current_env_data = updated_env_data
                observation_content = result
            except Exception as e:
                observation_content = f"Error executing {action['name']}: {str(e)}"
            
            conversations.append({
                "role": "observation",
                "content": [observation_content]
            })
        
        return conversations
    
    def process_tau_bench_tasks(self, domain: str) -> List[TauBenchQuestion]:
        """Process tau-bench tasks and return formatted questions"""
        # Load environment data
        env_data = self.load_tau_bench_data(domain)
        
        # Load tasks directly from the tasks.py file
        tasks_file = f"data/tau-bench-envs/{domain}/tasks.py"
        
        try:
            # Read and execute the tasks.py file to get the tasks list
            import importlib.util
            spec = importlib.util.spec_from_file_location(f"{domain}_tasks", tasks_file)
            tasks_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tasks_module)
            tasks = tasks_module.tasks
        except Exception as e:
            print(f"Could not load tasks from {tasks_file}: {e}")
            return []
        
        # Convert each task
        converted_tasks = []
        for i, task in enumerate(tasks):
            try:
                formatted_task = self.format_sample(task, domain, env_data, sample_id=f"{domain}-{i}")
                converted_tasks.append(formatted_task)
            except Exception as e:
                print(f"Error converting task {i}: {e}")
                continue
        
        print(f"Converted {len(converted_tasks)} tau-bench {domain} tasks")
        return converted_tasks

    def _extract_tool_schemas_from_domain(self, domain: str) -> List[Dict[str, Any]]:
        """Extract tool schemas from tau-bench tools using get_info() method"""
        tools_path = f"data/tau-bench-envs/{domain}/tools"
        schemas = []
        
        if not os.path.exists(tools_path):
            raise Exception(f"Tools path {tools_path} not found")
        
        # Add tau-bench-envs to Python path so we can import the Tool base class
        tau_bench_path = "data/tau-bench-envs"
        if tau_bench_path not in sys.path:
            sys.path.insert(0, tau_bench_path)
        
        import importlib.util
        
        for file_name in os.listdir(tools_path):
            if file_name.endswith('.py') and file_name != '__init__.py':
                tool_file = os.path.join(tools_path, file_name)
                tool_name = file_name.replace('.py', '')
                
                try:
                    # Read the file and fix the import path
                    with open(tool_file, 'r', encoding='utf-8') as f:
                        tool_content = f.read()
                    
                    # Replace tau_bench.envs.tool with tool for local import
                    tool_content = tool_content.replace('from tau_bench.envs.tool import Tool', 'from tool import Tool')
                    
                    # Load the modified module
                    spec = importlib.util.spec_from_file_location(f"{domain}_{tool_name}", tool_file)
                    tool_module = importlib.util.module_from_spec(spec)
                    exec(tool_content, tool_module.__dict__)
                    sys.modules[spec.name] = tool_module
                    
                    # Find the tool class (should be the capitalized version)
                    class_name = ''.join(word.capitalize() for word in tool_name.split('_'))
                    if hasattr(tool_module, class_name):
                        tool_class = getattr(tool_module, class_name)
                        if hasattr(tool_class, 'get_info'):
                            schema = tool_class.get_info()
                            schemas.append(schema)
                        else:
                            print(f"Warning: {class_name} does not have get_info() method")
                    else:
                        print(f"Warning: Could not find class {class_name} in {file_name}")
                        
                except Exception as e:
                    print(f"Error loading tool {tool_name}: {e}")
                    continue
        
        if not schemas:
            raise Exception("No schemas were successfully extracted")
        
        return schemas
    
    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any], domain: str, env_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tau-bench tool and return the result (legacy method)"""
        result, _ = self._execute_tool_with_state(tool_name, arguments, domain, env_data)
        return result
    
    def _execute_tool_with_state(self, tool_name: str, arguments: Dict[str, Any], domain: str, env_data: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Execute a tau-bench tool and return both result and updated env_data"""
        tools_path = f"data/tau-bench-envs/{domain}/tools"
        
        # Add tau-bench-envs to Python path
        tau_bench_path = "data/tau-bench-envs"
        if tau_bench_path not in sys.path:
            sys.path.insert(0, tau_bench_path)
        
        import importlib.util
        
        # Load the specific tool module
        tool_file = os.path.join(tools_path, f"{tool_name}.py")
        if not os.path.exists(tool_file):
            return {"error": f"Tool file {tool_name}.py not found"}, env_data
        
        try:
            # Read and fix imports
            with open(tool_file, 'r', encoding='utf-8') as f:
                tool_content = f.read()
            
            # Replace tau_bench.envs.tool with tool for local import
            tool_content = tool_content.replace('from tau_bench.envs.tool import Tool', 'from tool import Tool')
            
            # Load the module
            spec = importlib.util.spec_from_file_location(f"{domain}_{tool_name}", tool_file)
            tool_module = importlib.util.module_from_spec(spec)
            exec(tool_content, tool_module.__dict__)
            
            # Find the tool class
            class_name = ''.join(word.capitalize() for word in tool_name.split('_'))
            if hasattr(tool_module, class_name):
                tool_class = getattr(tool_module, class_name)
                
                # Execute the tool with environment data and arguments
                # env_data is passed as mutable reference, so changes persist
                if hasattr(tool_class, 'invoke'):
                    result = tool_class.invoke(env_data, **arguments)
                    # Try to parse as JSON if it's a string, otherwise return as-is
                    if isinstance(result, str):
                        try:
                            parsed_result = json.loads(result)
                        except json.JSONDecodeError:
                            parsed_result = {"result": result}
                    else:
                        parsed_result = result if isinstance(result, dict) else {"result": result}
                    
                    return parsed_result, env_data
                else:
                    return {"error": f"Tool {class_name} does not have invoke method"}, env_data
            else:
                return {"error": f"Could not find class {class_name} in {tool_name}.py"}, env_data
                
        except Exception as e:
            return {"error": f"Error executing {tool_name}: {str(e)}"}, env_data
    
    def get_tool_schemas(self, domain: str) -> List[Dict[str, Any]]:
        """Generate and return tool schemas"""
        schemas = self._extract_tool_schemas_from_domain(domain)
        print(f"Generated {len(schemas)} tool schemas for {domain} domain")
        return schemas
    
    def load_questions(self) -> List[TauBenchQuestion]:
        """Load all questions from two domains and format them into FormattedQuestion objects"""
        all_questions = []
        all_questions.extend(self.process_tau_bench_tasks("retail"))
        all_questions.extend(self.process_tau_bench_tasks("airline"))

        return all_questions