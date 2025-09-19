import json
import os
import sys
import re
from typing import Dict, Any, List
from collections import defaultdict
from . import BaseLoader
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.types import TauBenchQuestion, Benchmark



class TauBenchLoader(BaseLoader):
    """Formatter for tau-bench dataset"""
    
    FUNCTION_NAMES_MODIFYING_DATABASE = {
        'retail': ['cancel_pending_order', "exchange_delivered_order_items", "modify_pending_order_address", "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address", "return_delivered_order_items"],
        'airline': ["book_reservation", "cancel_reservation", "send_certificate", "update_reservation_baggages", "update_reservation_flights", "update_reservation_passengers"] 
    }
        

    def __init__(self):
        # Add the project root to path for imports
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.responses_by_question_id = self._load_responses()
    
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

    def _load_responses(self, response_path="benchmark/tau-bench-evaluation") -> Dict[str, list]:
        responses_by_question_id = defaultdict(list)
        for file_name in os.listdir(response_path):
            file_path = os.path.join(response_path, file_name)
            if not file_path.endswith(".jsonl"):
                continue
            with open(file_path, "r") as f:
                for response_str in f:
                    response = json.loads(response_str)
                    question_id = f"{response['task_name']}-{response['meta']['id']}"
                    responses_by_question_id[question_id].append(response)
        return dict(responses_by_question_id)

    
    def load_tau_bench_tools(self, domain: str) -> List[Dict[str, Any]]:
        """Load tool schemas for tau-bench domain using get_info() methods"""
        all_schemas = self._extract_tool_schemas_from_domain(domain)
        # Filter to only include database-modifying functions
        # db_modifying_functions = self.FUNCTION_NAMES_MODIFYING_DATABASE.get(domain, [])
        # filtered_schemas = [
        #     schema for schema in all_schemas 
        #     if schema.get('function', {}).get('name') in db_modifying_functions
        # ]
        # return filtered_schemas
        return all_schemas
    
    def format_sample(self, sample: Dict[str, Any], domain: str = None, env_data: Dict[str, Any] = None, sample_id: str = None) -> TauBenchQuestion:
        """Format tau-bench task to standard evaluation format"""
        if domain is None:
            raise ValueError("domain parameter is required for tau-bench formatting")
        
        if env_data is None:
            env_data = self.load_tau_bench_data(domain)
        
        # Extract task components
        
        user_id = sample['user_id']
        instruction = sample.get('instruction', '')
        actions = sample.get('actions', [])
        outputs = sample.get('outputs', [])
        
        agent_system_prompt = self._get_agent_system_prompt(user_id, env_data, domain)
        user_context = self._generate_user_context(user_id, env_data, sample_id)
        conversations = self._convert_actions_to_conversations(actions, domain, env_data)
        functions = self.load_tau_bench_tools(domain)
        
        return TauBenchQuestion(
            question_id=sample_id or f"{domain}-{user_id}",
            task_name=domain,
            instruction=instruction,
            gt_conv_traj=conversations,
            available_function_list=functions,
            benchmark=Benchmark.TAU_BENCH,
            agent_system_prompt=agent_system_prompt,
            user_context=user_context,
            meta={
                'tau_bench_context': {
                    'user_id': user_id,
                    'domain': domain,
                    'gt_outputs': outputs,
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

    def _get_agent_system_prompt(self, user_id: str, env_data: Dict[str, Any], domain: str = None) -> str:
        if domain not in ["retail", "airline"]:
            raise ValueError(f"Domain {domain} is not supported in tau-bench")
        wiki_file = f"data/tau-bench-envs/{domain}/wiki.md"
        if os.path.exists(wiki_file):
            with open(wiki_file, 'r', encoding='utf-8') as f:
                wiki_content = f.read().strip()
            if wiki_content:
                return demote_markdown_headings(wiki_content, 3)
    
    def _generate_user_context(self, user_id: str, env_data: Dict[str, Any], question_id: str) -> str:
        """Generate user context string based on user's orders/reservations and related products/flights."""
        domain = question_id.split("-")[0]

        if domain == "retail":
            return self._generate_retail_user_context(user_id, env_data, question_id)
        elif domain == "airline":
            return self._generate_airline_user_context(user_id, env_data, question_id)
        else:
            raise ValueError(f"Domain {domain} is not supported")
    
    def _generate_retail_user_context(self, user_id: str, env_data: Dict[str, Any], question_id: str) -> str:
        context_parts = []

        context_parts.append("\n#### User Information")
        users = env_data.get('users', {})
        orders = env_data.get('orders', {})
        products = env_data.get('products', {})
        
        user_info = users.get(user_id)
        if not user_info:
            return f"* User ID: {user_id} (No additional user information found)"
        context_parts.append(f"* User ID: {user_id}")

        # Add user details
        name_info = user_info.get('name', {})
        if isinstance(name_info, dict):
            first_name = name_info.get('first_name', '')
            last_name = name_info.get('last_name', '')
            user_name = f"{first_name} {last_name}".strip()
        else:
            user_name = str(name_info) if name_info else 'Unknown'
        context_parts.append(f"* Name: {user_name}")
        
        if 'email' in user_info:
            context_parts.append(f"* Email: {user_info['email']}")
        if 'phone' in user_info:
            context_parts.append(f"* Phone: {user_info['phone']}")
        if 'address' in user_info:
            address_info = user_info['address']
            addr_parts = []
            for key in ['address1', 'address2', 'city', 'state', 'zip', 'country']:
                addr_parts.append(str(address_info[key]))
            address_str = ', '.join(addr_parts)
            context_parts.append(f"* Address: {address_str}")
        
        if 'payment_methods' in user_info:
            payment_methods = user_info['payment_methods']
            context_parts.append(f"* Payment methods: \n```json\n{json.dumps(payment_methods, indent=2)}```")

        # retrieve relevant product ids from responses
        relevant_product_ids = []
        try:
            responses = self.responses_by_question_id[question_id]
            for response in responses:
                messages = response.get("messages", [])
                for message in messages:
                    if "tool_calls" in message:
                        for tool_call in message["tool_calls"]:
                            function_arguments = json.loads(tool_call["function"]["arguments"])
                            if "product_id" in function_arguments:
                                relevant_product_ids.append(function_arguments["product_id"])
        except:
            print(f"{question_id}: Error retrieving relevant product ids")
        relevant_product_ids = set(relevant_product_ids)
    
        if 'orders' in user_info:
            relevant_order_ids = user_info['orders']
            context_parts.append(f"\n#### Relevant Order Details:")
            for order_id in relevant_order_ids:
                order_info = orders.get(order_id)
                if order_info:
                    context_parts.append(f"\nOrder {order_id}:")
                    order_json = json.dumps(order_info, indent=2)
                    context_parts.append(f"```json\n{order_json}\n```")
                else:
                    context_parts.append(f"\nOrder {order_id}: Not found in system")

        if relevant_product_ids:
            context_parts.append(f"\n#### Relevant Product Details:")
            for relevant_product_id in relevant_product_ids:
                if relevant_product_id in products:
                    context_parts.append(f"\nProduct {relevant_product_id}:")
                    product_json = json.dumps(products[relevant_product_id])
                    context_parts.append(f"```json\n{product_json}\n```")
        return "\n".join(context_parts)
    
    
    def _generate_airline_user_context(self, user_id: str, env_data: Dict[str, Any], question_id: str) -> str:
        """Generate user context for airline domain using users, reservations, and flights."""
        import json
        context_parts = []

        context_parts.append("\n#### User Information")
        users = env_data.get('users', {})
        reservations = env_data.get('reservations', {})
        flights = env_data.get('flights', {})
        
        # Find user information - users is a dict with user_id as keys
        user_info = users.get(user_id)
        
        if not user_info:
            return f"* User ID: {user_id} (No additional user information found)"
        
        context_parts.append(f"* User ID: {user_id}")
        
        # Add user details
        name_info = user_info.get('name', {})
        if isinstance(name_info, dict):
            first_name = name_info.get('first_name', '')
            last_name = name_info.get('last_name', '')
            user_name = f"{first_name} {last_name}".strip()
        else:
            user_name = str(name_info) if name_info else 'Unknown'
        context_parts.append(f"* Name: {user_name}")
        
        if 'email' in user_info:
            context_parts.append(f"* Email: {user_info['email']}")
        if 'phone' in user_info:
            context_parts.append(f"* Phone: {user_info['phone']}")
        if 'loyalty_program' in user_info:
            context_parts.append(f"* Loyalty Program: {user_info['loyalty_program']}")
        if 'address' in user_info:
            address_info = user_info['address']
            if isinstance(address_info, dict):
                addr_parts = []
                for key in ['address1', 'address2', 'city', 'state', 'zip', 'country']:
                    if key in address_info and address_info[key]:
                        addr_parts.append(str(address_info[key]))
                address_str = ', '.join(addr_parts)
                context_parts.append(f"* Address: {address_str}")
            else:
                context_parts.append(f"* Address: {address_info}")


        if 'membership' in user_info:
            context_parts.append(f"* Membership: {user_info['membership']}")
        
        if 'saved_passengers' in user_info:
            context_parts.append(f"* Saved passengers: \n{user_info['saved_passengers']}")

        if 'payment_methods' in user_info:
            payment_methods = user_info['payment_methods']
            context_parts.append(f"* Payment methods: \n```json\n{json.dumps(payment_methods, indent=2)}```")
        
        if 'reservations' in user_info:
            reservation_ids = user_info['reservations']
            context_parts.append(f"\n#### Relevant Reservation Details:")
            for reservation_id in reservation_ids:
                reservation_info = reservations.get(reservation_id)
                if reservation_info:
                    context_parts.append(f"\nReservation {reservation_id}:")
                    reservation_json = json.dumps(reservation_info, indent=2)
                    context_parts.append(f"```json\n{reservation_json}\n```")
                else:
                    context_parts.append(f"\nReservation {reservation_id}: Not found in system")
        
        return "\n".join(context_parts)
    
    def _convert_actions_to_conversations(self, actions: List[Dict[str, Any]], domain: str = None, env_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Convert tau-bench actions to conversation format with real tool execution results"""
        
        conversations = []
        # Create a mutable copy of env_data to track state changes
        current_env_data = json.loads(json.dumps(env_data)) if env_data else {}
        
        # Get database-modifying functions for this domain
        db_modifying_functions = self.FUNCTION_NAMES_MODIFYING_DATABASE.get(domain, [])
        
        # Convert each action to assistant message with tool call + real observation
        # Only include actions that modify the database
        for i, action in enumerate(actions):
            action_name = action["name"]
            
            # Skip actions that don't modify the database
            # if action_name not in db_modifying_functions:
            #     continue
                
            # Assistant message with tool call
            conversations.append({
                "role": "assistant", 
                "function_call": [
                    {
                        "name": action_name,
                        "arguments": action.get("arguments", {})
                    }
                ]
            })
            
            # Execute the actual tool to get real observation with updated env_data
            try:
                result, updated_env_data = self._execute_tool_with_state(action_name, action.get("arguments", {}), domain, current_env_data)
                current_env_data = updated_env_data
                observation_content = result
            except Exception as e:
                observation_content = f"Error executing {action_name}: {str(e)}"
            
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
    

def demote_markdown_headings(markdown_text: str, levels_to_demote: int = 3) -> str:
    processed_lines = []
    heading_pattern = re.compile(r"^\s*(#+)\s+(.*)")

    for line in markdown_text.splitlines():
        match = heading_pattern.match(line)
        if match:
            hashes = match.group(1)
            title = match.group(2)
            new_level = min(len(hashes) + levels_to_demote, 6)
            processed_lines.append(f"{'#' * new_level} {title}")
        else:
            processed_lines.append(line)
            
    return "\n".join(processed_lines)
