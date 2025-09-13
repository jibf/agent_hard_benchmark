import json
import os
import sys
import logging
import toml
from typing import Dict, Any, List, Optional
from . import BaseLoader
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.types import Tau2BenchQuestion, Benchmark



class Tau2BenchLoader(BaseLoader):
    def __init__(self):
        # Suppress tau2 logs
        os.environ['LOGURU_LEVEL'] = 'ERROR'
        super().__init__()
    
    def load_questions(self) -> List[Tau2BenchQuestion]:
        """Load questions from the dataset"""

        all_questions = []
        domains_path = "data/tau2-bench-envs/data/tau2/domains"

        # Process each domain
        for domain_name in ["airline", "retail", "telecom"]:
            domain_path = os.path.join(domains_path, domain_name)
            tasks_file = os.path.join(domain_path, "tasks.json")

            if not os.path.exists(tasks_file):
                continue

            # Load domain database data
            env_data = self.load_tau2_bench_data(domain_name)

            # Load tasks from JSON file
            with open(tasks_file, 'r', encoding='utf-8') as f:
                tasks = json.load(f)

            # Process each task
            for task in tasks:
                question = self._format_tau2_task(task, domain_name, env_data)
                all_questions.append(question)

        return all_questions

    def load_tau2_bench_data(self, domain: str) -> Dict[str, Any]:
        """Load tau2-bench environment data (users, flights/products/customers, reservations/orders/bills)"""
        domain_path = f"data/tau2-bench-envs/data/tau2/domains/{domain}"

        env_data = {}

        if domain == "telecom":
            # Telecom uses TOML files
            db_file = os.path.join(domain_path, "db.toml")
            user_db_file = os.path.join(domain_path, "user_db.toml")

            if os.path.exists(db_file):
                with open(db_file, 'r', encoding='utf-8') as f:
                    env_data.update(toml.load(f))

            if os.path.exists(user_db_file):
                with open(user_db_file, 'r', encoding='utf-8') as f:
                    env_data['user_device_state'] = toml.load(f)
        else:
            # Airline and retail use JSON files
            db_file = os.path.join(domain_path, "db.json")

            if os.path.exists(db_file):
                with open(db_file, 'r', encoding='utf-8') as f:
                    env_data = json.load(f)

        return env_data

    def _format_tau2_task(self, task: Dict[str, Any], domain: str, env_data: Dict[str, Any] = None) -> Tau2BenchQuestion:
        """Format a tau2-bench task to Tau2BenchQuestion"""
        task_id = task.get('id', 'unknown')
        
        # Extract user scenario information
        user_scenario = task.get('user_scenario', {})
        instructions = user_scenario.get('instructions', {})
        
        # Build instruction from user scenario (remove all newlines)
        instruction_parts = []
        if instructions.get('reason_for_call'):
            reason_for_call = instructions['reason_for_call'].replace('\n', ' ')
            instruction_parts.append(f"* Reason for call: {reason_for_call}")
        if instructions.get('task_instructions'):
            task_instructions = instructions['task_instructions'].replace('\n', ' ')
            instruction_parts.append(f"* Task instructions: {task_instructions}")
        if instructions.get('known_info'):
            known_info = instructions['known_info'].replace('\n', ' ')
            instruction_parts.append(f"* Known info: {known_info}")
        if instructions.get('unknown_info'):
            unknown_info = instructions['unknown_info'].replace('\n', ' ')
            instruction_parts.append(f"* Unknown info: {unknown_info}")
        
        instruction = "\n\n".join(instruction_parts)
        
        # Extract conversation trajectory and evaluation criteria
        gt_conv_traj = task["evaluation_criteria"]["actions"]
        evaluation_criteria = task.get("evaluation_criteria", {})
        initial_state = task.get("initial_state")

        return Tau2BenchQuestion(
            question_id=task_id,
            task_name=domain,
            instruction=instruction,
            gt_conv_traj=gt_conv_traj,
            available_function_list=self._get_tau2_tool_schemas(domain),
            benchmark=Benchmark.TAU2_BENCH,
            agent_system_prompt=self._get_agent_system_prompt(domain),
            user_context=self._get_user_context(task, domain, env_data),
            available_user_function_list=self._get_user_function_schemas(domain),
            initial_state=initial_state,
            evaluation_criteria=evaluation_criteria,
            meta={
                'tau2_bench_context': {
                    'domain': domain,
                    'original_task': task
                }
            }
        )
    
    def _get_tau2_tool_schemas(self, domain: str) -> List[Dict[str, Any]]:
        """Get tool schemas for tau2-bench domain using tau2 package"""
        try:
            # Add tau2-bench-envs/src to Python path for imports
            tau2_bench_path = "data/tau2-bench-envs/src"
            if tau2_bench_path not in sys.path:
                sys.path.insert(0, tau2_bench_path)
            
            # Suppress loguru logs before importing tau2
            try:
                from loguru import logger
                logger.disable("tau2")
                logger.disable("litellm")
            except ImportError:
                pass
            
            # Use registry directly to get tools
            try:
                from tau2.registry import registry
                
                env_constructor = registry.get_env_constructor(domain)
                environment = env_constructor()
                
                # Get assistant tools (exclude user tools)
                assistant_tools = environment.get_tools()
                schemas = []
                
                for tool in assistant_tools:
                    if hasattr(tool, 'openai_schema'):
                        schemas.append(tool.openai_schema)
                
                return schemas
                
            except (ImportError, Exception) as e:
                print(f"Could not load tau2 tools for domain {domain}: {e}")
                return []
                    
        except Exception as e:
            print(f"Error loading tau2 tool schemas for domain {domain}: {e}")
            return []
    
    def _get_agent_system_prompt(self, domain: str) -> str:
        """Get agent system prompt for domain"""
        try:
            # Add tau2-bench-envs/src to Python path for imports
            tau2_bench_path = "data/tau2-bench-envs/src"
            if tau2_bench_path not in sys.path:
                sys.path.insert(0, tau2_bench_path)
            
            from tau2.registry import registry
            env_constructor = registry.get_env_constructor(domain)
            environment = env_constructor()
            
            # Get policy from environment
            policy = getattr(environment, 'policy', '')
            return policy or f"You are a helpful assistant for the {domain} domain."
            
        except Exception as e:
            print(f"Error getting agent system prompt for domain {domain}: {e}")
            return f"You are a helpful assistant for the {domain} domain."
    
    def _get_user_context(self, task: Dict[str, Any], domain: str, env_data: Dict[str, Any] = None) -> str:
        """Generate user context from task with database information"""
        if not env_data:
            return f"User context for {domain} domain task."

        # Extract user ID from task
        user_id = self._extract_user_id(task, domain)
        if not user_id:
            return f"User context for {domain} domain task - no user ID found."

        if domain == "airline":
            return self._generate_airline_context(user_id, env_data, task)
        elif domain == "retail":
            return self._generate_retail_context(user_id, env_data, task)
        elif domain == "telecom":
            return self._generate_telecom_context(user_id, env_data, task)
        else:
            return f"User context for {domain} domain task."

    def _extract_user_id(self, task: Dict[str, Any], domain: str) -> Optional[str]:
        """Extract user ID from task based on domain-specific patterns"""
        import re

        # Try to extract from known_info first
        if domain == "airline":
            known_info = task["user_scenario"]["instructions"]["known_info"]
            user_id_match = re.search(r'Your user id is:?\s+?["\']?([a-zA-Z0-9_]+)["\']?', known_info)
            if user_id_match:
                return user_id_match.group(1)
        elif domain == "retail":
            known_info = task["user_scenario"]["instructions"]["known_info"]
            user_id_match = re.search(r'([a-z]+_[a-z]+_\d+)', known_info)
            if user_id_match:
                return user_id_match.group(1)
        elif domain == "telecom":
            # For telecom, check initial_state for customer_id
            initial_state = task.get('initial_state', {})
            if initial_state:
                initialization_actions = initial_state.get('initialization_actions', [])
                for action in initialization_actions:
                    if action.get('func_name') == 'set_data_usage':
                        arguments = action.get('arguments', {})
                        if 'customer_id' in arguments:
                            return arguments['customer_id']

        # If not found in known_info, check evaluation criteria actions for get_user_details
        evaluation_criteria = task.get('evaluation_criteria', {})
        actions = evaluation_criteria.get('actions', [])

        for action in actions:
            if action.get('name') == 'get_user_details':
                arguments = action.get('arguments', {})
                if 'user_id' in arguments:
                    return arguments['user_id']
                if 'customer_id' in arguments:  # For telecom
                    return arguments['customer_id']

        return None

    def _generate_airline_context(self, user_id: str, env_data: Dict[str, Any], task: Dict[str, Any]) -> str:
        """Generate airline domain user context"""
        context_parts = ["#### User Information"]

        users = env_data.get('users', {})
        flights = env_data.get('flights', {})
        reservations = env_data.get('reservations', {})

        user_info = users.get(user_id)
        if not user_info:
            return f"User {user_id} not found in airline database."

        # User basic info
        context_parts.append(f"* User ID: {user_id}")
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
        if 'dob' in user_info:
            context_parts.append(f"* Date of Birth: {user_info['dob']}")
        if 'address' in user_info:
            address_info = user_info['address']
            if isinstance(address_info, dict):
                addr_parts = []
                for key in ['address1', 'address2', 'city', 'state', 'zip', 'country']:
                    if key in address_info and address_info[key]:
                        addr_parts.append(str(address_info[key]))
                address_str = ', '.join(addr_parts)
                context_parts.append(f"* Address: {address_str}")

        if 'payment_methods' in user_info:
            payment_methods = user_info['payment_methods']
            context_parts.append(f"* Payment methods:\n```json\n{json.dumps(payment_methods, indent=2)}\n```")

        # Reservations
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

        # Add task-specific context
        user_scenario = task.get('user_scenario', {})
        instructions = user_scenario.get('instructions', {})
        if instructions.get('known_info'):
            context_parts.append(f"\n#### Additional Context:")
            context_parts.append(f"Known information: {instructions['known_info']}")
        if instructions.get('unknown_info'):
            context_parts.append(f"Unknown information: {instructions['unknown_info']}")

        return "\n".join(context_parts)

    def _generate_retail_context(self, user_id: str, env_data: Dict[str, Any], task: Dict[str, Any]) -> str:
        """Generate retail domain user context"""
        context_parts = ["#### User Information"]

        users = env_data.get('users', {})
        products = env_data.get('products', {})
        orders = env_data.get('orders', {})

        user_info = users.get(user_id)
        if not user_info:
            return f"User {user_id} not found in retail database."

        # User basic info
        context_parts.append(f"* User ID: {user_id}")
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
        if 'address' in user_info:
            address_info = user_info['address']
            if isinstance(address_info, dict):
                addr_parts = []
                for key in ['address1', 'address2', 'city', 'state', 'zip', 'country']:
                    if key in address_info and address_info[key]:
                        addr_parts.append(str(address_info[key]))
                address_str = ', '.join(addr_parts)
                context_parts.append(f"* Address: {address_str}")

        if 'payment_methods' in user_info:
            payment_methods = user_info['payment_methods']
            context_parts.append(f"* Payment methods:\n```json\n{json.dumps(payment_methods, indent=2)}\n```")

        # Orders
        if 'orders' in user_info:
            order_ids = user_info['orders']
            context_parts.append(f"\n#### Relevant Order Details:")
            for order_id in order_ids:
                order_info = orders.get(order_id)
                if order_info:
                    context_parts.append(f"\nOrder {order_id}:")
                    order_json = json.dumps(order_info, indent=2)
                    context_parts.append(f"```json\n{order_json}\n```")
                else:
                    context_parts.append(f"\nOrder {order_id}: Not found in system")

        # Add task-specific context
        user_scenario = task.get('user_scenario', {})
        instructions = user_scenario.get('instructions', {})
        if instructions.get('known_info'):
            context_parts.append(f"\n#### Additional Context:")
            context_parts.append(f"Known information: {instructions['known_info']}")
        if instructions.get('unknown_info'):
            context_parts.append(f"Unknown information: {instructions['unknown_info']}")

        return "\n".join(context_parts)

    def _generate_telecom_context(self, customer_id: str, env_data: Dict[str, Any], task: Dict[str, Any]) -> str:
        """Generate telecom domain user context"""
        context_parts = ["#### Customer Information"]

        customers = env_data.get('customers', [])
        lines = env_data.get('lines', [])
        bills = env_data.get('bills', [])
        plans = env_data.get('plans', [])
        devices = env_data.get('devices', [])
        user_device_state = env_data.get('user_device_state', {})

        # Find customer
        customer_info = None
        for customer in customers:
            if customer.get('customer_id') == customer_id:
                customer_info = customer
                break

        if not customer_info:
            return f"Customer {customer_id} not found in telecom database."

        # Customer basic info
        context_parts.append(f"* Customer ID: {customer_id}")
        context_parts.append(f"* Name: {customer_info.get('full_name', 'Unknown')}")
        context_parts.append(f"* Email: {customer_info.get('email', 'N/A')}")
        context_parts.append(f"* Phone: {customer_info.get('phone_number', 'N/A')}")
        context_parts.append(f"* Date of Birth: {customer_info.get('date_of_birth', 'N/A')}")
        context_parts.append(f"* Account Status: {customer_info.get('account_status', 'Unknown')}")

        # Payment methods
        if 'payment_methods' in customer_info:
            payment_methods = customer_info['payment_methods']
            context_parts.append(f"* Payment methods:\n```json\n{json.dumps(payment_methods, indent=2)}\n```")

        # Lines
        line_ids = customer_info.get('line_ids', [])
        if line_ids:
            context_parts.append(f"\n#### Customer Lines:")
            for line_id in line_ids:
                line_info = None
                for line in lines:
                    if line.get('line_id') == line_id:
                        line_info = line
                        break

                if line_info:
                    context_parts.append(f"\nLine {line_id}:")
                    line_json = json.dumps(line_info, indent=2)
                    context_parts.append(f"```json\n{line_json}\n```")

        # Bills
        bill_ids = customer_info.get('bill_ids', [])
        if bill_ids:
            context_parts.append(f"\n#### Customer Bills:")
            for bill_id in bill_ids:
                bill_info = None
                for bill in bills:
                    if bill.get('bill_id') == bill_id:
                        bill_info = bill
                        break

                if bill_info:
                    context_parts.append(f"\nBill {bill_id}:")
                    bill_json = json.dumps(bill_info, indent=2)
                    context_parts.append(f"```json\n{bill_json}\n```")

        # Current device state
        if user_device_state:
            context_parts.append(f"\n#### Current Device State:")
            device_state_json = json.dumps(user_device_state, indent=2)
            context_parts.append(f"```json\n{device_state_json}\n```")

        # Add task-specific context
        user_scenario = task.get('user_scenario', {})
        instructions = user_scenario.get('instructions', {})
        if instructions.get('known_info'):
            context_parts.append(f"\n#### Additional Context:")
            context_parts.append(f"Known information: {instructions['known_info']}")
        if instructions.get('unknown_info'):
            context_parts.append(f"Unknown information: {instructions['unknown_info']}")

        return "\n".join(context_parts)
    
    def _get_user_function_schemas(self, domain: str) -> List[Dict[str, Any]]:
        """Get user function schemas for domain (mainly for telecom)"""
        if domain != 'telecom':
            return []
            
        try:
            # Add tau2-bench-envs/src to Python path for imports
            tau2_bench_path = "data/tau2-bench-envs/src"
            if tau2_bench_path not in sys.path:
                sys.path.insert(0, tau2_bench_path)
            
            from tau2.registry import registry
            env_constructor = registry.get_env_constructor(domain)
            environment = env_constructor()
            
            # Get user tools
            if hasattr(environment, 'get_user_tools'):
                user_tools = environment.get_user_tools()
                schemas = []
                
                for tool in user_tools:
                    if hasattr(tool, 'openai_schema'):
                        schemas.append(tool.openai_schema)
                
                return schemas
            
        except Exception as e:
            print(f"Error getting user function schemas for domain {domain}: {e}")
            
        return []