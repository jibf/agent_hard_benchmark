import json
import os
import sys
import logging
import toml
import re
from copy import deepcopy
from typing import Dict, Any, List, Optional
from collections import defaultdict
from . import BaseLoader
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.types import Tau2BenchQuestion, Benchmark



class Tau2BenchLoader(BaseLoader):
    def __init__(self):
        # Suppress tau2 logs
        os.environ['LOGURU_LEVEL'] = 'ERROR'
        super().__init__()
        self.responses_by_question_id = self._load_responses()

        # Cache user function schemas for each domain
        self.user_function_schemas_cache = {}
        for domain in ["airline", "retail", "telecom"]:
            self.user_function_schemas_cache[domain] = self._get_user_function_schemas(domain)
    
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

    def _load_responses(self, response_path="benchmark/tau2-bench-evaluation") -> Dict[str, list]:
        """Load responses for extracting relevant product IDs"""
        responses_by_question_id = defaultdict(list)
        if not os.path.exists(response_path):
            return dict(responses_by_question_id)

        for file_name in os.listdir(response_path):
            file_path = os.path.join(response_path, file_name)
            if not file_path.endswith(".jsonl"):
                continue
            with open(file_path, "r") as f:
                for response_str in f:
                    response = json.loads(response_str)
                    task_name = response['task_name']
                    meta_id = response['meta']['id']
                    # For telecom, meta['id'] already includes the domain prefix (e.g., "telecom_[...]")
                    # For airline/retail, meta['id'] is just a number
                    # Standardize to underscore format (domain_id) to match pipeline conventions
                    if meta_id.startswith(f"{task_name}_"):
                        question_id = meta_id  # Already in correct underscore format
                    elif meta_id.startswith(f"{task_name}-"):
                        # Convert hyphen format to underscore format for consistency
                        question_id = meta_id.replace(f"{task_name}-", f"{task_name}_", 1)
                    else:
                        # Format as domain_id (underscore) to match pipeline conventions
                        question_id = f"{task_name}_{meta_id}"
                    responses_by_question_id[question_id].append(response)
        return dict(responses_by_question_id)

    def _format_tau2_task(self, task: Dict[str, Any], domain: str, env_data: Dict[str, Any] = None) -> Tau2BenchQuestion:
        """Format a tau2-bench task to Tau2BenchQuestion"""
        task_id = task.get('id', 'unknown')
        
        # For telecom domain, task IDs in tasks.json don't include the "telecom_" prefix,
        # but response files do. Add the prefix to match response format.
        if domain == "telecom" and not task_id.startswith("telecom_"):
            task_id = f"telecom_{task_id}"
        
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
        gt_actions = task["evaluation_criteria"]["actions"]

        initial_state = task.get("initial_state")

        # Generate conversations with observations for each tool call
        gt_conv_traj = self._convert_actions_to_conversations(
            gt_actions,
            domain,
            env_data,
            initial_state,
        )

        # combine task purpose to the evaluation_criteria
        evaluation_criteria = task.get("evaluation_criteria", {})
        evaluation_criteria["task_purpose"] = task["description"]["purpose"]
        initial_state_with_state_dump = deepcopy(initial_state) if initial_state is not None else None
        if initial_state_with_state_dump is not None:
            post_init_state = self._get_post_initialization_environment_state(
                domain=domain,
                initial_state=initial_state_with_state_dump,
            )
            if post_init_state is not None:
                initial_state_with_state_dump[
                    "post_initialization_environment_state"
                ] = post_init_state

        return Tau2BenchQuestion(
            question_id=task_id,
            task_name=domain,
            instruction=instruction,
            gt_conv_traj=gt_conv_traj,
            available_function_list=self._get_tau2_tool_schemas(domain),
            benchmark=Benchmark.TAU2_BENCH,
            agent_system_prompt=self._get_agent_system_prompt(domain),
            user_context=self._get_user_context(task, domain, env_data),
            available_user_function_list=self.user_function_schemas_cache[domain],
            initial_state=initial_state_with_state_dump,
            evaluation_criteria=evaluation_criteria,
            meta={
                'tau2_bench_context': {
                    'domain': domain,
                    'original_task': task
                }
            }
        )

    def _get_post_initialization_environment_state(
        self, domain: str, initial_state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Run the initialization actions for a task and capture resulting environment state."""

        if not initial_state:
            return None

        initialization_actions = initial_state.get("initialization_actions") or []
        initialization_data = initial_state.get("initialization_data")
        message_history = initial_state.get("message_history") or []

        if not initialization_actions and not initialization_data and not message_history:
            return None

        tau2_bench_path = "data/tau2-bench-envs/src"
        if tau2_bench_path not in sys.path:
            sys.path.insert(0, tau2_bench_path)

        try:
            from loguru import logger

            logger.disable("tau2")
            logger.disable("litellm")
        except ImportError:
            pass

        try:
            from tau2.registry import registry
        except Exception:
            return None

        try:
            environment = registry.get_env_constructor(domain)()
        except Exception:
            return None

        init_data_obj, init_actions_obj, message_history_obj = (
            self._prepare_initial_state_objects(initial_state)
        )

        if init_data_obj is init_actions_obj is message_history_obj is None:
            return None

        try:
            environment.set_state(
                initialization_data=init_data_obj,
                initialization_actions=init_actions_obj,
                message_history=message_history_obj or [],
            )
        except Exception:
            return None

        return self._snapshot_environment_state(environment)

    @staticmethod
    def _snapshot_environment_state(environment) -> Optional[Dict[str, Any]]:
        """Serialize relevant environment state for prompt inclusion."""

        snapshot: Dict[str, Any] = {}

        tools = getattr(environment, "tools", None)
        if tools is not None:
            tool_db = getattr(tools, "db", None)
            if tool_db is not None and hasattr(tool_db, "model_dump"):
                try:
                    snapshot["assistant_db"] = tool_db.model_dump(mode="json")
                except Exception:
                    snapshot["assistant_db"] = tool_db.model_dump()

        user_tools = getattr(environment, "user_tools", None)
        if user_tools is not None:
            user_db = getattr(user_tools, "db", None)
            if user_db is not None and hasattr(user_db, "model_dump"):
                try:
                    snapshot["user_db"] = user_db.model_dump(mode="json")
                except Exception:
                    snapshot["user_db"] = user_db.model_dump()

        return snapshot or None

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
        # Add tau2-bench-envs/src to Python path for imports
        tau2_bench_path = "data/tau2-bench-envs/src"
        if tau2_bench_path not in sys.path:
            sys.path.insert(0, tau2_bench_path)

        from tau2.registry import registry
        env_constructor = registry.get_env_constructor(domain)
        environment = env_constructor()

        # Get policy from environment
        policy = getattr(environment, 'policy', '')
        return demote_markdown_headings(policy, 3) if policy else f"You are a helpful assistant for the {domain} domain."
    
    def _get_user_context(self, task: Dict[str, Any], domain: str, env_data: Dict[str, Any] = None) -> str:
        """Generate user context from task with database information"""
        if not env_data:
            return f"User context for {domain} domain task."

        # Extract user ID from task
        user_id = self._extract_user_id(task, domain)

        # If no user_id found, try to find by email first (most accurate for retail)
        if not user_id and domain == "retail":
            email = self._extract_email_from_known_info(task)
            if email:
                user_id = self._find_user_id_by_email_in_db(email, env_data, domain)
        
        # If still no user_id found, try to find by name from "You are [name]" pattern
        if not user_id:
            name_info = self._extract_user_name_from_known_info(task)
            if name_info:
                first_name, last_name = name_info
                # For retail domain, also try to extract zip code for more accurate lookup
                zip_code = None
                if domain == "retail":
                    zip_code = self._extract_zip_code_from_known_info(task)
                user_id = self._find_user_id_by_name_in_db(first_name, last_name, env_data, domain, zip_code=zip_code)

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

        # If not found in known_info, check evaluation criteria actions for get_user_details or find_user_id_by_name_zip
        evaluation_criteria = task.get('evaluation_criteria', {})
        actions = evaluation_criteria.get('actions', [])

        for action in actions:
            if action.get('name') == 'get_user_details':
                arguments = action.get('arguments', {})
                if 'user_id' in arguments:
                    return arguments['user_id']
                if 'customer_id' in arguments:  # For telecom
                    return arguments['customer_id']

        # Fallback: Extract name from "You are [first_name] [last_name]" pattern in known_info
        # This will be used to look up user_id by name in the context generation
        return None

    def _extract_user_name_from_known_info(self, task: Dict[str, Any]) -> Optional[tuple[str, str]]:
        """Extract first and last name from various name patterns in known_info"""
        import re

        known_info = task["user_scenario"]["instructions"]["known_info"]
        # Pattern to match various forms:
        # - "You are [first_name] [last_name]"
        # - "You're [first_name] [last_name]"
        # - "[anything] name is [first_name] [last_name]"
        patterns = [
            r'(?:You are|You\'re)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)',
            r'name is\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)'
        ]

        for pattern in patterns:
            name_match = re.search(pattern, known_info)
            if name_match:
                return name_match.group(1), name_match.group(2)
        return None

    def _extract_zip_code_from_known_info(self, task: Dict[str, Any]) -> Optional[str]:
        """Extract zip code from known_info for retail domain"""
        import re

        known_info = task["user_scenario"]["instructions"]["known_info"]
        # Pattern to match zip code: "your zip code is 78705", "zip code is 78705", "zip code 78705", or "zip 78705"
        zip_patterns = [
            r'(?:your\s+)?zip\s+code\s+is\s+(\d{5})',
            r'zip\s+code\s+(\d{5})',
            r'zip\s+(\d{5})'
        ]

        for pattern in zip_patterns:
            zip_match = re.search(pattern, known_info, re.IGNORECASE)
            if zip_match:
                return zip_match.group(1)
        return None

    def _extract_email_from_known_info(self, task: Dict[str, Any]) -> Optional[str]:
        """Extract email from known_info for retail domain"""
        import re

        known_info = task["user_scenario"]["instructions"]["known_info"]
        # Pattern to match email: "your email is x@y.com", "email is x@y.com", "email x@y.com"
        email_patterns = [
            r'(?:your\s+)?email\s+is\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            r'email\s+is\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            r'email\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'  # Fallback: any email pattern
        ]

        for pattern in email_patterns:
            email_match = re.search(pattern, known_info, re.IGNORECASE)
            if email_match:
                return email_match.group(1).lower().strip()
        return None

    def _find_user_id_by_email_in_db(self, email: str, env_data: Dict[str, Any], domain: str) -> Optional[str]:
        """Find user_id by searching for email in the database"""
        if domain == "retail" or domain == "airline":
            users = env_data.get('users', {})
            for user_id, user_info in users.items():
                user_email = user_info.get('email', '')
                if user_email.lower() == email.lower():
                    return user_id
        return None

    def _find_user_id_by_name_in_db(self, first_name: str, last_name: str, env_data: Dict[str, Any], domain: str, zip_code: Optional[str] = None) -> Optional[str]:
        """Find user_id by searching for name in the database. For retail domain, zip_code can be used for disambiguation."""
        if domain == "airline" or domain == "retail":
            users = env_data.get('users', {})
            for user_id, user_info in users.items():
                name_info = user_info.get('name', {})
                if isinstance(name_info, dict):
                    user_first = name_info.get('first_name', '').lower()
                    user_last = name_info.get('last_name', '').lower()
                    if user_first == first_name.lower() and user_last == last_name.lower():
                        # For retail domain, if zip_code is provided, also match zip code
                        if domain == "retail" and zip_code:
                            address_info = user_info.get('address', {})
                            if isinstance(address_info, dict):
                                user_zip = address_info.get('zip', '')
                                if user_zip == zip_code:
                                    return user_id
                            # If zip doesn't match, continue searching
                            continue
                        # If no zip_code provided or domain is airline, return first match
                        return user_id
        elif domain == "telecom":
            customers = env_data.get('customers', [])
            for customer in customers:
                full_name = customer.get('full_name', '')
                # Split full name and compare
                if ' ' in full_name:
                    parts = full_name.split(' ', 1)
                    user_first = parts[0].lower()
                    user_last = parts[1].lower() if len(parts) > 1 else ''
                    if user_first == first_name.lower() and user_last == last_name.lower():
                        return customer.get('customer_id')
        return None

    def _generate_airline_context(self, user_id: str, env_data: Dict[str, Any], task: Dict[str, Any]) -> str:
        """Generate airline domain user context"""
        context_parts = ["#### User Information"]
        domain = "airline"
        task_id = task.get('id', 'unknown')

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
        domain = "retail"
        task_id = task.get('id', 'unknown')

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

        # Retrieve relevant product IDs from responses (for retail domain)
        relevant_product_ids = []
        # Normalize question_id to match the format used in responses_by_question_id (underscore format)
        if task_id.startswith(f"{domain}_"):
            normalized_qid = task_id  # Already in correct underscore format
        elif task_id.startswith(f"{domain}-"):
            normalized_qid = task_id.replace(f"{domain}-", f"{domain}_", 1)
        else:
            normalized_qid = f"{domain}_{task_id}"
        responses = self.responses_by_question_id.get(normalized_qid, [])
        for response in responses:
            messages = response.get("messages", [])
            for message in messages:
                if "tool_calls" in message:
                    for tool_call in message["tool_calls"]:
                        function_arguments = tool_call["arguments"]
                        if "product_id" in function_arguments:
                            relevant_product_ids.append(function_arguments["product_id"])
        relevant_product_ids = list(set(relevant_product_ids))

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

        # Add relevant product details
        if relevant_product_ids:
            products = env_data.get('products', {})
            context_parts.append(f"\n#### Relevant Product Details:")
            for relevant_product_id in relevant_product_ids:
                if relevant_product_id in products:
                    context_parts.append(f"\nProduct {relevant_product_id}:")
                    product_json = json.dumps(products[relevant_product_id], indent=2)
                    context_parts.append(f"```json\n{product_json}\n```")

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
        
        return []

    def _convert_actions_to_conversations(
        self,
        actions: List[Dict[str, Any]],
        domain: str,
        env_data: Dict[str, Any] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Convert tau2-bench actions to conversation format with real tool execution results"""
        conversations = []
        # Create a mutable copy of env_data to track state changes
        current_env_data = json.loads(json.dumps(env_data)) if env_data else {}
        # Create a single environment instance to persist state across tool calls
        tau2_bench_path = "data/tau2-bench-envs/src"
        if tau2_bench_path not in sys.path:
            sys.path.insert(0, tau2_bench_path)

        # Suppress loguru logs
        try:
            from loguru import logger
            logger.disable("tau2")
            logger.disable("litellm")
        except ImportError:
            pass

        from tau2.registry import registry
        env_constructor = registry.get_env_constructor(domain)
        environment = env_constructor()

        init_data_obj, init_actions_obj, message_history_obj = (
            self._prepare_initial_state_objects(initial_state)
        )

        if not (init_data_obj is init_actions_obj is message_history_obj is None):
            try:
                environment.set_state(
                    initialization_data=init_data_obj,
                    initialization_actions=init_actions_obj,
                    message_history=message_history_obj or [],
                )
            except Exception:
                pass

        # Set the initial environment data
        if hasattr(environment, '_data'):
            environment._data.update(current_env_data)

        # Convert each action to assistant message with tool call + real observation
        for i, action in enumerate(actions):
            action_name = action.get("func_name", action.get("name", ""))
            action_arguments = action.get("arguments", {})

            is_user_function = False
            for user_function_schema in self.user_function_schemas_cache[domain]:
                if action_name == user_function_schema["function"]["name"]:
                    is_user_function = True

            # Assistant message with tool call
            conversations.append({
                "role": "user" if is_user_function else "assistant",
                "function_call": [
                    {
                        "name": action_name,
                        "arguments": action_arguments
                    }
                ]
            })

            if is_user_function:
                result = self._execute_tool_with_state_user(action_name, action_arguments, domain, environment)
            else:
                result = self._execute_tool_with_state_agent(action_name, action_arguments, domain, environment)
                # Sync tools after assistant tool execution to propagate state changes
                # (e.g., payment requests need to be synced from assistant DB to user DB)
                if hasattr(environment, 'sync_tools'):
                    try:
                        environment.sync_tools()
                    except Exception:
                        pass  # Ignore sync errors during reconstruction

            # Update current_env_data from environment state
            if hasattr(environment, '_data'):
                current_env_data = environment._data

            observation_content = result

            conversations.append({
                "role": "observation",
                "content": [observation_content]
            })

        return conversations

    def _prepare_initial_state_objects(
        self, initial_state: Optional[Dict[str, Any]]
    ) -> tuple[Optional[Any], Optional[List[Any]], Optional[List[Any]]]:
        """Convert initial state dictionary into tau2 model instances for environment.set_state."""

        if not initial_state:
            return None, None, None

        initialization_data = initial_state.get("initialization_data")
        initialization_actions = initial_state.get("initialization_actions")
        message_history = initial_state.get("message_history") or []

        tau2_bench_path = "data/tau2-bench-envs/src"
        if tau2_bench_path not in sys.path:
            sys.path.insert(0, tau2_bench_path)

        try:
            from tau2.data_model.tasks import InitializationData, EnvFunctionCall
            from tau2.data_model.message import Message
        except Exception:
            return None, None, None

        init_data_obj = None
        if initialization_data is not None:
            try:
                init_data_obj = InitializationData.model_validate(initialization_data)
            except Exception:
                init_data_obj = None

        init_actions_obj = None
        if initialization_actions:
            try:
                init_actions_obj = [
                    EnvFunctionCall.model_validate(action)
                    for action in initialization_actions
                ]
            except Exception:
                return None, None, None

        message_history_obj: Optional[List[Any]] = []
        if message_history:
            try:
                message_history_obj = [Message.model_validate(msg) for msg in message_history]
            except Exception:
                message_history_obj = []

        return init_data_obj, init_actions_obj, message_history_obj

    def _execute_tool_with_state_user(self, tool_name: str, arguments: Dict[str, Any], domain: str, environment) -> Dict[str, Any]:
        """Execute a tau2-bench user tool using persistent environment"""
        # Get user tools from the persistent environment
        user_tools = environment.get_user_tools()
        target_tool = None

        for tool in user_tools:
            if hasattr(tool, 'openai_schema') and tool.openai_schema.get('function', {}).get('name') == tool_name:
                target_tool = tool
                break

        if target_tool is None:
            return {"error": f"User tool {tool_name} not found"}

        # Execute the tool
        try:
            result = target_tool(**arguments)
        except Exception as e:
            return {"error": f"Error executing user tool {tool_name}: {str(e)}"}

        # User tools return plain text strings, wrap in dict for consistency
        if isinstance(result, str):
            parsed_result = {"content": result}
        else:
            parsed_result = {"content": str(result)}

        return parsed_result

    def _execute_tool_with_state_agent(self, tool_name: str, arguments: Dict[str, Any], domain: str, environment) -> Dict[str, Any]:
        """Execute a tau2-bench assistant tool using persistent environment"""
        # Get assistant tools from the persistent environment
        assistant_tools = environment.get_tools()
        target_tool = None

        for tool in assistant_tools:
            if hasattr(tool, 'openai_schema') and tool.openai_schema.get('function', {}).get('name') == tool_name:
                target_tool = tool
                break

        if target_tool is None:
            return {"error": f"Assistant tool {tool_name} not found"}

        # Execute the tool
        try:
            result = target_tool(**arguments)
        except Exception as e:
            return {"error": f"Error executing assistant tool {tool_name}: {str(e)}"}

        # Assistant tools return JSON-parseable strings or objects
        if isinstance(result, str):
            try:
                parsed_result = json.loads(result)
            except json.JSONDecodeError:
                parsed_result = {"content": result}
        elif hasattr(result, "__dict__"):
            parsed_result = dict(result)
        else:
            parsed_result = {"content": str(result)}

        return parsed_result


def demote_markdown_headings(markdown_text: str, levels_to_demote: int = 3) -> str:
    """Demote markdown headings by specified number of levels"""
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
