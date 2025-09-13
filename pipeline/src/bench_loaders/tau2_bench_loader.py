import json
import os
import sys
import logging
from typing import Dict, Any, List
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
                
            # Load tasks from JSON file
            with open(tasks_file, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
            
            # Process each task
            for task in tasks:
                question = self._format_tau2_task(task, domain_name)
                all_questions.append(question)
        
        return all_questions
    
    def _format_tau2_task(self, task: Dict[str, Any], domain: str) -> Tau2BenchQuestion:
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
        
        # Convert evaluation criteria actions to conversation trajectory
        gt_conv_traj = []
        evaluation_criteria = task.get('evaluation_criteria', {})
        actions = evaluation_criteria.get('actions', [])
        
        for action in actions:
            # Add assistant message with function call
            gt_conv_traj.append({
                "role": "assistant",
                "function_call": [{
                    "name": action.get('name', ''),
                    "arguments": action.get('arguments', {})
                }]
            })
            
            # Add observation (we don't have actual results, so leave empty)
            gt_conv_traj.append({
                "role": "observation", 
                "content": [""]
            })
        
        return Tau2BenchQuestion(
            question_id=f"{domain}-{task_id}",
            task_name=domain,
            instruction=instruction,
            gt_conv_traj=gt_conv_traj,
            available_function_list=self._get_tau2_tool_schemas(domain),
            benchmark=Benchmark.TAU2_BENCH,
            agent_system_prompt=self._get_agent_system_prompt(domain),
            user_context=self._get_user_context(task, domain),
            available_user_function_list=self._get_user_function_schemas(domain),
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
    
    def _get_user_context(self, task: Dict[str, Any], domain: str) -> str:
        """Generate user context from task"""
        user_scenario = task.get('user_scenario', {})
        instructions = user_scenario.get('instructions', {})
        
        context_parts = []
        if instructions.get('known_info'):
            context_parts.append(f"Known information: {instructions['known_info']}")
        if instructions.get('unknown_info'):
            context_parts.append(f"Unknown information: {instructions['unknown_info']}")
            
        return "\n\n".join(context_parts) if context_parts else f"User context for {domain} domain task."
    
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