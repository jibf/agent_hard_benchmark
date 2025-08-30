import json
import os
import sys
from typing import Dict, Any, List, Optional
from . import BaseLoader
import inspect
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.types import FormattedQuestion, Benchmark

# Import will be done dynamically to avoid import issues



class NexusBenchLoader(BaseLoader):
    """Formatter for NexusBench dataset"""
    
    def __init__(self, nexusbench_path: str = "data"):
        self.nexusbench_path = nexusbench_path
    
    def load_questions(self) -> List[FormattedQuestion]:
        """Load all questions from NexusBench benchmarks by directly using datasets"""
        try:
            from datasets import load_dataset
            from dataclasses import dataclass
            from typing import Any
            
            @dataclass
            class Sample:
                query: str
                reference: Any
                
        except ImportError as e:
            print(f"Failed to import required modules: {e}")
            return []
        
        all_questions = []
        
        # Define benchmark configurations
        benchmark_configs = [
            {
                'name': 'NVDLibraryBenchmark',
                'dataset_name': 'Nexusflow/NVDLibraryBenchmark',
                'tools': ['searchCVE', 'searchCPE'],
                'reference_processor': lambda ref: ref.replace("r = nvdlib.", "")
            },
            {
                'name': 'VirusTotalBenchmark',
                'dataset_name': 'Nexusflow/VirusTotalBenchmark',
                'tools': ['vt_get_domain_report', 'vt_get_ip_report', 'vt_get_file_report'],
                'reference_processor': lambda ref: ref
            },
            {
                'name': 'ITType0Benchmark',
                'dataset_name': 'Nexusflow/ITType0Benchmark',
                'tools': ['match_values'],
                'reference_processor': lambda ref: ref
            },
            {
                'name': 'ITType1Benchmark',
                'dataset_name': 'Nexusflow/ITType1Benchmark',
                'tools': ['match_values'],
                'reference_processor': lambda ref: ref
            },
            {
                'name': 'LangChainMath',
                'dataset_name': 'Nexusflow/LangChainMathBenchmark',
                'tools': ['multiply', 'add', 'subtract', 'divide', 'sin', 'cos', 'power', 'log', 'pi', 'negate', 'return_constant'],
                'reference_processor': lambda ref: ref
            },
            {
                'name': 'MultiverseMathHard',
                'dataset_name': 'Nexusflow/MultiverseMathHard',
                'tools': ['multiply', 'add', 'subtract', 'divide', 'sin', 'cos', 'power', 'log', 'pi', 'negate', 'return_constant'],
                'reference_processor': lambda ref: ref
            },
            {
                'name': 'LangChainMultitoolTypeWriterHard',
                'dataset_name': 'Nexusflow/LangChainMultitoolTypeWriterHard',
                'tools': ['type_a', 'type_b', 'type_c'],  # These will be dynamically loaded
                'reference_processor': lambda ref: ref
            },
            {
                'name': 'LangChainTypeWriterHard',
                'dataset_name': 'Nexusflow/Langchain-Typewriter-hard-no-whitespaces',
                'tools': ['type_a', 'type_b', 'type_c'],  # These will be dynamically loaded
                'reference_processor': lambda ref: ref
            },
            {
                'name': 'LangChainRelational',
                'dataset_name': 'Nexusflow/LangChainRelational',
                'tools': ['get_data', 'filter_data', 'sort_data'],  # These will be dynamically loaded
                'reference_processor': lambda ref: ref
            },
            {
                'name': 'ClimateBenchmark',
                'dataset_name': 'Nexusflow/ClimateAPIBenchmark',
                'tools': ['get_weather', 'get_forecast', 'get_climate_data'],
                'reference_processor': lambda ref: ref
            },
            {
                'name': 'CVECPEBenchmark',
                'dataset_name': 'Nexusflow/CVECPEAPIBenchmark',
                'tools': ['searchCVE', 'searchCPE', 'mergeCVEs', 'mergeCPEs'],
                'reference_processor': lambda ref: ref
            },
            {
                'name': 'VirusTotalAgentic',
                'dataset_name': 'Nexusflow/VirusTotalAgentic',
                'tools': ['vt_get_domain_report', 'vt_get_ip_report', 'vt_get_file_report'],
                'reference_processor': lambda ref: ref
            },
            {
                'name': 'TicketTracking',
                'dataset_name': 'Nexusflow/TicketTrackingBenchmark',
                'tools': ['search_tickets'],
                'reference_processor': lambda ref: ref
            },
            {
                'name': 'TMIHallucination',
                'dataset_name': 'Nexusflow/HallucinationTMIBenchmark',
                'tools': ['test_function'],  # This is a special benchmark
                'reference_processor': lambda ref: ref
            }
        ]
        
        for config in benchmark_configs:
            try:
                # Load dataset directly from HuggingFace
                dataset = load_dataset(config['dataset_name'], split="train")
                samples = []
                
                for i, data in enumerate(dataset):
                    # Create Sample objects based on dataset structure
                    if config['name'] in ['NVDLibraryBenchmark', 'VirusTotalBenchmark', 'ITType0Benchmark', 'ITType1Benchmark', 'TicketTracking', 'ClimateBenchmark', 'CVECPEBenchmark', 'VirusTotalAgentic']:
                        query = data.get('Input', '')
                        reference = config['reference_processor'](data.get('Output', ''))
                    elif config['name'] == 'LangChainMath':
                        query = data.get('inputs', {}).get('question', '')
                        reference = data.get('outputs', {}).get('reference', '')
                    elif config['name'] in ['LangChainMultitoolTypeWriterHard', 'LangChainTypeWriterHard']:
                        query = data.get('answer', '')  # For typewriter tasks
                        reference = data.get('answer', '')
                    elif config['name'] == 'LangChainRelational':
                        query = data.get('user_query', '')
                        reference = data.get('ground_truth', '')
                    elif config['name'] == 'MultiverseMathHard':
                        query = data.get('prompt', '')
                        reference = data.get('ground_truth', '')
                    elif config['name'] == 'TMIHallucination':
                        query = data.get('user_query', '')
                        reference = data.get('modified_correct_ground_truth', '')
                    else:
                        # Handle other dataset formats as needed
                        query = str(data.get('Input', data.get('question', data.get('query', data.get('user_query', '')))))
                        reference = str(data.get('Output', data.get('answer', data.get('reference', data.get('ground_truth', '')))))
                    
                    sample = Sample(query=query, reference=reference)
                    samples.append(sample)
                
                for i, sample in enumerate(samples):
                    formatted_question = self.format_nexus_sample(
                        sample, config, f"{config['name']}_{i}"
                    )
                    if formatted_question:
                        all_questions.append(formatted_question)
                        
                print(f"Loaded {len(samples)} samples from {config['name']}")
                
            except Exception as e:
                print(f"Failed to load benchmark {config['name']}: {e}")
                continue
        
        print(f"Total loaded {len(all_questions)} questions from NexusBench")
        return all_questions
    
    def format_nexus_sample(self, sample, config, sample_id: str) -> Optional[FormattedQuestion]:
        """Format a NexusBench sample to standard evaluation format"""
        try:
            # Extract query and reference from sample
            query = sample.query
            reference = sample.reference
            
            # Handle different query types
            if isinstance(query, list):
                # For agent benchmarks with multiple turns
                user_prompt = "\n".join([f"Turn {i+1}: {q}" for i, q in enumerate(query)])
            else:
                user_prompt = str(query)
            
            # Get tool schemas
            tool_schemas = self._get_tool_schemas_from_config(config)
            
            # Create conversations based on benchmark type
            conversations = self._create_conversations(sample, config)
            
            return FormattedQuestion(
                question_id=sample_id,
                user_prompt=user_prompt,
                conversations=conversations,
                available_function_list=tool_schemas,
                benchmark=Benchmark.NEXUS_BENCH,
                meta={
                    'nexus_bench_context': {
                        'benchmark_name': config['name'],
                        'reference': str(reference),
                        'sample_type': type(sample).__name__,
                        'tools': config['tools']
                    }
                }
            )
            
        except Exception as e:
            print(f"Error formatting sample {sample_id}: {e}")
            return None
    
    def _get_tool_schemas_from_config(self, config) -> List[Dict[str, Any]]:
        """Create basic tool schemas from configuration"""
        tool_schemas = []
        for tool_name in config['tools']:
            tool_schemas.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": f"Function {tool_name} for {config['name']}",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            })
        return tool_schemas
    
    def _create_schemas_from_tools(self, tools) -> List[Dict[str, Any]]:
        """Create basic schemas from tool functions"""
        schemas = []
        for tool in tools:
            try:
                sig = inspect.signature(tool)
                parameters = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
                
                for param_name, param in sig.parameters.items():
                    param_type = "string"  # Default type
                    if param.annotation != inspect.Parameter.empty:
                        if param.annotation == int:
                            param_type = "integer"
                        elif param.annotation == float:
                            param_type = "number"
                        elif param.annotation == bool:
                            param_type = "boolean"
                        elif param.annotation == list:
                            param_type = "array"
                        elif param.annotation == dict:
                            param_type = "object"
                    
                    parameters["properties"][param_name] = {"type": param_type}
                    
                    if param.default == inspect.Parameter.empty:
                        parameters["required"].append(param_name)
                
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": tool.__name__,
                        "description": tool.__doc__ or f"Function {tool.__name__}",
                        "parameters": parameters
                    }
                })
            except Exception as e:
                print(f"Error creating schema for tool {tool.__name__}: {e}")
                continue
                
        return schemas
    
    def _create_conversations(self, sample, config) -> List[Dict[str, Any]]:
        """Create conversation format for the sample"""
        conversations = []
        
        # For most NexusBench samples, we expect a single-turn interaction
        # The reference contains the expected function call
        try:
            reference = sample.reference
            if isinstance(reference, str) and reference.strip():
                # Create assistant message with function call
                conversations.append({
                    "role": "assistant",
                    "function_call": [{
                        "name": "expected_function",
                        "arguments": {"call": reference}
                    }]
                })
                
                # Create observation with expected result
                conversations.append({
                    "role": "observation", 
                    "content": [f"Expected output: {reference}"]
                })
                
        except Exception as e:
            print(f"Error creating conversations: {e}")
            
        return conversations
    
    def _extract_sample_metadata(self, sample) -> Dict[str, Any]:
        """Extract additional metadata from sample"""
        metadata = {}
        
        # Handle different sample types with additional attributes
        if hasattr(sample, 'breadth'):
            metadata['breadth'] = sample.breadth
        if hasattr(sample, 'depth'):
            metadata['depth'] = sample.depth
        if hasattr(sample, 'expected_steps'):
            metadata['expected_steps'] = sample.expected_steps
        if hasattr(sample, 'num_functions'):
            metadata['num_functions'] = sample.num_functions
        if hasattr(sample, 'ground_truth_original'):
            metadata['ground_truth_original'] = sample.ground_truth_original
        if hasattr(sample, 'ground_truth_removed'):
            metadata['ground_truth_removed'] = sample.ground_truth_removed
        if hasattr(sample, 'tool'):
            metadata['tool_info'] = sample.tool
            
        return metadata
    
    def load_specific_benchmark(self, benchmark_name: str) -> List[FormattedQuestion]:
        """Load questions from a specific benchmark"""
        try:
            # Map benchmark names to classes
            benchmark_map = {
                'NVDLibraryBenchmark': 'NVDLibraryBenchmark',
                'VirusTotalBenchmark': 'VirusTotalBenchmark', 
                'ITType0Benchmark': 'ITType0Benchmark',
                'ITType1Benchmark': 'ITType1Benchmark',
                'LangChainMath': 'LangChainMath',
                'MultiverseMathHard': 'MultiverseMathHard',
                'LangChainMultitoolTypeWriterHard': 'LangChainMultitoolTypeWriterHard',
                'LangChainTypeWriterHard': 'LangChainTypeWriterHard',
                'LangChainRelational': 'LangChainRelational',
                'ClimateBenchmark': 'ClimateBenchmark',
                'CVECPEBenchmark': 'CVECPEBenchmark',
                'VirusTotalAgentic': 'VirusTotalAgentic',
                'TicketTracking': 'TicketTracking',
                'TMIHallucination': 'TMIHallucination'
            }
            
            if benchmark_name not in benchmark_map:
                print(f"Unknown benchmark: {benchmark_name}")
                return []
                
            from nexusbench import benchmarks
            benchmark_class = getattr(benchmarks, benchmark_map[benchmark_name])
            benchmark_instance = benchmark_class()
            
            samples = benchmark_instance.get_samples()
            questions = []
            
            for i, sample in enumerate(samples):
                formatted_question = self.format_nexus_sample(
                    sample, benchmark_instance, f"{benchmark_name}_{i}"
                )
                if formatted_question:
                    questions.append(formatted_question)
            
            print(f"Loaded {len(questions)} questions from {benchmark_name}")
            return questions
            
        except Exception as e:
            print(f"Error loading benchmark {benchmark_name}: {e}")
            return []