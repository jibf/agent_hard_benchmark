import os
import sys
import inspect
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from datasets import load_dataset
from . import BaseLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.types import NexusBenchQuestion, Benchmark

# Import nexusbench components at module level
try:
    # First try local nexusbench in data directory
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    from nexusbench.config import BENCHMARKS
    from nexusbench.benchmarks import BaseBenchmark
except ImportError:
    try:
        # Fallback to installed nexusbench
        from nexusbench.config import BENCHMARKS
        from nexusbench.benchmarks import BaseBenchmark
    except ImportError:
        BENCHMARKS = None
        BaseBenchmark = None


@dataclass
class Sample:
    """Sample data structure for NexusBench"""
    query: str
    reference: Any

class NexusBenchLoader(BaseLoader):
    """Formatter for NexusBench dataset"""
    TASK_SIZE_DICT = {
        "VirusTotal": 151,
        "ClimateBenchmark": 197,
        "CVECPEBenchmark": 56,
        "ITType0Benchmark": 111,
        "ITType1Benchmark": 125,
        "LangChainMath": 20,
        "LangChainMultitoolTypeWriterHard": 30,
        "LangChainRelational": 20,
        "LangChainTypeWriterHard": 30,
        "MultiverseMathHard": 100,
        "NVDLibraryBenchmark": 78,
        "TicketTracking": 91,
        "TMIHallucination": 61,
        "VirusTotalAgentic": 54
    }

    # TODO: add logic to check this
    TASKS_TO_TRUNCATE_30 = ["LangChainMultitoolTypeWriterHard", "LangChainTypeWriterHard"]

    # Special cases for dataset naming (exceptions to the default pattern)
    DATASET_NAME_OVERRIDES = {
        'VirusTotal': 'Nexusflow/VirusTotalBenchmark',
        'LangChainMath': 'Nexusflow/LangChainMathBenchmark',
        'LangChainTypeWriterHard': 'Nexusflow/Langchain-Typewriter-hard-no-whitespaces',
        'ClimateBenchmark': 'Nexusflow/ClimateAPIBenchmark',
        'CVECPEBenchmark': 'Nexusflow/CVECPEAPIBenchmark',
        'TicketTracking': 'Nexusflow/TicketTrackingBenchmark',
        'TMIHallucination': 'Nexusflow/HallucinationTMIBenchmark'
    }

    # Field mappings for different benchmark types
    FIELD_MAPPINGS = {
        'NVDLibraryBenchmark': {'query': 'Input', 'reference': 'Output'},
        'VirusTotal': {'query': 'Input', 'reference': 'Output'},
        'VirusTotalBenchmark': {'query': 'Input', 'reference': 'Output'},
        'ITType0Benchmark': {'query': 'Input', 'reference': 'Output'},
        'ITType1Benchmark': {'query': 'Input', 'reference': 'Output'},
        'TicketTracking': {'query': 'Input', 'reference': 'Output'},
        'ClimateBenchmark': {'query': 'Input', 'reference': 'Output'},
        'CVECPEBenchmark': {'query': 'Input', 'reference': 'Output'},
        'VirusTotalAgentic': {'query': 'Input', 'reference': 'Output'},
        'LangChainMath': {'query': ('inputs', 'question'), 'reference': ('outputs', 'reference')},
        'LangChainMultitoolTypeWriterHard': {'query': 'answer', 'reference': 'answer'},
        'LangChainTypeWriterHard': {'query': 'answer', 'reference': 'answer'},
        'LangChainRelational': {'query': 'user_query', 'reference': 'ground_truth'},
        'MultiverseMathHard': {'query': 'prompt', 'reference': 'ground_truth'},
        'TMIHallucination': {'query': 'user_query', 'reference': 'modified_correct_ground_truth'}
    }

    def __init__(self, nexusbench_path: str = "data"):
        self.nexusbench_path = nexusbench_path
        self._benchmark_instances = {}

    def _get_dataset_name(self, benchmark_name: str) -> str:
        """Get dataset name for a benchmark, using overrides or default pattern"""
        return self.DATASET_NAME_OVERRIDES.get(benchmark_name, 'Nexusflow/' + benchmark_name)

    def _get_benchmark_instance(self, benchmark_name: str) -> Optional[BaseBenchmark]:
        """Get benchmark instance from nexusbench"""
        if benchmark_name in self._benchmark_instances:
            return self._benchmark_instances[benchmark_name]

        if BENCHMARKS is None:
            return None

        for benchmark_cls in BENCHMARKS:
            if benchmark_cls.NAME == benchmark_name:
                instance = benchmark_cls()
                self._benchmark_instances[benchmark_name] = instance
                return instance
        return None

    def _get_tools_for_benchmark(self, benchmark_name: str) -> List[str]:
        """Get tools for a benchmark from nexusbench instance"""
        benchmark_instance = self._get_benchmark_instance(benchmark_name)
        if benchmark_instance is None:
            return []

        try:
            tools = benchmark_instance.tools
            return [tool.__name__ for tool in tools]
        except Exception as e:
            print(f"Error getting tools for {benchmark_name}: {e}")
            return []

    def _extract_field_value(self, data: Dict, field_spec) -> str:
        """Extract value from data using field specification"""
        if isinstance(field_spec, tuple):
            # Nested field access like ('inputs', 'question')
            value = data
            for key in field_spec:
                value = value.get(key, '')
            return str(value)
        # Simple field access
        return str(data.get(field_spec, ''))

    def load_task(self, task_name: str) -> List[NexusBenchQuestion]:
        # TODO: Implement specific task loading
        return self.load_specific_benchmark(task_name)

    def load_questions(self) -> List[NexusBenchQuestion]:
        """Load all questions from NexusBench benchmarks by directly using datasets"""
        all_questions = []

        # Get available benchmark names from TASK_SIZE_DICT
        benchmark_names = list(self.TASK_SIZE_DICT.keys())

        for benchmark_name in benchmark_names:
            # Load dataset directly from HuggingFace
            dataset_name = self._get_dataset_name(benchmark_name)
            try:
                dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
            except Exception as e:
                print(f"Failed to load {dataset_name} with error: {e}")
                print(f"Trying alternative approach for {benchmark_name}")
                # Try without trust_remote_code
                try:
                    dataset = load_dataset(dataset_name, split="train")
                except Exception as e2:
                    print(f"Alternative approach also failed: {e2}")
                    # Skip this dataset for now but continue with others
                    continue
            samples = []

            # Get field mappings for this benchmark
            field_mapping = self.FIELD_MAPPINGS.get(
                benchmark_name, {'query': 'Input', 'reference': 'Output'})

            # Get tools for this benchmark
            tools = self._get_tools_for_benchmark(benchmark_name)

            for i, data in enumerate(dataset):
                # Check if we need to truncate this benchmark to 30 samples
                if benchmark_name in self.TASKS_TO_TRUNCATE_30 and i >= 30:
                    break
                # Extract query and reference using field mappings
                query = self._extract_field_value(data, field_mapping['query'])
                reference = self._extract_field_value(data, field_mapping['reference'])

                # Special processing for NVDLibraryBenchmark reference
                if benchmark_name == 'NVDLibraryBenchmark' and reference:
                    reference = reference.replace("r = nvdlib.", "")

                sample = Sample(query=query, reference=reference)
                # Store original data for special handling (e.g., TMIHallucination json_tools)
                if hasattr(sample, '__dict__'):
                    sample.__dict__['_original_data'] = data
                samples.append(sample)

            # Create benchmark config dynamically
            config = {
                'name': benchmark_name,
                'dataset_name': dataset_name,
                'tools': tools
            }

            for i, sample in enumerate(samples):
                formatted_question = self.format_nexus_sample(
                    sample, config, f"{benchmark_name}_{i}"
                )
                if formatted_question:
                    all_questions.append(formatted_question)

            print(f"Loaded {len(samples)} samples from {benchmark_name}")

            # Validate task size if specified
            expected_size = self.TASK_SIZE_DICT.get(benchmark_name)
            if expected_size and len(samples) != expected_size:
                print(f"Warning: {benchmark_name} has {len(samples)} "
                        f"samples, expected {expected_size}")

            # except Exception as e:
            #     print(f"Failed to load benchmark {benchmark_name}: {e}")
            #     continue

        print(f"Total loaded {len(all_questions)} questions from NexusBench")
        return all_questions

    def format_nexus_sample(self, sample, config, sample_id: str) -> Optional[NexusBenchQuestion]:
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

            # Get tool schemas - handle special cases
            tool_schemas = self._get_tool_schemas_from_config(config, sample)

            # Create conversations based on benchmark type
            conversations = self._create_conversations(sample, config)

            # Build the formatted question first so we can attach extra helper fields
            formatted_question = NexusBenchQuestion(
                question_id=sample_id,
                instruction=user_prompt,
                gt_conv_traj=conversations,
                available_function_list=tool_schemas,
                benchmark=Benchmark.NEXUS_BENCH,
                meta={
                    'nexus_bench_context': {
                        'benchmark_name': config['name'],
                        'reference': str(reference),
                        'sample_type': type(sample).__name__,
                        'tools': config['tools']
                    },
                    # Expose user_prompt and conversations for downstream prompt formatting
                    'user_prompt': user_prompt,
                    'conversations': conversations
                }
            )

            # ------------------------------------------------------------------
            # Attach helper attributes expected by the judge prompt templates.
            # We do this *after* construction to avoid pydantic validation errors
            # for unexpected fields.
            # These attributes are dynamically added so that format_judge_prompt
            # can access them via ``hasattr(question, field)``.
            # ------------------------------------------------------------------
            formatted_question.user_prompt = user_prompt
            formatted_question.conversations = conversations

            return formatted_question

        except Exception as e:
            print(f"Error formatting sample {sample_id}: {e}")
            return None

    def _get_tool_schemas_from_config(self, config, sample=None) -> List[Dict[str, Any]]:
        """Create tool schemas from configuration using benchmark's get_json_representation"""
        
        # Special handling for TMIHallucination - use sample's json_tools
        if config['name'] == 'TMIHallucination' and sample:
            try:
                import json
                # Try to get json_tools from original data
                original_data = getattr(sample, '_original_data', None)
                json_tools_raw = None
                
                if original_data and 'json_tools' in original_data:
                    json_tools_raw = original_data['json_tools']
                elif hasattr(sample, 'json_tools'):
                    json_tools_raw = sample.json_tools
                    
                if json_tools_raw:
                    # Parse JSON if it's a string
                    if isinstance(json_tools_raw, str):
                        json_tools = json.loads(json_tools_raw)
                    else:
                        json_tools = json_tools_raw
                        
                    if isinstance(json_tools, list) and len(json_tools) > 0:
                        tool_schemas = []
                        for func_spec in json_tools:
                            if isinstance(func_spec, dict) and 'name' in func_spec:
                                tool_schemas.append({
                                    "type": "function",
                                    "function": func_spec
                                })
                        return tool_schemas
            except Exception as e:
                print(f"Error getting json_tools from TMIHallucination sample: {e}")
                import traceback
                traceback.print_exc()
        
        # Special handling for VirusTotal - use get_all_json_specs
        if config['name'] == 'VirusTotal':
            try:
                from data.nexusbench.tools.virustotal import get_all_json_specs
                json_schemas = get_all_json_specs()
                tool_schemas = []
                
                if json_schemas:
                    for _, func_schema in json_schemas.items():
                        tool_schemas.append({
                            "type": "function",
                            "function": func_schema
                        })
                return tool_schemas
            except Exception as e:
                print(f"Error getting VirusTotal json specs: {e}")
        
        # Standard handling using benchmark instance
        benchmark_instance = self._get_benchmark_instance(config['name'])
        if benchmark_instance:
            try:
                # Use get_json_representation to get proper function schemas
                json_schemas = benchmark_instance.get_json_representation
                tool_schemas = []
                
                if json_schemas:  # Check if json_schemas is not empty
                    for _, func_schema in json_schemas.items():
                        tool_schemas.append({
                            "type": "function",
                            "function": func_schema
                        })
                else:
                    print(f"Warning: {config['name']} has empty get_json_representation")
                
                return tool_schemas
            except Exception as e:
                print(f"Error getting json representation from benchmark instance: {e}")

        # Fallback to basic schema creation if benchmark instance not available
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

    def _create_conversations(self, sample, _) -> List[Dict[str, Any]]:
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

    def load_specific_benchmark(self, benchmark_name: str) -> List[NexusBenchQuestion]:
        """Load questions from a specific benchmark"""
        try:
            if benchmark_name not in self.TASK_SIZE_DICT:
                print(f"Unknown benchmark: {benchmark_name}")
                return []

            benchmark_instance = self._get_benchmark_instance(benchmark_name)
            if benchmark_instance is None:
                print(f"Could not get benchmark instance for {benchmark_name}")
                return []

            samples = benchmark_instance.get_samples()
            questions = []

            # Create config for this specific benchmark
            config = {
                'name': benchmark_name,
                'dataset_name': self._get_dataset_name(benchmark_name),
                'tools': self._get_tools_for_benchmark(benchmark_name)
            }

            for i, sample in enumerate(samples):
                formatted_question = self.format_nexus_sample(
                    sample, config, f"{benchmark_name}_{i}"
                )
                if formatted_question:
                    questions.append(formatted_question)

            # Validate task size
            expected_size = self.TASK_SIZE_DICT.get(benchmark_name)
            if expected_size and len(questions) != expected_size:
                print(f"Warning: {benchmark_name} has {len(questions)} "
                      f"questions, expected {expected_size}")

            print(f"Loaded {len(questions)} questions from {benchmark_name}")
            return questions

        except Exception as e:
            print(f"Error loading benchmark {benchmark_name}: {e}")
            return []
