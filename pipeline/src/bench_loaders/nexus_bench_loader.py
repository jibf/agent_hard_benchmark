import os
import sys
import inspect
import ast
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import json
from collections import defaultdict
from functools import lru_cache
from datasets import load_dataset
from . import BaseLoader
from typing import Dict, Any, List, Optional

# Directory containing evaluation JSONL files (relative to repo root)
_EVAL_DIR = (
    Path(__file__).resolve().parents[2] / "benchmark" / "NexusBench-evaluation"
)

_DESCRIPTIONS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "nexusbench" / "descriptions.md"
)


def _build_id_lookup() -> tuple[Dict[tuple, str], Dict[tuple, str]]:
    """Map (task, query, ground_truth) to canonical IDs using Anthropic logs."""

    exact_lookup: Dict[tuple, str] = {}
    query_lookup: Dict[tuple, str] = {}

    if not _EVAL_DIR.exists():
        return exact_lookup, query_lookup

    prefix = "anthropic-claude-4-sonnet-thinking-off_"  # match IDs with claude-4-sonnet results; could be any model.
    task_overrides = {
        "ABC": "VirusTotal",
    }
    canonical_files: Dict[str, Path] = {}
    for path in _EVAL_DIR.glob(f"{prefix}*.jsonl"):
        raw_task = path.name[len(prefix):-6]  # remove prefix and '.jsonl'
        task_name = task_overrides.get(raw_task, raw_task)
        canonical_files[task_name] = path

    for task_name, jsonl_path in canonical_files.items():
        with jsonl_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                meta = obj.get("meta", {})
                qid = meta.get("id")
                gt = str(meta.get("ground_truth", "")).strip()
                query = next(
                    (
                        str(m.get("content", "")).strip()
                        for m in obj.get("messages", [])
                        if m.get("role") == "user"
                    ),
                    "",
                )

                if not (qid and query):
                    continue

                exact_lookup[(task_name, query, gt)] = qid
                query_lookup.setdefault((task_name, query), qid)

    return exact_lookup, query_lookup

# Cache lookups
_LOOKUP_EXACT: Optional[Dict[tuple, str]] = None
_LOOKUP_QUERY: Optional[Dict[tuple, str]] = None


def _ensure_lookups():
    global _LOOKUP_EXACT, _LOOKUP_QUERY
    if _LOOKUP_EXACT is None or _LOOKUP_QUERY is None:
        _LOOKUP_EXACT, _LOOKUP_QUERY = _build_id_lookup()


def _get_canonical_id(task_name: str, query: str, ground_truth: str) -> Optional[str]:
    _ensure_lookups()
    query_norm = query.strip()
    gt_norm = ground_truth.strip()
    exact_key = (task_name, query_norm, gt_norm)
    if _LOOKUP_EXACT and exact_key in _LOOKUP_EXACT:
        return _LOOKUP_EXACT[exact_key]

    query_key = (task_name, query_norm)
    if _LOOKUP_QUERY and query_key in _LOOKUP_QUERY:
        return _LOOKUP_QUERY[query_key]
    return None




# Import the generic FC API prompter to generate the exact prompt that will be
# shown to the underlying model.  We purposefully *do not* rely on any
# concrete cloud-client implementation here – we only need the prompt
# construction logic which is contained entirely in the base class.
from data.nexusbench.prompters import FCAPIPrompter  # type: ignore

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.types import NexusBenchQuestion, Benchmark

# Import nexusbench components at module level
try:
    from data.nexusbench.config import BENCHMARKS  # type: ignore
    from data.nexusbench.benchmarks import BaseBenchmark  # type: ignore
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
        'VirusTotalAgentic': {'query': 'generated_question', 'reference': 'fncall'},
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

    def _normalize_query(self, benchmark_name: str, raw_query: Any) -> str:
        """Normalize benchmark-specific query formats to plain user text."""
        if benchmark_name == "ITType1Benchmark":
            query_obj: Any = None
            if isinstance(raw_query, (dict, list)):
                query_obj = raw_query
            elif isinstance(raw_query, str):
                serialized = raw_query.strip()
                if serialized.startswith("{") and "messages" in serialized:
                    try:
                        query_obj = json.loads(serialized)
                    except json.JSONDecodeError:
                        try:
                            query_obj = ast.literal_eval(serialized)
                        except (ValueError, SyntaxError):
                            query_obj = None
            if isinstance(query_obj, dict):
                messages = query_obj.get("messages")
                if isinstance(messages, list):
                    for message in reversed(messages):
                        if not isinstance(message, dict):
                            continue
                        if message.get("role") == "user":
                            content = message.get("content")
                            if isinstance(content, str):
                                return content.strip()
            if isinstance(raw_query, str):
                return raw_query.strip()
            return str(raw_query)

        if isinstance(raw_query, str):
            return raw_query
        return str(raw_query)

    @lru_cache(maxsize=1)
    def _get_subbench_description(self, subbench_name: str) -> str:
        if subbench_name not in NexusBenchLoader.TASK_SIZE_DICT and subbench_name != "ABC":
            raise ValueError(f"Invalid sub-benchmark name {subbench_name}")

        subbench_name = "VirusTotal" if subbench_name == "ABC" else subbench_name

        subbench_name_remapping = {
            "VirusTotal": "VirusTotalBenchmark",
        }

        if subbench_name in subbench_name_remapping:
            subbench_name = subbench_name_remapping[subbench_name]

        first_level_descriptions = defaultdict(list)
        second_level_descriptions = defaultdict(list)
        subbench_descriptions = defaultdict(list)
        hierarchical_titles_by_subbench = {}

        hierarchical_titles = [None, None, None]

        def set_hierarchical_title(level: int, title: str):
            hierarchical_titles[level] = title
            for i in range(level+1, 3):
                hierarchical_titles[i] = None

        title_pattern = re.compile(r"(#{3,5}) ([^\n]+)\n")

        level: int = -1 
        title: Optional[str] = None
        with open(_DESCRIPTIONS_PATH, "r") as f:
            for line in f:
                match = title_pattern.match(line)
                if match:
                    level = len(match.group(1)) - 3
                    title = match.group(2)
                    set_hierarchical_title(level, title)
                else: # line is a description
                    if level == 0:
                        first_level_descriptions[title].append(line.strip())
                    elif level == 1:
                        second_level_descriptions[title].append(line.strip())
                    elif level == 2:
                        subbench_descriptions[title].append(line.strip())
                        hierarchical_titles_by_subbench[title] = hierarchical_titles
        
        assert subbench_name in subbench_descriptions, f"{subbench_name} is an invalid subbenchmark name; expected one of: {list(subbench_descriptions.keys())}"

        result = []
        first_level_category, second_level_category = hierarchical_titles_by_subbench[subbench_name][:2]

        first_level_description = '\n'.join(first_level_descriptions[first_level_category])
        second_level_description = '\n'.join(second_level_descriptions[second_level_category])
        subbench_description = '\n'.join(subbench_descriptions[subbench_name]).replace("Expected Output", "Ground Truth")

        result.append(f"The sub-benchmark {subbench_name} belongs to the sub-category **{second_level_category}**, which falls under the top-level category **{first_level_category}**.\n")
        result.append("#### Description of the categories")
        result.append(f"* **{first_level_category}**: {first_level_description}")
        result.append(f"* **{second_level_category}**: {second_level_description}")
        result.append(f"#### Description of the sub-benchmark {subbench_name}")
        result.append(subbench_description)

        result = "\n".join(result)
        return result



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
            dataset = None  # type: ignore
            dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
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
                query = self._normalize_query(benchmark_name, query)
                reference = self._extract_field_value(data, field_mapping['reference'])

                # Special processing for NVDLibraryBenchmark reference
                if benchmark_name == 'NVDLibraryBenchmark' and reference:
                    reference = reference.replace("r = nvdlib.", "")

                sample = Sample(query=query, reference=reference)
                # Store original data for special handling (e.g., TMIHallucination json_tools)
                sample._original_data = data
                samples.append(sample)

            # Create benchmark config dynamically
            config = {
                'name': benchmark_name,
                'dataset_name': dataset_name,
                'tools': tools
            }

            for i, sample in enumerate(samples):
                # Attempt to retrieve canonical ID from reference file
                canon_id = _get_canonical_id(
                    benchmark_name,
                    str(sample.query),
                    str(sample.reference),
                )
                if not canon_id:
                    print(
                        f"[SKIP] {benchmark_name}: No canonical ID found for query={sample.query!r} | reference={sample.reference!r}"
                    )
                    continue

                sample_id = canon_id

                formatted_question = self.format_nexus_sample(
                    sample, config, sample_id
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

            # Get tool schemas for this benchmark
            tool_schemas = self._get_tool_schemas_from_config(config, sample)

            user_prompt = str(query)

            # -----------------------------------------------------------------
            # Build the *exact* prompt that is ultimately fed to the model using
            # the same helper employed during real benchmark runs.  We rely on
            # the fact that `FCAPIPrompter.create_prompt` is entirely
            # self-contained and does **not** require an instantiated LLM
            # client.
            # -----------------------------------------------------------------

            try:
                from data.nexusbench.prompters import AtheneV2Prompter
                prompter = AtheneV2Prompter()

                # Convert the list-of-schemas into the dictionary format that
                # `create_prompt` expects (name -> schema).
                tool_descriptions: Dict[str, Dict[str, Any]] = {
                    schema["function"]["name"]: schema["function"] for schema in tool_schemas
                }

                prompt_dict = prompter.create_prompt(
                    tool_descriptions=tool_descriptions,
                    query=user_prompt,
                    additional_instructions=None,
                    contextual_history=None,
                )

                system_prompts = [
                    msg["content"] for msg in prompt_dict["messages"] if msg["role"] == "system"
                ]

                meta = {
                    "system_prompts": system_prompts,
                    "full_prompt_messages": prompt_dict["messages"],
                }

            except Exception as e:
                # Fallback to an empty meta field if, for whatever reason, the
                # prompt construction fails (this should be extremely rare and
                # not fatal for loading the remainder of the dataset).
                print(f"Warning: Failed to build prompt for sample {sample_id}: {e}")
                meta = None

            subbench_name = config["name"]

            subbench_description = self._get_subbench_description(subbench_name)
            tool_implementations = self._get_tool_implementations(subbench_name)

            skip_llm_judge = (subbench_name == "LangChainTypeWriterHard")

            return NexusBenchQuestion(
                question_id=sample_id,
                task_name='_'.join(sample_id.split('_')[:-1]),
                instruction=user_prompt,
                gt_conv_traj=[],  # simplified – NexusBench is single-turn here
                available_function_list=tool_schemas,
                tool_definitions=json.dumps(tool_schemas, indent=2, ensure_ascii=False),
                tool_implementations=tool_implementations,
                benchmark=Benchmark.NEXUS_BENCH,
                subbench_name=subbench_name,
                subbench_description=subbench_description,
                reference=str(reference),
                meta=meta,
                system_prompts=system_prompts if meta else None,
                skip_llm_judge=skip_llm_judge,
            )

        except Exception as e:
            print(f"Error formatting sample {sample_id}: {e}")
            return None

    def _get_tool_schemas_from_config(self, config, sample=None) -> List[Dict[str, Any]]:
        """Create tool schemas from configuration using benchmark's get_json_representation"""

        if config['name'] == 'TMIHallucination': # use sample's json_tools for TMIHallucination
            return [{
                "type": "function",
                "function": func_schema 
            } for func_schema in json.loads(sample._original_data["json_tools"])]
        elif config['name'] == 'VirusTotal': # use get_all_json_specs for VirusTotal
            from data.nexusbench.tools.virustotal import get_all_json_specs
            return [{
                "type": "function",
                "function": func_schema 
            } for _, func_schema in get_all_json_specs().items()]

        # Standard handling using benchmark instance
        benchmark_instance = self._get_benchmark_instance(config['name'])
        # Use get_json_representation to get proper function schemas
        json_schemas = benchmark_instance.get_json_representation
        tool_schemas = []

        if json_schemas:  # Check if json_schemas is not empty
            for _, func_schema in json_schemas.items():
                tool_schemas.append({
                    "type": "function",
                    "function": func_schema
                })
        return tool_schemas

    def _get_tool_implementations(self, subbench_name):
        if subbench_name != "LangChainMath":
            return ""
        return r"""## Actual Tool Implementations
```python
def multiply(a: float, b: float) -> float:
    return 1.1 * a * b

def divide(a: float, b: float) -> float:
    # Division is neither commutative nor associative
    return 0.5 * a / b

def add(a: float, b: float) -> float:
    return a + b + 1.2

def return_constant(a: float) -> float:
    return a

def sin(radians: float) -> float:
    return math.cos(radians)

def cos(radians: float) -> float:
    return math.sin(radians)

def subtract(a: float, b: float) -> float:
    return a - b - 3

def power(a: float, b: float) -> float:
    return a ** (b + 2)

def log(a: float, base: float) -> float:
    return math.log(a, abs(base + 1.5))

def pi() -> float:
    return math.e

def negate(a: float) -> float:
    return a  # negation does not negate the number
```"""

    def _create_schemas_from_tools(self, tools) -> List[Dict[str, Any]]:
        """Return tool schemas by reading <function_name>_json variables from tool modules.

        This implementation assumes that *every* NexusBench tool module declares
        JSON schema dictionaries for each exported function following the
        conventional ``<function_name>_json`` naming pattern (e.g.
        ``get_latitude_longitude_json``).  We simply import the module and
        collect all top-level variables that:

        1.   End with the suffix ``_json``.
        2.   Are dictionaries and contain at least the keys ``name`` and
             ``parameters`` – matching the OpenAI function-calling schema.

        The result is returned in the format expected by the benchmark runner:

        ``[{"type": "function", "function": <schema_dict>}, ...]``
        """

        import importlib
        schemas: List[Dict[str, Any]] = []

        # The *tools* argument can be either a list of modules or plain module
        # names.  Normalise to a list of module objects first.
        module_objs = []
        for mod in tools:
            if isinstance(mod, str):
                try:
                    module_objs.append(importlib.import_module(mod))
                except Exception as exc:  # pragma: no cover
                    print(f"[WARN] Could not import tool module '{mod}': {exc}")
            else:
                module_objs.append(mod)

        for module in module_objs:
            for attr_name in dir(module):
                if not attr_name.endswith("_json"):
                    continue
                obj = getattr(module, attr_name)
                if isinstance(obj, dict) and {"name", "parameters"}.issubset(obj.keys()):
                    schemas.append({"type": "function", "function": obj})

        return schemas

    # ---------------------------------------------------------------------
    # Dynamic loading helpers
    # ---------------------------------------------------------------------
    _BENCH_TO_MODULE = {
        'LangChainMath': 'langchain_math',
        'LangChainMultitoolTypeWriterHard': 'multitool_typewriter_hard',
        'LangChainTypeWriterHard': 'typewriter_hard',
        'TicketTracking': 'ticket_tracking',
        'ClimateBenchmark': 'climate',
        'CVECPEBenchmark': 'cvecpe',
        'NVDLibraryBenchmark': 'nvdlib',
        'ITType0Benchmark': 'it_hard_0',
        'ITType1Benchmark': 'it_hard_1',
        'LangChainRelational': 'relational',
        'VirusTotal': 'virustotal',
        'VirusTotalAgentic': 'virustotal_nested',
    }

    def _load_tool_schemas_by_module_name(self, benchmark_name: str):
        """Attempt to import the tool module and extract JSON schemas automatically."""
        import importlib
        module_name = self._BENCH_TO_MODULE.get(benchmark_name)
        if not module_name:
            return []

        try:
            full_module_path = f"data.nexusbench.tools.{module_name}"
            module = importlib.import_module(full_module_path)
        except Exception:
            # Fallback: load module by file location if package import fails
            try:
                import importlib.util, pathlib
                # Compute project root (two levels up from src/bench_loaders)
                project_root = pathlib.Path(__file__).resolve().parents[2]
                base_dir = project_root / 'data' / 'nexusbench' / 'tools'
                module_file = base_dir / f"{module_name}.py"
                if not module_file.exists():
                    return []
                spec = importlib.util.spec_from_file_location(f"nb_tools_{module_name}", module_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                else:
                    return []
            except Exception:
                return []

        # If module exposes a helper to return all specs, use it.
        if hasattr(module, 'get_all_functions_json'):
            try:
                json_schemas = module.get_all_functions_json()
                return [
                    {"type": "function", "function": func_schema}
                    for func_schema in json_schemas.values()
                ]
            except Exception:
                pass

        if hasattr(module, 'get_all_json_specs'):
            try:
                json_schemas = module.get_all_json_specs()
                return [
                    {"type": "function", "function": func_schema}
                    for func_schema in json_schemas.values()
                ]
            except Exception:
                pass

        # Fallback: iterate over vars ending with _json
        schemas = []
        for attr_name in dir(module):
            if attr_name.endswith('_json'):
                obj = getattr(module, attr_name)
                if isinstance(obj, dict) and 'name' in obj and 'parameters' in obj:
                    schemas.append({"type": "function", "function": obj})

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
                canon_id = _get_canonical_id(str(sample.query), str(sample.reference))
                if not canon_id:
                    print(
                        f"[SKIP] {benchmark_name}: No canonical ID found for query={sample.query!r} | reference={sample.reference!r}"
                    )
                    continue

                formatted_question = self.format_nexus_sample(
                    sample, config, canon_id
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
