import json
import os
import copy
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys_path_added = False

try:
    from .base_loader import BaseLoader
except ImportError:  # pragma: no cover - running file directly
    if not sys_path_added:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        sys_path_added = True
    import importlib.util
    base_loader_path = Path(__file__).resolve().parent / "base_loader.py"
    spec = importlib.util.spec_from_file_location("tool_sandbox_base_loader", base_loader_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    BaseLoader = module.BaseLoader  # type: ignore[attr-defined]

from src.utils.types import ToolSandboxQuestion, Benchmark

# Ensure ToolSandbox package is importable
PIPELINE_DATA_ROOT = Path(__file__).resolve().parents[3] / "pipeline" / "data"
if str(PIPELINE_DATA_ROOT) not in os.sys.path:
    os.sys.path.append(str(PIPELINE_DATA_ROOT))

from tool_sandbox.scenarios import named_scenarios
from tool_sandbox.common.tool_discovery import ToolBackend, get_all_tools
from tool_sandbox.common.execution_context import DatabaseNamespace, RoleType, new_context
from tool_sandbox.common.scenario import Scenario
from tool_sandbox.common.tool_conversion import convert_to_openai_tools


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_value(val) for key, val in value.items()}
    return value


class ToolSandBoxLoader(BaseLoader):
    def __init__(self) -> None:
        self.scenarios = named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT)
        self.tool_specs = get_all_tools(ToolBackend.DEFAULT)

    def load_questions(self) -> List[ToolSandboxQuestion]:
        questions: List[ToolSandboxQuestion] = []
        for name, scenario in self.scenarios.items():
            questions.append(self._convert_scenario_to_question(name, scenario))
        return questions

    def _convert_scenario_to_question(self, scenario_name: str, scenario: Scenario) -> ToolSandboxQuestion:
        print("="*10, scenario_name, "="*10)

        starting_context = scenario.starting_context

        starting_db = starting_context.get_database(
            DatabaseNamespace.SANDBOX,
            drop_sandbox_message_index=False,
            get_all_history_snapshots=True,
        )

        # 1. Extract available function list 
        with new_context(starting_context):
            available_tools = {
                name: tool
                for name, tool in starting_context.get_available_tools(scrambling_allowed=True).items()
                if RoleType.AGENT in getattr(tool, "visible_to", (RoleType.AGENT,))
            }
            available_functions = convert_to_openai_tools(available_tools)
        
        # 2. Extract instruction (to the user model)
        
        for i in range(len(starting_db)-1, -1, -1):      # fetch the last message sent from the system to the user
            entry = starting_db[i]
            if entry["sender"][0] == RoleType.SYSTEM and entry["recipient"][0] == RoleType.USER:
                assert len(entry["content"]) == 1
                user_system_prompt_and_instruction: str = entry["content"][0]
                end_of_system_prompt = "Answer User B's questions given the following task you (User A) want User B to complete: "
                idx = user_system_prompt_and_instruction.find(end_of_system_prompt)
                if idx < 0:
                    raise ValueError("Invalid system prompt")
                instruction = user_system_prompt_and_instruction[idx:]
                user_system_prompt = user_system_prompt_and_instruction[:idx]
                break

        # 3. Extract milestone and minefield and serialize them.
        # TODO: add more context, e.g., description of each matching method




        evaluation = scenario.evaluation
        milestone_matcher = evaluation.milestone_matcher
        minefield_matcher = evaluation.minefield_matcher
        milestones = self._remove_irrelevants_from_milestone(self._serialize_milestones(milestone_matcher.milestones))
        minefields = self._remove_irrelevants_from_milestone(self._serialize_milestones(minefield_matcher.milestones))

        #4. Extract initial status of relevant database entries, adaptively to each scenario
        all_initial_databases = {
            namespace.value: self._dataframe_to_rows(
                starting_context.get_database(namespace, drop_sandbox_message_index=False)
            )
            for namespace in DatabaseNamespace
        }

        all_namespaces = set(DatabaseNamespace)
        relevant_namespaces = {DatabaseNamespace.SANDBOX}

        def add_namespaces_from_milestones(matcher: Any) -> None:
            milestones = getattr(matcher, "milestones", None)
            if not milestones:
                return
            for milestone in milestones:
                for constraint in getattr(milestone.snapshot_constraints, "snapshot_constraints", []):
                    if constraint.target_dataframe:
                        relevant_namespaces.add(constraint.database_namespace)
                guardrail_list = getattr(milestone, "guardrail_database_list", None)
                if guardrail_list:
                    guardrail_set = set(guardrail_list)
                    if guardrail_set and guardrail_set != all_namespaces:
                        relevant_namespaces.update(guardrail_set)

        add_namespaces_from_milestones(milestone_matcher)
        add_namespaces_from_milestones(minefield_matcher)

        initial_databases: Dict[str, List[Dict[str, Any]]] = {}
        for namespace in DatabaseNamespace:
            if namespace not in relevant_namespaces:
                continue
            rows = all_initial_databases.get(namespace.value)
            if rows:
                initial_databases[namespace.value] = rows


        import pdb; pdb.set_trace()

        # return ToolSandboxQuestion(
        #     question_id=scenario_name,
        #     instruction=instruction,
        #     available_function_list=available_functions,
        #     benchmark=Benchmark.TOOL_SANDBOX,
        #     initial_databases=initial_databases
        #     milestones=milestones,
        #     minefields=minefields
        # )
    


    # expected_output: Optional[str] = None
    # starting_messages: Optional[List[Dict[str, Any]]] = None
    # initial_databases: Optional[Dict[str, List[Dict[str, Any]]]] = None
    # milestones: Optional[List[Dict[str, Any]]] = None
    # minefields: Optional[List[Dict[str, Any]]] = None
    # categories: Optional[List[str]] = None
    # tool_allow_list: Optional[List[str]] = None
    # tool_augmentation_list: Optional[List[str]] = None


    def _remove_irrelevants_from_milestone(self, milestone: List[Dict[str, List]]) -> List[Dict[str, List]]:
        milestone = copy.deepcopy(milestone)
        indices_without_target_df = []
        for i, entry in enumerate(milestone):
            new_snapshot_constraints = []
            for constraint in entry["snapshot_constraints"]:
                if constraint["target_dataframe"]:
                    del constraint["column_similarity_measure"]
                    new_snapshot_constraints.append(constraint)
            entry["snapshot_constraints"] = new_snapshot_constraints
        
        return milestone

    def _find_message(
        self,
        messages: List[Dict[str, Any]],
        sender: RoleType,
        recipient: RoleType,
        first: bool = False,
    ) -> Optional[str]:
        iterable = messages if first else reversed(messages)
        for message in iterable:
            if message.get("sender") == sender.name and message.get("recipient") == recipient.name:
                content = message.get("content")
                if content:
                    return content
        return None


    def _extract_ground_truth(self, scenario: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
        calls: List[Dict[str, Any]] = []
        responses: List[str] = []
        milestones = scenario.evaluation.milestone_matcher.milestones or []
        for milestone in milestones:
            for constraint in milestone.snapshot_constraints:
                if constraint.database_namespace != DatabaseNamespace.SANDBOX:
                    continue
                rows = self._dataframe_to_rows(constraint.target_dataframe)
                for row in rows:
                    trace_raw = row.get("tool_trace")
                    if trace_raw:
                        traces = trace_raw
                        if isinstance(traces, str):
                            traces = json.loads(traces)
                        if isinstance(traces, dict):
                            traces = [traces]
                        for trace in traces:
                            calls.append(
                                {
                                    "tool_name": trace.get("tool_name"),
                                    "arguments": trace.get("arguments", {}),
                                }
                            )
                    if row.get("sender") == RoleType.AGENT.name and row.get("recipient") == RoleType.USER.name:
                        content = row.get("content")
                        if content:
                            responses.append(content)
        return calls, responses

    def _build_ground_truth_events(
        self,
        messages: List[Dict[str, Any]],
        calls: List[Dict[str, Any]],
        final_messages: List[str],
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        import_snippet = self._find_message(messages, RoleType.SYSTEM, RoleType.EXECUTION_ENVIRONMENT)
        if import_snippet:
            events.append({"type": "system_setup", "content": import_snippet})
        for call in calls:
            events.append({"type": "tool_call", **call})
        if final_messages:
            events.append({"type": "assistant_message", "content": final_messages[-1]})
        return events

    def _serialize_milestones(self, milestones: Optional[List[Any]]) -> List[Dict[str, Any]]:
        serialized: List[Dict[str, Any]] = []
        if not milestones:
            return serialized
        for milestone in milestones:
            constraints: List[Dict[str, Any]] = []
            for constraint in milestone.snapshot_constraints:
                column_similarity = None
                if constraint.column_similarity_measure:
                    column_similarity = {
                        column: getattr(func, "__name__", str(func))
                        for column, func in constraint.column_similarity_measure.items()
                    }
                constraints.append(
                    {
                        "database_namespace": constraint.database_namespace.value,
                        "snapshot_constraint": getattr(constraint.snapshot_constraint, "__name__", str(constraint.snapshot_constraint)),
                        "reference_milestone_node_index": constraint.reference_milestone_node_index,
                        "target_dataframe": self._dataframe_to_rows(constraint.target_dataframe),
                        "column_similarity_measure": column_similarity,
                    }
                )
            serialized.append({"snapshot_constraints": constraints})
        return serialized

    def _dataframe_to_rows(self, dataframe: Any) -> List[Dict[str, Any]]:
        if dataframe is None:
            return []
        if isinstance(dataframe, str):
            return [] if dataframe == "" else [dataframe]
        if hasattr(dataframe, "to_dicts"):
            if getattr(dataframe, "height", None) == 0:
                return []
            rows: List[Dict[str, Any]] = []
            for row in dataframe.to_dicts():
                rows.append({key: _normalize_value(value) for key, value in row.items()})
            return rows
        value = _normalize_value(dataframe)
        if isinstance(value, list):
            return value
        return [value]
