import json
import os
import sys
from typing import Dict, Any, List
from . import BaseLoader
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.types import ToolSandboxQuestion, Benchmark



class ToolSandBoxLoader(BaseLoader):
    """Loader for the ToolSandbox benchmark (Apple, 2024).

    The raw dataset is assumed to be provided in a JSONL file where **each line**
    corresponds to **one task instance** produced by the official ToolSandbox
    evaluation script.  A minimal example of a line looks like::

        {
            "model_path": "anthropic/claude-4-sonnet-thinking-off",
            "user_model_path": "openai/gpt-4o-20240806",
            "benchmark_name": "toolsandbox",
            "task_name": "update_contact_relationship_with_relationship_twice_multiple_user_turn",
            "sampling_params": {...},
            "messages": [ ... ]
        }

    The important fields for evaluation are:

    * ``task_name`` – we use this as the unique ``question_id``.
    * ``messages`` – the full ground-truth conversation trajectory.
    * ``functions`` or ``available_function_list`` – the tool schema list (if
      present in the dump).  Some public dumps omit this; in that case we fall
      back to an empty list so that downstream components can still run.

    You can override the default ``data_path`` when instantiating the loader,
    which makes it easy to point at custom subsets or updated dataset dumps.
    """

    def __init__(self, data_path: str | None = None):
        # Allow caller to specify arbitrary data file; otherwise fall back to a
        # sensible default.  We intentionally keep this *relative* so that it
        # works inside Docker as well as local execution environments.
        self.data_path = (
            data_path
            if data_path is not None
            else "data/tool_sandbox.jsonl"
        )

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _extract_first_user_prompt(self, messages: List[dict]) -> str:
        """Return the *content* field of the first message whose role=='user'."""
        for msg in messages:
            if msg.get("role") == "user":
                return msg.get("content", "")
        # Fallback – should not really happen in a well-formed sample.
        return ""

    def _format_line(self, line: dict) -> ToolSandboxQuestion:
        """Convert a raw dataset line to the unified ``ToolSandboxQuestion``."""

        messages = line.get("messages", [])

        return ToolSandboxQuestion(
            benchmark=Benchmark.TOOL_SANDBOX,
            question_id=line.get("task_name", line.get("id", "unknown")),
            instruction=self._extract_first_user_prompt(messages),
            gt_conv_traj=messages,
            # Function schema list – support multiple possible field names.
            available_function_list=line.get(
                "functions", line.get("available_function_list", [])
            ),
            # Anything else that might be useful for debugging / meta-analysis
            meta={
                k: line[k]
                for k in (
                    "model_path",
                    "user_model_path",
                    "sampling_params",
                    "benchmark_name",
                )
                if k in line
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_questions(self) -> List[ToolSandboxQuestion]:
        """Load the ToolSandbox dataset and return a list of formatted tasks."""

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"ToolSandbox data file not found: {self.data_path}. "
                "Please double-check the path or provide a custom one when "
                "instantiating ToolSandBoxLoader()."
            )

        questions: List[ToolSandboxQuestion] = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                try:
                    questions.append(self._format_line(json.loads(raw_line)))
                except Exception as e:
                    # We swallow individual line errors so that a single bad
                    # record does not ruin the entire loading process.
                    print(f"[ToolSandboxLoader] Skipped line due to error: {e}")
                    continue

        return questions