import json
import os
import sys
from typing import Dict, Any, List
from . import BaseLoader
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.types import ToolSandboxQuestion, Benchmark



class ToolSandBoxLoader(BaseLoader):
    """Loader for the ToolSandbox benchmark dataset.

    The dataset lives in ``data/ToolSandbox.json`` and follows the structure
    illustrated in the repository (see README / sample file).  Each entry
    corresponds to a *scenario* to be evaluated by the LLM-as-a-judge
    pipeline.  This loader converts the raw scenario dictionary into the
    unified :class:`~src.utils.types.ToolSandboxQuestion` dataclass so the
    rest of the evaluation code can treat it the same way as the other
    benchmarks (Tau-Bench, ComplexFuncBench, …).

    We intentionally keep the conversion logic lightweight:

    • *question_id*  → ``scenario_name`` from the JSON.
    • *instruction*  → The first user utterance that is directed to the
      assistant **within** the SANDBOX messages.
    • *gt_conv_traj* → A minimal conversation list constructed from the raw
      SANDBOX messages.  We preserve the original ordering and map the
      ToolSandbox sender tags (``USER``, ``AGENT``, ``SYSTEM``) to the
      OpenAI role names (``user``, ``assistant``, ``system``).  If a message
      already represents a tool call (``openai_function_name`` is not
      ``None``) we surface it through the ``function_call`` field so that
      downstream components can pick it up.
    • *available_function_list* → We build a *very* thin schema list from the
      ``tool_allow_list`` guard-rail in the starting context.  The real tool
      JSON schema lives inside the ToolSandbox package but replicating the
      full import logic here would add unnecessary complexity.  For judging
      purposes the **name** field alone is sufficient.
    • *meta* → We keep the full raw scenario so future debugging is easier.
    """

    ROLE_MAPPING = {
        "USER": "user",
        "AGENT": "assistant",
        "SYSTEM": "system",
        # Fallback – if the sender is e.g., EXECUTION_ENVIRONMENT or anything
        # else we map it to system so the content is still visible.
    }

    def __init__(self, data_path: str | None = None):
        # Default path relative to the project root
        self.data_path = data_path or "data/ToolSandbox.json"

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def load_questions(self) -> List[ToolSandboxQuestion]:
        """Parse *all* scenarios in the dataset and return formatted questions."""
        import logging, json
        from pathlib import Path

        # Prepare file logger (no console)
        log_path = Path("logs") / "toolsandbox_inputs.jsonl"
        log_path.parent.mkdir(exist_ok=True)
        # We open once at beginning for overwrite
        log_file = log_path.open("w", encoding="utf-8")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"ToolSandbox dataset not found at {self.data_path}.")

        with open(self.data_path, "r", encoding="utf-8") as f:
            scenarios = json.load(f)

        questions: List[ToolSandboxQuestion] = []
        for scenario in scenarios:
            try:
                questions.append(self._format_scenario(scenario))
                # Persist raw scenario for auditing
                try:
                    log_file.write(json.dumps(scenario, ensure_ascii=False) + "\n")
                except Exception:
                    pass
            except Exception as e:
                scenario_name = scenario.get("scenario_name", "<unknown>")
                print(f"[ToolSandboxLoader] Skipping scenario '{scenario_name}': {e}")
                continue

        # Flush and close log file before returning
        try:
            log_file.close()
        except Exception:
            pass

        return questions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _format_scenario(self, scenario: Dict[str, Any]) -> ToolSandboxQuestion:
        """Convert a *single* scenario dict into a ToolSandboxQuestion."""

        scenario_name: str = scenario.get("scenario_name", "unknown_scenario")
        starting_ctx: Dict[str, Any] = scenario["data"]["starting_context"]

        # Attempt to capture system prompts for additional context
        sandbox_msgs_full: List[Dict[str, Any]] = starting_ctx.get("_dbs", {}).get("SANDBOX", [])

        system_to_agent = next(
            (
                m.get("content", "")
                for m in sandbox_msgs_full
                if m.get("sender") == "SYSTEM" and m.get("recipient") == "AGENT"
            ),
            "",
        )

        task_description = next(
            (
                m.get("content", "")
                for m in sandbox_msgs_full
                if m.get("sender") == "SYSTEM" and m.get("recipient") == "USER"
            ),
            "",
        )

        # ------------------------------------------------------------------
        # 1) Extract user *instruction* – we take the first USER→AGENT message
        # ------------------------------------------------------------------
        sandbox_msgs: List[Dict[str, Any]] = sandbox_msgs_full
        user_first_utterance = next(
            (
                msg.get("content", "")
                for msg in sandbox_msgs
                if msg.get("sender") == "USER" and msg.get("recipient") == "AGENT"
            ),
            "",
        )

        # Compose a richer instruction block for the judge
        instruction_parts = [
            f"Scenario: {scenario_name}",
        ]
        if task_description:
            instruction_parts.append("\nUSER Task Description (system→user):\n" + task_description.strip())
        if system_to_agent:
            instruction_parts.append("\nSystem Policy (system→agent):\n" + system_to_agent.strip())
        if user_first_utterance:
            instruction_parts.append("\nFirst user utterance:\n" + user_first_utterance.strip())

        rich_instruction = "\n\n".join(instruction_parts)

        # ------------------------------------------------------------------
        # 2) Convert *all* SANDBOX messages into OpenAI conversation format
        # ------------------------------------------------------------------
        gt_conversation: List[Dict[str, Any]] = []
        for msg in sandbox_msgs:
            # Skip utterly empty stub rows
            if msg.get("sender") is None and (msg.get("content") is None or msg.get("content") == ""):
                continue

            func_name = msg.get("openai_function_name")

            if func_name:  # Assistant tool invocation
                gt_conversation.append(
                    {
                        "role": "assistant",
                        "function_call": [
                            {
                                "name": func_name,
                                "arguments": {},
                            }
                        ],
                    }
                )

                # Observation immediately after (if available)
                obs_content = msg.get("tool_trace") or msg.get("content") or "<tool output omitted>"
                gt_conversation.append({"role": "observation", "content": obs_content})
                continue

            # EXECUTION_ENVIRONMENT messages are considered pure observations (environment response)
            if msg.get("recipient") == "EXECUTION_ENVIRONMENT" or msg.get("sender") == "EXECUTION_ENVIRONMENT":
                gt_conversation.append({"role": "observation", "content": msg.get("content", "")})
                continue

            # Regular conversational message
            role = self.ROLE_MAPPING.get(msg.get("sender"), "system")
            content = msg.get("content", "") or ""
            if content.strip():  # keep non-empty
                gt_conversation.append({"role": role, "content": content})

        # ------------------------------------------------------------------
        # 3) Available functions – minimal schema list
        # ------------------------------------------------------------------
        tool_allow_list = starting_ctx.get("tool_allow_list", []) or []
        functions: List[Dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"ToolSandbox allowed function {name}",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    },
                },
            }
            for name in tool_allow_list
        ]

        # ------------------------------------------------------------------
        # Assemble result
        # ------------------------------------------------------------------

        # Determine expected textual output – take last assistant message content
        expected_output = next(
            (
                m.get("content", "")
                for m in reversed(sandbox_msgs_full)
                if m.get("sender") == "AGENT" and (m.get("openai_function_name") is None)
                and (m.get("content") and m.get("content").strip())
            ),
            "",
        )

        return ToolSandboxQuestion(
            question_id=scenario_name,
            instruction=rich_instruction,
            gt_conv_traj=gt_conversation,
            available_function_list=functions,
            benchmark=Benchmark.TOOL_SANDBOX,
            expected_output=expected_output,
            meta={"raw": scenario},
        )