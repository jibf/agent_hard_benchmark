import json

FILTERING_PROMPT = """
You are a rigorous evaluator for the **ToolSandbox** benchmark (Apple, 2024), a stateful, multi-turn tool-use benchmark.  
Your task is to decide if the provided *ground-truth* conversation trajectory is
**flawed** given the user's intent and the available tool schemas.

ToolSandbox specifics to keep in mind:
• Tasks are evaluated via **milestones** (essential actions) and **minefields** (undesirable actions), not just final text.
• Scenarios are **stateful**: tools can mutate world/DB state and later steps may depend on earlier ones.

────────────────────────────────────────────────────────────────────────
You will receive **three** inputs:

1. **User Prompt** – the first instruction from the human user.
2. **Available Function List** – JSON schema of all tools.
3. **Ground-Truth Conversation** – the reference assistant messages *including*
   tool calls (and subsequent tool observations).

────────────────────────────────────────────────────────────────────────
Your task is to decide whether the **ground-truth (GT) milestone trajectory** for a scenario is **flawed**, focusing these issues:

1. **Ambiguous User Query / Over-Specific Ground Truth**  
   – The user’s request is vague or underspecified
   – The GT enforces an answer at a **more specific level** than requested (e.g., requires a numeric distance when the user just asked *“Where is X?”*).
   – In such cases, multiple valid answers exist; enforcing one path is unfair → mark as flawed.

2. **Unneeded Milestone**  
   – The GT includes intermediate tool calls designated as milestones that are **not logically necessary** to answer the query.  
   – Example: requiring `get_current_timestamp` before retrieving the first message, even though the query *“What’s my first text?”* can be answered directly with `search_messages`.  
   – Each milestone must be **indispensable** for reaching the answer. If not, mark as flawed.

────────────────────────────────────────────────────────────
RULES
• Always check: Does the **level of detail in the GT final answer** match exactly what the user asked for? If it overshoots (too detailed), it is flawed.  
• Always check: Could the task be solved **without one or more of the required milestones**? If yes, it is flawed.  
• Ignore minor wording differences or plausible missing conversation turns.  
• Stop at the **earliest undeniable flaw**.

────────────────────────────────────────────────────────────
FLAW CATEGORIES (choose one)
• Ambiguous User Query / Over-Specific Ground Truth  
• Unneeded Milestone  
• Not Flawed

────────────────────────────────────────────────────────────────────────
Output exactly the following JSON (no extra keys, no commentary):

```json
{{
  "reasoning": "<step-by-step explanation focusing on the first flaw or why the ground-truth is correct>",
  "reasoning_summary": "<one-sentence summary>",
  "error_category": "<one of the categories above>",
  "is_flawed": <true_or_false>
}}
```

## Target Sample

### User Prompt
```
{instruction}
```

### Available Function List
```json
{available_function_list}
```

### Ground-Truth Conversation
*Messages with `"role": "observation"` are tool outputs for the function call immediately before them.*
```json
{gt_conv_traj}
```

### Expected Final Assistant Message
```
{expected_output}
```
"""


SCORING_PROMPT = """
You are an expert evaluator for the **ToolSandbox** benchmark.  Assess how well
each sample measures advanced tool-use capabilities along five dimensions.  Use
the scale **1 (poor) – 5 (excellent)**.  Provide concise, critical reasoning for
each score.

Dimensions (fixed order):

1. tool necessity – Does solving the task *fundamentally require* the provided
   tools?
2. planning and context depth – Complexity of state tracking & dependency
   reasoning across turns.
3. parameter generation – Difficulty of deriving correct parameters from
   context (dates, IDs, dynamic values, etc.).
4. tool selection difficulty – How challenging is it to pick the correct tool
   amidst plausible distractors?
5. real-world applicability – How representative is the task of real user
   scenarios?

───────────
You will receive:
* **User Prompt**
* **Available Function List**
* **Ground-Truth Conversation** (including milestones)

───────────
Output a JSON **array** (no extra commentary) following this template:

```
[
  {{
    "dimension": "tool necessity",
    "reasoning": "...",
    "score": <1-5>
  }},
  {{
    "dimension": "planning and context depth",
    "reasoning": "...",
    "score": <1-5>
  }},
  {{
    "dimension": "parameter generation",
    "reasoning": "...",
    "score": <1-5>
  }},
  {{
    "dimension": "tool selection difficulty",
    "reasoning": "...",
    "score": <1-5>
  }},
  {{
    "dimension": "real-world applicability",
    "reasoning": "...",
    "score": <1-5>
  }}
]
```

## Target Sample

### User Prompt
```
{instruction}
```

### Available Function List
```json
{available_function_list}
```

### Ground-Truth Conversation
```json
{gt_conv_traj}
```

### Expected Final Assistant Message
```
{expected_output}
```
"""


def _format_json_block(data):
    if isinstance(data, str):
        return data
    return json.dumps(data, indent=2, ensure_ascii=False)


def _build_prompt(question, prompt_type):
    payload = {
        "instruction": question.instruction or "",
        "available_function_list": _format_json_block(question.available_function_list),
        "gt_conv_traj": _format_json_block(question.gt_conv_traj),
        "expected_output": (question.expected_output or "").strip(),
    }
    if prompt_type == "filtering":
        return FILTERING_PROMPT.format(**payload)
    if prompt_type == "scoring":
        return SCORING_PROMPT.format(**payload)
    raise ValueError(f"Unsupported prompt type: {prompt_type}")


def _safe_stem(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Render ToolSandbox prompts similar to NexusBench/Tau-Bench helpers."
    )
    parser.add_argument(
        "-q",
        "--question_id",
        type=str,
        help="Scenario name to render (e.g., sample_toolscenario_01).",
    )
    parser.add_argument(
        "-p",
        "--prompt-type",
        choices=["filtering", "scoring"],
        default="filtering",
        help="Prompt template to render (default: filtering).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file path, or directory when --save-all is used.",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Render prompts for all ToolSandbox samples into a directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import os

    from src.utils.types import Benchmark
    from src.bench_loaders import get_bench_loader

    args = _parse_args()

    loader_cls = get_bench_loader(Benchmark.TOOL_SANDBOX)
    toolsandbox_loader = loader_cls()
    questions = toolsandbox_loader.load_questions()

    prompt_type = args.prompt_type

    if args.save_all:
        output_dir = args.output or f"toolsandbox_{prompt_type}_prompts"
        os.makedirs(output_dir, exist_ok=True)

        saved = 0
        for question in questions:
            prompt_text = _build_prompt(question, prompt_type)
            file_stem = _safe_stem(question.question_id)
            filename = f"{file_stem}_{prompt_type}_prompt.txt"
            output_path = os.path.join(output_dir, filename)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(prompt_text)
            saved += 1

        print(f"Saved {saved} ToolSandbox {prompt_type} prompts to {output_dir}/")
    else:
        if not args.question_id:
            print("Error: --question_id is required unless --save-all is specified.")
        else:
            selected = next(
                (question for question in questions if question.question_id == args.question_id),
                None,
            )
            if not selected:
                print(f"No ToolSandbox sample found with ID '{args.question_id}'")
            else:
                print(
                    f"ToolSandbox sample found - ID: {selected.question_id}"
                )
                prompt_text = _build_prompt(selected, prompt_type)
                output_path = (
                    args.output
                    if args.output
                    else f"tool_sandbox_{_safe_stem(selected.question_id)}_{prompt_type}_prompt.txt"
                )
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(prompt_text)
                print(f"ToolSandbox {prompt_type} prompt saved to {output_path}")
