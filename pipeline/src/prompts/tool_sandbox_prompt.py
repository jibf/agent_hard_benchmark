import json

FILTERING_PROMPT = """
You are an expert evaluator for **NexusBench**, a benchmark designed to assess precise tool-use across diverse tool-use tasks.
Your task is to determine if a given benchmark sample has a fundamental flaw in its user prompt, environment, or ground-truths, which would make it unable to be incorporated in the evaluation.

You will be provided with the following context for each sample:
* **User System Prompt**: The system-level instructions supplied to the model that simulates the user.
* **Task Description/Instructions**: The prompt or scenario given to the model that simulates user. You need to expect how the model that simulates the user would behave given this instruction.
* **Available Tool List**: a list of functions available for the agents and their schema.
* **Milestones**: actions or final database states that is essential for the task-solving trajectory to be marked as correct.
* **Minefields**: actions or final database states that are undesirable and therefore makes the task-solving trajectory to be marked as incorrect.
* **Initial State**: The initial environment setup and conditions before the task begins. This defines the starting state of the system.

## Flaw Categories

Below is the categorization of benchmark issues, outlined according to its **relevant benchmark component**. A sample is considered flawed if it has one or more of the issues below.

### User

* Ambiguous user prompt: the task description/instructions are underspecified 

### Environment

This category covers flaws within the agent's operating environment—the tools and API results—which can make a task unsolvable regardless of the agent's logic.

* Flawed function design: the naming or the description of an available function is misleading or contradicts its actual functionality.
  * Example: A function named `vt_get_votes_on_ip_address` provides "example.com" as an example for its argument value in its schema. 
  * Note that some samples use anonymized tool names, e.g., utilities_1 or reminder_0, or lack the tool description. This is part of the sample's intentional design and therefore is not a flaw. 

### Ground Truth and Evaluation System

This category addresses errors in the provided evaluation method (milestones and minefields), which may force any correct agent to fail the evaluation.

* Incorrect tool call/final state: A tool call or the final state is logically flawed. For example, the milestone/minefield functions or expected final state contradicts the user's request or the context. 
  * Unjustified/Hallucinated Parameters: The anticipated parameter value that appears without any grounding context. 
  * Contradictory: A value that directly contradicts a constraint in the user's prompt. However, it is NOT a flaw if there is any chance that the agent's action was a necessary alternative due to constraints like an insufficient budget or a lack of available seats.


## Crucial Note

In ToolSandbox, the tool call sequence and the final state is assessed to determine if the task-completion trajectory is correct or incorrect. To this end, milestones and minefields are utilized; a trajectory is correct if and only if it contains all the entities(tool call and DB state) that match milestone and contains no entities that match minefields.
Beware that the provided milestone sequence is not the full sequence of all tool calls; it only contains those that are utilized for the evaluation. Therefore, you should not judge a sample as incorrect because of lack of specific function call in the milestone sequence. Imagine a possible conversation history that would justify the ground truth milestone function call trajectory. When you contemplate of a plausible trajectory, note that the user can make a request that is not mentioned in the prompt, guided by the agent. Flag a sample as flawed ONLY if a function call is impossible to justify, even with a hypothetical conversation. 

Also, note that ToolSandbox utilize various matching methods to compare the predicted tool call and final states to milestones and minefields as follows:

### Column-level Matching Methods
- **`column_exact_match_similarity`** — Returns 0/1 similarity based on exact equality (`==`).  
  If `value is None`, checks with `.is_null()`.
- **`column_close_similarity`** — For numeric types, returns 1 if `is_close()` within `atol_dict` tolerance, else 0.
- **`column_one_similarity`** — Always returns 1.0 (treats all as perfect match).
- **`column_contains_similarity`** — For string columns, returns 1 if `.str.contains_any([value])` is true, else 0.
- **`column_tool_trace_exact_match_similarity`** — Parses JSON tool traces and checks:
  - same `tool_name`
  - all `arguments` close via `is_close()`
  - returns 1 if **any** golden trace matches, otherwise 0.
- **`column_rouge_l_similarity`** — Uses `rouge_scorer.RougeScorer` to compute ROUGE-L F1 score between strings (returns [0,1] float).

### Snapshot-level Matching Methods
- **`snapshot_similarity`** —  
  Returns 0 if row count or columns differ.  
  Otherwise:
  - computes per-row geometric mean of column similarities,  
  - builds cost matrix (`-log(similarity)`),  
  - uses `linear_sum_assignment` (Hungarian algorithm) to find optimal 1-to-1 row mapping maximizing overall similarity.
- **`addition_similarity`** —  
  Returns 0 if `reference_snapshot` rows are not fully contained in `snapshot`.  
  Otherwise compares the **anti-join difference** with `target_dataframe` using `snapshot_similarity`.
- **`removal_similarity`** —  
  Inverse of `addition_similarity` (swaps `snapshot` and `reference_snapshot`).
- **`update_similarity`** —  
  Returns 0 if row counts differ.  
  Otherwise, compares the **anti-join difference** between `snapshot` and `reference_snapshot` against `target_dataframe`.
- **`tool_trace_dependant_similarity`** —  
  SANDBOX-only.  
  Extracts values from `reference_snapshot.tool_trace` using a provided extractor.  
  For each extracted value:
  - fills into `target_dataframe` (either `tool_trace` or `content`),  
  - computes `snapshot_similarity`,  
  - takes the **maximum similarity** across all filled variants.
- **`guardrail_similarity`** —  
  Returns 1 if `snapshot.equals(reference_snapshot)` (identical), else 0.


## Evaluation and Output Format
Carefully analyze the provided sample. Think step-by-step to determine if the ground-truth trajectory is a correct and logical solution to the user's prompt.

Your final output must be a JSON object with the following structure, with no additional commentary:

```json
{{
  "reasoning": "<step-by-step explanation focusing on the first flaw or why the ground-truth is correct>",
  "reasoning_summary": "<one-sentence summary>",
  "error_category": "<one of the categories above>",
  "is_flawed": <true_or_false>
}}
```

## Target Sample

### Task Description/Instructions

{instruction}

### User System Prompt

{user_system_prompt}

### Available Function List

```json
{available_function_list}
```

### Initial Databases

```json
{initial_databases}
```

### Milestones
```json
{milestones}
```

### Minefields 
```json
{minefields}
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

### Initial Databases
```json
{initial_databases}
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
    instruction = (getattr(question, "instruction", "") or "").strip()
    available_function_list = _format_json_block(getattr(question, "available_function_list", []))
    gt_conv_traj = _format_json_block(getattr(question, "gt_conv_traj", []))
    expected_output = (getattr(question, "expected_output", "") or "").strip()
    user_system_prompt = (getattr(question, "user_system_prompt", "") or "").strip()

    initial_databases_raw = getattr(question, "initial_databases", {})
    if initial_databases_raw is None:
        initial_databases_raw = {}
    initial_databases = _format_json_block(initial_databases_raw)

    milestones_raw = getattr(question, "milestones", [])
    if milestones_raw is None:
        milestones_raw = []
    milestones = _format_json_block(milestones_raw)

    minefields_raw = getattr(question, "minefields", [])
    if minefields_raw is None:
        minefields_raw = []
    minefields = _format_json_block(minefields_raw)

    payload = {
        "instruction": instruction,
        "available_function_list": available_function_list,
        "gt_conv_traj": gt_conv_traj,
        "expected_output": expected_output,
        "user_system_prompt": user_system_prompt,
        "initial_databases": initial_databases,
        "milestones": milestones,
        "minefields": minefields,
    }
    if prompt_type == "filtering":
        template = FILTERING_PROMPT
    elif prompt_type == "scoring":
        template = SCORING_PROMPT
    else:
        raise ValueError(f"Unsupported prompt type: {prompt_type}")

    return template.format(**payload)


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
