# IN BFCL, agent system prompt includes available_function_list

import json

FILTERING_PROMPT = """

You are an expert evaluator for BFCL, a benchmark designed to assess an agent's multi-turn and multi-step function calling abilities.
Your task is to determine if a given benchmark sample has a fundamental flaw in its user prompt, environment, or ground-truths, which would make it unable to be incorporated in the evaluation.


You will be provided with the following information:
* **Instruction**: The description of the task given to the agent. 
* **Agent System Prompt**: the system prompt used to initialize the agent model. This may contain a specific instruction on the answer style, domain-specific policy that the agent needs to follow, a list of available functions and their schema (in JSON format), etc.
* **Available Functions**: a list of functions available for the agents and their schema. Note that functions related to the file system (e.g., `wc`, `ls`, `sort`, etc.), if provied, abide by the standard Unix semantics unless specified otherwise directly.
* **Missed Functions**: This is only provided in the category `multi_turn_miss_func`. This is the function that is not provided to the agent at the first turn, but will be provided after a specified number of agent responses.  
* **Initial Configuration**: The initial environment setup and conditions before the task begins. 
* **Ground-Truth Milestone Function Call Trajectory**: the provided ground-truth trajectory of crucial function calls. When this is empty or None, it means that the agent needs to call nothing to be scored as correct. Note that entries with `"role": "tool"` are the results of the directly preceding agent tool calls.
A sample is **flawed** if it exhibits one or more of the issues described below.


## Flaw Categories

Below is the categorization of benchmark issues, outlined according to its **relevant benchmark component**. A sample is considered flawed if it has one or more of the issues below.

### Environment

This category covers flaws within the agent's operating environment—the tools and API results—which can make a task unsolvable regardless of the agent's logic.

* Insufficient toolsets: The environment does not provide the necessary tools for the agent to complete the task. 
  * Look for:  
    * Empty Function List: No functions are provided but the test expects function calls.
    * Missing Core Functionality: Essential functions for completing the task are absent from the functions list.

* Flawed tool design: Tools exist, but their interface or description makes them unusable or misleading.
  * Look for:  
    * Incompatible Parameters: Functions exist but their parameters don't match requirements.
    * Environment–Function Mismatch: Available functions don't match the described environment.

### Ground-Truth

This category addresses errors in the provided ground-truth trajectory, where the supposed correct solution is itself incorrect, forcing any correct agent to fail the evaluation.

* Malformed function calls: A technical error where a ground-truth function call violates the provided API schema.
  * Note that If a function has only one parameter, it may be invoked without using a keyword argument. This is not a flaw. e.g., `sort('final_report.pdf')`

* Incorrect function calls: A function call is syntactically valid but logically flawed. The function choice or a parameter value contradicts the user's request or the context from previous steps.
  * Unjustified/Hallucinated Parameters: A value (e.g., a file name, user name) that appears without any grounding context. 
  * Contradictory: A value that directly contradicts a constraint in the user's prompt. However, it is NOT a flaw if there is any chance that the agent's action was a necessary alternative due to constraints like an insufficient budget or a lack of available seats.
  * Policy Violation: A function call in the ground truth trajectory directly violates the provided system policy.
  * Misspelled or Incorrectly Identified Parameter Values: A misspelled name or an ID/slug that points to the wrong entity.

* Redundant/ungrounded function calls: The ground truth function call trajectory consists of function calls that are redundant in solving the task, ungrounded by the context, or irrelevant in solving the task.
  * Irrelevant tool call: A function call in the ground truth trajectory is totally irrelevant to the task or belongs to a completely different domain. 
  * Redundant tool call: A function call that is not necessary in solving the task. 

## Crucial Rules

### Actively Reconstruct the Conversation

The ground-truth trajectory only contains crucial function calls from the agent's response. It intentionally omits agents responses in natural language (e.g., confirmations, request, clarifications, or follow-up questions), or less important and obvious function calls, such as `get_user_id`.
Your task is to find undeniable flaws. Therefore, you MUST operate under the following assumption:

* For example, the user may provide an additional information or permits to use a new function after an agent prints an empty response with no tool call. This is not a flaw, since the agent would have requested for the information or the function, though it is not revealed in the provided ground truth.
* If a sequence of function calls can be justified by a plausible, un-shown conversation, then it is NOT a flaw.

### Do NOT Judge Tool Results

The tool results in the ground-truth trajectory are automatically generated via actually calling the corresponding tools, and are not subject to judgement. Flaws in tool results should NOT be the reason you mark a sample as flawed.
 
## Evaluation and Output Format
Carefully analyze the provided sample. Think step-by-step to determine if the ground-truth trajectory is a correct and logical solution to the user's prompt.

Your final output must be a JSON object with the following structure, with no additional commentary:

```json
{{
  "reasoning": "Provide a clear, step-by-step explanation for your decision. If the sample is flawed, specify what is incorrect and why it contradicts the user's prompt, system policies, or the user's role. If it is not flawed, briefly explain why the sample is valid.",
  "reasoning_summary": "A shorter rationale for your decision. If the sample is not flawed, just mention that it is not flawed. If it is flawed, specify the issue concisely. e.g., The ground truth books a connecting flight, but the user requested a direct flight.",
  "error_category": "The category that corresponds to the issue. e.g., \"Incorrect function calls\". If the sample is not flawed, use \"Not Flawed\".",
  "is_flawed": <true or false>
}}
```

## Sample to be evaluated

### Category 
* category: {category}
* subcategory: {subcategory}

### Instruction

{instruction}

### Agent System Prompt

{system_prompt}

{missed_function}

### Initial Config

{initial_pwd_description}
```json
{initial_config}
```

### Tool Default States

{default_states}

### Ground-Truth Function Call(s):
```json
{gt_conv_traj}
```
"""

# Optional: scoring template (only used if you run with scoring enabled)
SCORING_PROMPT = """
You are scoring the task difficulty of a BFCL single-turn sample.
Consider: clarity of mapping from instruction to tool(s), parameter complexity, and ambiguity.
Return a JSON array of objects with fields: dimension, reasoning, score (1-5).

Example output:
[
  {{"dimension": "tool selection difficulty", "reasoning": "…", "score": 3}},
  {{"dimension": "parameter complexity", "reasoning": "…", "score": 3}}
]
"""

# Test
if __name__ == "__main__":
    from src.utils.types import Benchmark
    from src.bench_loaders import get_bench_loader
    import argparse
    import sys
    import os

    parser = argparse.ArgumentParser(description="Generate formatted prompt for BFCL")
    parser.add_argument("-q", "--question_id", type=str,
                       help="Question ID to format")
    parser.add_argument("--save-all", action="store_true",
                       help="Save all prompts to bfcl_prompts folder")
    args = parser.parse_args()

    # Test BFCL
    bfcl_loader = get_bench_loader(Benchmark.BFCL)()
    bfcl_questions = bfcl_loader.load_questions()

    def to_pretty_json(value):
        if value is None:
            return "null"
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)

    def as_text(value, fallback=""):
        return value if value is not None else fallback

    if args.save_all:
        # Save all prompts to bfcl_prompts folder
        os.makedirs("bfcl_prompts", exist_ok=True)

        for question in bfcl_questions:
            bfcl_prompt = FILTERING_PROMPT.format(
                category=getattr(question, 'category', ''),
                subcategory=getattr(question, 'subcategory', ''),
                instruction=question.instruction,
                system_prompt=getattr(question, 'system_prompt', ''),
                initial_pwd_description=as_text(getattr(question, 'initial_pwd_description', ''), ''),
                initial_config=to_pretty_json(getattr(question, 'initial_config', None)),
                default_states=as_text(getattr(question, 'default_states', ''), '* (no tool default states provided)'),
                gt_conv_traj=to_pretty_json(getattr(question, 'gt_conv_traj', [])),
                missed_function=as_text(getattr(question, 'missed_function', ''), '')
            )

            filename = f"bfcl_prompts/{question.question_id}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(bfcl_prompt)

        print(f"Saved {len(bfcl_questions)} prompts to bfcl_prompts/ folder")
    else:
        # Original single question functionality
        if not args.question_id:
            print("Error: --question_id is required when not using --save-all")
            sys.exit(1)

        bfcl_sample = None
        for question in bfcl_questions:
            if question.question_id == args.question_id:
                bfcl_sample = question
                break

        if bfcl_sample is None:
            print(f"Error: Could not find question with ID {args.question_id}")
            sys.exit(1)

        # Generate formatted prompt using the FILTERING_PROMPT template
        bfcl_prompt = FILTERING_PROMPT.format(
            category=getattr(bfcl_sample, 'category', ''),
            subcategory=getattr(bfcl_sample, 'subcategory', ''),
            instruction=bfcl_sample.instruction,
            initial_config=to_pretty_json(bfcl_sample.initial_config),
            initial_pwd_description=as_text(bfcl_sample.initial_pwd_description, ''),
            system_prompt=getattr(bfcl_sample, 'system_prompt', ''),
            default_states=as_text(bfcl_sample.default_states, '* (no tool default states provided)'),
            gt_conv_traj=to_pretty_json(bfcl_sample.gt_conv_traj),
            missed_function=as_text(bfcl_sample.missed_function, '')
        )

        with open("bfcl_formatted_prompt.txt", "w", encoding="utf-8") as f:
            f.write(bfcl_prompt)

        print("BFCL formatted prompt saved to bfcl_formatted_prompt.txt")
