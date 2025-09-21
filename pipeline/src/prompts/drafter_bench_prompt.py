import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.bench_loaders import get_bench_loader
from src.utils.types import Benchmark, DrafterBenchQuestion, LLMJudgeStep


FILTERING_PROMPT = """You are an expert evaluator for Drafter-Bench, a benchmark that measures whether an agent can draft the correct function call for a user's request under a well-defined tool schema.

Your job is to decide if the provided ground-truth function call draft is fundamentally flawed. If the draft is flawed, the sample cannot be used for evaluation.

You will receive the following information:
* **User Instruction** – the request that should be satisfied by the ground-truth draft.
* **Agent System Prompt** – policy and tooling constraints the assistant must obey.
* **Ground-Truth Draft** – the function call (often multi-line code) labeled as correct.
A sample is **flawed** if it exhibits one or more of the issues described below.

## Flaw Categories


Below is the categorization of benchmark issues, outlined according to its **relevant benchmark component**. A sample is considered flawed if it has one or more of the issues below.

### Environment

This category covers flaws within the agent's operating environment—the tools and API results—which can make a task unsolvable regardless of the agent's logic.

* Insufficient toolsets: the environment does not provide the necessary tools (functions), making the agent impossible to solve the task even with a combination of multiple tools and reasoning.
  * Example: A user asks for an advanced file manipulation, while the environment only provides basic tools like `mk` or `ls`.

* Flawed function design: the naming or the description of an available function is misleading or contradicts its actual functionality.
  * Example: A function named `vt_get_votes_on_ip_address` provides "example.com" as an example for its argument value in its schema. 

### Ground-Truth

This category addresses errors in the provided ground-truth trajectory, where the supposed correct solution is itself incorrect, forcing any correct agent to fail the evaluation.

* Incorrect function calls: A function call is syntactically valid but logically flawed. The function choice or a parameter value contradicts the user's request or the context from previous steps.
  * Unjustified/Hallucinated Parameters: A value (e.g., a date, a coordinate) that appears without any grounding context. For example, searching for a hotel on a date that was not returned by a preceding flight search.
  * Contradictory: A value that directly contradicts a constraint in the user's prompt. However, it is NOT a flaw if there is any chance that the agent's action was a necessary alternative due to constraints like an insufficient budget or a lack of available seats.
  * Policy Violation: A function call in the ground truth trajectory directly violates the provided system policy. Example: The ground truth where the agent calls a specific function twice, although it is mentioned in the system policy that the function can only be called once.

* Redundant/ungrounded function calls: The ground truth function call trajectory consists of function calls that are redundant in solving the task, ungrounded by the context, or irrelevant in solving the task.
  * Irrelevant tool call: A function call in the ground truth trajectory is totally irrelevant to the task or belongs to a completely different domain. Example: agent calls a function to reserve a flight, though it was asked to process product exchange.
  * Redundant tool call: A function call that is not necessary in solving the task. Example: the agent is asked to search for attractions until it finds one that meets a certain condition; However, the agent performs the search in an arbitrary order, resulting in an excessive number of function calls.


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

## Target Sample

### User Instruction

{instruction}

### Agent System Prompt
```
{agent_system_prompt}
```

### Ground-Truth Draft
```python
{groundtruth}
```

"""

SCORING_PROMPT = ""


def build_filtering_prompt(question: DrafterBenchQuestion) -> str:
    """Render the filtering prompt for a single Drafter-Bench sample."""

    drafter_context = {}
    if question.meta and isinstance(question.meta, dict):
        drafter_context = question.meta.get("drafter_bench_context", {}) or {}

    formatted_context = _format_json(drafter_context, fallback="null")
    formatted_functions = _format_json(question.available_function_list, fallback="[]")
    formatted_conversation = _format_json(question.gt_conv_traj, fallback="[]")

    return FILTERING_PROMPT.format(
        task_name=question.task_name,
        question_id=question.question_id,
        agent_system_prompt=question.agent_system_prompt,
        instruction=question.instruction,
        groundtruth=question.groundtruth,
        gt_conv_traj=formatted_conversation,
        available_function_list=formatted_functions,
        drafter_bench_context=formatted_context,
    )


def build_prompt(question: DrafterBenchQuestion, step: LLMJudgeStep) -> str:
    """Entry point used by the prompt formatter infrastructure."""

    if step == LLMJudgeStep.SPECIFIC_FILTER:
        return build_filtering_prompt(question)

    if step == LLMJudgeStep.SCORE:
        if SCORING_PROMPT:
            return SCORING_PROMPT
        raise ValueError("Scoring prompt is not defined for Drafter-Bench.")

    raise ValueError(f"Unsupported judge step for Drafter-Bench: {step.value}")


def _format_json(value, fallback: str) -> str:
    """Serialize dictionaries/lists for prompt readability."""

    if value in (None, ""):
        return fallback

    if isinstance(value, (list, dict)):
        if not value:
            return fallback
        return json.dumps(value, indent=2)

    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview the formatted Drafter-Bench filtering prompt for a question."
    )
    parser.add_argument(
        "-q",
        "--question-id",
        type=str,
        help="Question identifier in the form <task_type>-<id>. Defaults to the first sample if omitted.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        help="Optional path to save the rendered prompt to a file.",
    )

    args = parser.parse_args()

    loader = get_bench_loader(Benchmark.DRAFTER_BENCH)()
    questions = loader.load_questions()

    if not questions:
        print("Error: No Drafter-Bench questions were loaded.")
        sys.exit(1)

    target = None
    if args.question_id:
        for question in questions:
            if question.question_id == args.question_id:
                target = question
                break
        if target is None:
            print(f"Error: Could not find question with ID {args.question_id}.")
            sys.exit(1)
    else:
        target = questions[0]

    prompt = build_filtering_prompt(target)

    if args.save_path:
        args.save_path.write_text(prompt, encoding="utf-8")
        print(f"Prompt saved to {args.save_path}")

    print(prompt)


if __name__ == "__main__":
    main()
