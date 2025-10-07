FILTERING_PROMPT = """
You are an expert evaluator for **NexusBench**, a benchmark designed to assess precise tool-use across diverse tool-use tasks.
Your task is to determine if a given benchmark sample has a fundamental flaw in its user prompt, environment, or ground-truths, which would make it unable to be incorporated in the evaluation.

You will be provided with the following context for each sample:
* **System Prompt**: The system-level instructions supplied to the agent model.
* **User Prompt**: The user's task description and instructions.
* **Ground Truth Output**: The benchmark’s reference answer. This can be either single or a chain of function calls, or final output expected.
* **Sub-benchmark Name**: The specific NexusBench sub-benchmark the sample belongs to.
* **Sub-benchmark Description**: The detailed description of the sub-benchmark the sample belongs.
* **Tool Schemas** – complete JSON definitions for every callable tool.


A sample is **flawed** if it exhibits one or more of the issues described below.

## Flaw Categories

Below is the categorization of benchmark issues, outlined according to its **relevant benchmark component**. A sample is considered flawed if it has one or more of the issues below.

### User

* Vague instruction: The user's prompt is too ambiguous or underspecified for a single, correct function call, yet the benchmark expects one.
  * Example: The user asks "Where's the Golden Gate Bridge?", but the ground truth arbitrarily expects a distance calculation from a specific, unmentioned point.
  * However, ONLY flag this if the user query cannot be reasonably inferred from the available tools and system instructions. If you are unsure, flag it as valid. 


### Environment

This category covers flaws within the agent's operating environment—the tools and API results—which can make a task unsolvable regardless of the agent's logic.

* Insufficient toolsets: the environment does not provide the necessary tools (functions), making the agent impossible to solve the task even with a combination of multiple tools and reasoning.
  * Example: A user asks for an advanced file manipulation, while the environment only provides basic tools like `mk` or `ls`.

* Flawed function design: the naming or the description of an available function is misleading or contradicts its actual functionality.
  * Example: A function named `vt_get_votes_on_ip_address` provides "example.com" as an example for its argument value in its schema. 


### Ground-Truth

This category covers flaws in the pre-defined ground truth of the sample. As described previously, the ground truth can be a single function call, a chain of function calls, or a numerical value that is the outcome of the ground truth function calls. If the ground truth is a single number, ignore this error type.

* Malformed function calls: A technical error where a ground-truth function call violates the provided API schema.
  * Example: A parameter requires a string but is given a number (e.g., dest_id: 123 instead of dest_id: "123"), a required parameter is missing, the function name is wrong, or an enum-typed parameter has the value not in the pre-defined set.

* Incorrect function calls: A function call is syntactically valid but logically flawed. The function choice or a parameter value contradicts the user's request or the context from previous steps.
  * Unjustified/Hallucinated Parameters: A value (e.g., a date, a coordinate) that appears without any grounding context. For example, searching for a hotel on a date that was not returned by a preceding flight search.
  * Contradictory Parameter Values: A value that directly contradicts a constraint in the user's prompt. For example, using the latitude of a hotel and the longitude of an airport to define a search coordinate, which is logically inconsistent.
  * Misspelled or Incorrectly Identified Parameter Values: A misspelled name or an ID/slug that points to the wrong entity (e.g., selecting the wrong airport ID).


* Redundant/ungrounded function calls: The ground truth function call trajectory consists of function calls that are redundant in solving the task, ungrounded by the context, or irrelevant in solving the task.
  * Irrelevant tool call: A function call in the ground truth trajectory is totally irrelevant to the task or belongs to a completely different domain. Example: agent calls a function to reserve a flight, though it was asked to process product exchange.
  * Redundant tool call: A function call that is not necessary in solving the task. Example: the agent is asked to search for attractions until it finds one that meets a certain condition; However, the agent performs the search in an arbitrary order, resulting in an excessive number of function calls.

## Crucial Note: Sub-benchmark-specific note for TMIHallucination
  
Some samples in TMIHallucination deliberately ship tool schemas where almost every parameter is flagged as `required`. That alone is not a flaw—those schemas are meant to pressure-test whether the ground truth omits the explicitly banned argument. Treat the usual NexusBench rules as primary; do **not** mark a sample flawed solely because a schema lists an excessive number of required fields, or because the ground-truth call omits required fields. Ignore this note if the sample belongs to other sub-benchmarks.


## Evaluation and Output Format
Carefully analyze the provided sample. Think step-by-step to determine if the ground-truth trajectory is a correct and logical solution to the user's prompt.

Your final output must be a JSON object with the following structure, with no additional commentary:

```json
{{
  "reasoning": "Provide a clear, step-by-step explanation for your decision. If the ground-truth is flawed, specify which argument is incorrect and why it contradicts the prompt or schema. If it is not flawed, briefly explain why the ground-truth is a correct interpretation of the user's request.",
  "reasoning_summary": "A shorter rationale for your decision. If the ground-truth is not flawed, just mention that it is not flawed. If the ground-truth is flawed, specify the issue concisely. e.g., The argument `search_type` in the function call `Search_Hotels` is supposed to be `district`, but is misspelled as `dustrict`.",
  "error_category": "The category that corresponds to the issue. e.g., \"Flawed function response\". If the sample is not flawed, use \"Not Flawed\".",
  "is_flawed": <true_or_false>
}}
```


## Target Sample

### Sub-benchmark Description

{subbench_description}

### System Prompt
```
{system_prompts}
```

### User Prompt
```
{instruction}
```

### Ground Truth Output
```
{reference}
```

### Available Tools (JSON Schemas):
```
{tool_definitions}
```

{tool_implementations}
"""


# ---------------------------------------------------------------------------
#                             SCORING PROMPT
# ---------------------------------------------------------------------------

SCORING_PROMPT = ""


if __name__ == "__main__":
    import argparse
    import json
    import os
    import sys
    from src.utils.types import Benchmark
    from src.bench_loaders import get_bench_loader

    parser = argparse.ArgumentParser(
        description="Generate formatted prompt for a NexusBench sample."
    )
    parser.add_argument(
        "-q",
        "--question_id",
        type=str,
        help="Question ID to format (e.g., VirusTotal_0001_0)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Optional path to save the formatted prompt (or directory when --save-all is used).",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Save prompts for all NexusBench samples to a dedicated folder.",
    )
    args = parser.parse_args()

    nexus_loader_cls = get_bench_loader(Benchmark.NEXUS_BENCH)
    nexus_loader = nexus_loader_cls()
    questions = nexus_loader.load_questions()
    print(f"Loaded {len(questions)} questions")

    def build_prompt(question):
        system_prompts = question.system_prompts or []
        if isinstance(system_prompts, list):
            system_prompt_str = "\n\n".join(system_prompts)
        else:
            system_prompt_str = str(system_prompts)

        tool_definitions = question.tool_definitions
        if not tool_definitions:
            tool_definitions = json.dumps(
                question.available_function_list,
                indent=2,
                ensure_ascii=False,
            )

        return FILTERING_PROMPT.format(
            system_prompts=system_prompt_str,
            instruction=question.instruction,
            reference=question.reference,
            subbench_description=question.subbench_description,
            tool_definitions=tool_definitions,
            tool_implementations=question.tool_implementations
        )

    if args.save_all:
        output_dir = args.output or "nexus_bench_prompts"
        os.makedirs(output_dir, exist_ok=True)

        saved = 0
        for question in questions:
            prompt_text = build_prompt(question)
            filename = f"{question.task_name}-{question.question_id}.txt"
            output_path = os.path.join(output_dir, filename)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(prompt_text)
            saved += 1

        print(f"Saved {saved} prompts to {output_dir}/")
        sys.exit(0)

    if not args.question_id:
        print("Error: --question_id is required unless --save-all is specified.")
        sys.exit(1)

    selected_question = next(
        (question for question in questions if question.question_id == args.question_id),
        None,
    )

    if not selected_question:
        print(f"No NexusBench sample found with ID '{args.question_id}'")
        sys.exit(1)

    print(
        f"NexusBench sample found - Benchmark: {selected_question.benchmark_name}, "
        f"Task: {selected_question.task_name}, ID: {selected_question.question_id}"
    )

    nexus_filtering_prompt = build_prompt(selected_question)

    output_path = (
        args.output
        if args.output
        else f"nexus_bench_{selected_question.question_id}_prompt.txt"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(nexus_filtering_prompt)

    print(f"NexusBench formatted prompt saved to {output_path}")
