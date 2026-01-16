import argparse
import json
import os
import re
from typing import Any, Dict, List

from src.utils.types import BFCLv4Question, LLMJudgeStep


DEFAULT_FILTERING_PROMPT = """"""

DEFAULT_SCORING_PROMPT = """"""

MEMORY_FILTERING_PROMPT = """
You are an expert evaluator for **Berkeley Function Calling Leaderboard (BFCL) V4 Agentic, Part 2: Memory**, which examines an LLM agent’s ability to correctly recall, reference, and integrate information from prior conversation context (memory) to answer follow-up questions that depend on earlier dialogue.
Your task is to **determine whether the given sample is fundamentally flawed**, meaning that even a perfect agent with accurate and complete memory access could not reasonably derive the correct answer from the provided conversation context (e.g., the ground-truth contradicts the memory, the memory lacks necessary information, or the question is logically incoherent).
You will be provided with the following information:

* **Question**: The question that was asked to the LLM model. It often requires accessing previous conversation memory to answer. 
* **Ground-Truth Answer**: The canonical answer string(s) that must be contained (after normalization) in a model’s output for it to be marked correct. If multiple ground-truth answers are provided, matching any one of them is sufficient for the sample to be considered correct.
  Normalization converts text to lowercase and strips punctuation marks such as ,./-\\_*^() so that superficial differences (e.g., “eiffel-tower” vs “Eiffel Tower”) do not affect scoring.
* **Reference Sources**: A full or partial excerpt of previous conversation where the required information is located. 
* **Conversation Memory**: A full multi-turn dialogue history between the user and the model, representing the memory state that the model is expected to recall from.

## Instruction

Go through each of the reference sources and the full conversation memory, think step-by-step, and judge if the ground-truth answer is reasonable given the information in the memory. 

## Sample to Evaluate

### Question
{instruction}

### Ground-Truth Answer
{ground_truth}

### Reference Sources
{sources}

### Conversation Memory
{memory_context}

## Evaluation and Output Format

Your final output must be a JSON object with the following structure, with no additional commentary:

```json
{{
  "reasoning": "Provide a clear, step-by-step explanation for your decision. If the sample is flawed, specify what is incorrect and why it contradicts the user's prompt, system policies, or the user's role. If it is not flawed, briefly explain why the sample is valid.",
  "reasoning_summary": "A shorter rationale for your decision. If the sample is not flawed, just mention that it is not flawed. If it is flawed, specify the issue concisely. e.g., The ground truth books a connecting flight, but the user requested a direct flight.",
  "error_category": "This is just a placeholder to match the required format. Just print \"Flawed\" or \"Not Flawed\" without the quote marks.",
  "is_flawed": <true or false>
}}
"""

MEMORY_SCORING_PROMPT = """"""

WEB_SEARCH_FILTERING_PROMPT = """
You are an expert evaluator for **Berkeley Function Calling Leaderboard (BFCL) V4 Agentic, Part 1: Web Search**, which examines an LLM's ability to use a web search API to answer knowledge-seeking questions that lie beyond its training data. This benchmark evaluates agents' ability to perform real-time web searches at the time questions are asked, meaning questions may reference time-sensitive information, future dates, or events that occur after the benchmark's creation date.
Your task is to **determine whether the given sample is fundamentally flawed**, meaning that even a perfect agent with unrestricted access to the internet could not reasonably solve the question as designed.

You will be provided with the following information:

* **Question**: The question that was asked to the LLM model. It often requires multi-hop reasoning and evidence retrieval through web search.
* **Ground-Truth Answer**: The canonical answer string(s) that must be contained (after normalization) in a model's output for it to be marked correct.  
  Normalization converts text to lowercase and strips punctuation marks such as ,./-\\_*^() so that superficial differences (e.g., "eiffel-tower" vs "Eiffel Tower") do not affect scoring.
* **Reference Sources**: Example URLs or text passages that the benchmark used to construct or validate this question. These represent the type of information a real-time web search might find.

## Instruction

Go through each of the reference sources, think step-by-step, and judge if the ground-truth answer is reasonable given the information in the sources. 
Since you are highly likely not to have the information needed to judge each source, do not judge if the information in each source is correct. Instead, only judge if a reasonable model would be able to deduce the ground-truth answer, given the sources.

When evaluating, consider that agents perform searches at the time questions are asked, so questions referencing future dates or time-sensitive information are answerable through real-time search. 

## Sample to Evaluate

### Question
{instruction}

### Ground-Truth Answer
{ground_truth}

### Reference Sources
{sources}

## Evaluation and Output Format

Your final output must be a JSON object with the following structure, with no additional commentary:

```json
{{
  "reasoning": "Provide a clear, step-by-step explanation for your decision. If the sample is flawed, specify what is incorrect and why it contradicts the user's prompt, system policies, or the user's role. If it is not flawed, briefly explain why the sample is valid.",
  "reasoning_summary": "A shorter rationale for your decision. If the sample is not flawed, just mention that it is not flawed. If it is flawed, specify the issue concisely. e.g., The ground truth books a connecting flight, but the user requested a direct flight.",
  "error_category": "This is just a placeholder to match the required format. Just print \"Flawed\" or \"Not Flawed\" without the quote marks.",
  "is_flawed": <true or false>
}}
"""

WEB_SEARCH_SCORING_PROMPT = """"""

# NOTE: Format sensitivity is not currently used in BFCL V4 (commented out in loader)
# The prompt below is kept for potential future use but is not active
# FORMAT_SENSITIVITY_FILTERING_PROMPT = """
# You are an expert evaluator for BFCL, a benchmark designed to assess an agent's multi-turn and multi-step function calling abilities.
# Your task is to determine if a given benchmark sample has a fundamental flaw in its user prompt, environment, or ground-truths, which would make it unable to be incorporated in the evaluation.
# 
# 
# You will be provided with the following information:
# * **Instruction**: The description of the task given to the agent. 
# * **Agent System Prompt**: the system prompt used to initialize the agent model. This may contain a specific instruction on the answer style, domain-specific policy that the agent needs to follow, a list of available functions and their schema (in JSON format), etc.
# * **Available Functions**: a list of functions available for the agents and their schema. Note that functions related to the file system (e.g., `wc`, `ls`, `sort`, etc.), if provied, abide by the standard Unix semantics unless specified otherwise directly.
# * **Missed Functions**: This is only provided in the category `multi_turn_miss_func`. This is the function that is not provided to the agent at the first turn, but will be provided after a specified number of agent responses.  
# * **Initial Configuration**: The initial environment setup and conditions before the task begins. 
# * **Ground-Truth Milestone Function Call Trajectory**: the provided ground-truth trajectory of crucial function calls. When this is empty or None, it means that the agent needs to call nothing to be scored as correct. Note that entries with `"role": "tool"` are the results of the directly preceding agent tool calls.
# A sample is **flawed** if it exhibits one or more of the issues described below.
# 
# 
# ## Flaw Categories
# 
# Below is the categorization of benchmark issues, outlined according to its **relevant benchmark component**. A sample is considered flawed if it has one or more of the issues below.
# 
# ### Environment
# 
# This category covers flaws within the agent's operating environment—the tools and API results—which can make a task unsolvable regardless of the agent's logic.
# 
# * Insufficient toolsets: The environment does not provide the necessary tools for the agent to complete the task. 
#   * Look for:  
#     * Empty Function List: No functions are provided but the test expects function calls.
#     * Missing Core Functionality: Essential functions for completing the task are absent from the functions list.
# 
# * Flawed tool design: Tools exist, but their interface or description makes them unusable or misleading.
#   * Look for:  
#     * Incompatible Parameters: Functions exist but their parameters don't match requirements.
#     * Environment–Function Mismatch: Available functions don't match the described environment.
# 
# ### Ground-Truth
# 
# This category addresses errors in the provided ground-truth trajectory, where the supposed correct solution is itself incorrect, forcing any correct agent to fail the evaluation.
# 
# * Malformed function calls: A technical error where a ground-truth function call violates the provided API schema.
#   * Example: A parameter requires a string but is given a number (e.g., dest_id: 123 instead of dest_id: "123"), a required parameter is missing, the function name is wrong, or a parameter value v
#   * Note that If a function has only one parameter, it may be invoked without using a keyword argument. This is not a flaw. e.g., `sort('final_report.pdf')`
# 
# * Incorrect function calls: A function call is syntactically valid but logically flawed. The function choice or a parameter value contradicts the user's request or the context from previous steps.
#   * Unjustified/Hallucinated Parameters: A value (e.g., a file name, user name) that appears without any grounding context. For example, searching for a hotel on a date that was not returned by a preceding flight search.
#   * Contradictory: A value that directly contradicts a constraint in the user's prompt. However, it is NOT a flaw if there is any chance that the agent's action was a necessary alternative due to constraints like an insufficient budget or a lack of available seats.
#   * Policy Violation: A function call in the ground truth trajectory directly violates the provided system policy. Example: The ground truth where the agent calls a specific function twice, although it is mentioned in the system policy that the function can only be called once.
#   * Misspelled or Incorrectly Identified Parameter Values: A misspelled name or an ID/slug that points to the wrong entity (e.g., selecting the wrong airport ID).
# 
# * Redundant/ungrounded function calls: The ground truth function call trajectory consists of function calls that are redundant in solving the task, ungrounded by the context, or irrelevant in solving the task.
#   * Irrelevant tool call: A function call in the ground truth trajectory is totally irrelevant to the task or belongs to a completely different domain. Example: agent calls a function to reserve a flight, though it was asked to process product exchange.
#   * Redundant tool call: A function call that is not necessary in solving the task. Example: the agent is asked to search for attractions until it finds one that meets a certain condition; However, the agent performs the search in an arbitrary order, resulting in an excessive number of function calls.
# 
# ## Crucial Rules
# 
# ### Actively Reconstruct the Conversation
# 
# The ground-truth trajectory only contains crucial function calls from the agent's response. It intentionally omits agents responses in natural language (e.g., confirmations, request, clarifications, or follow-up questions), or less important and obvious function calls, such as `get_user_id`.
# Your task is to find undeniable flaws. Therefore, you MUST operate under the following assumption:
# 
# * For example, the user may provide an additional information or permits to use a new function after an agent prints an empty response with no tool call. This is not a flaw, since the agent would have requested for the information or the function, though it is not revealed in the provided ground truth.
# * If a sequence of function calls can be justified by a plausible, un-shown conversation, then it is NOT a flaw.
# 
# ### Do NOT Judge Tool Results
# 
# The tool results in the ground-truth trajectory are automatically generated via actually calling the corresponding tools, and are not subject to judgement. Flaws in tool results should NOT be the reason you mark a sample as flawed.
#  
# ## Evaluation and Output Format
# Carefully analyze the provided sample. Think step-by-step to determine if the ground-truth trajectory is a correct and logical solution to the user's prompt.
# 
# Your final output must be a JSON object with the following structure, with no additional commentary:
# 
# ```json
# {{
#   "reasoning": "Provide a clear, step-by-step explanation for your decision. If the sample is flawed, specify what is incorrect and why it contradicts the user's prompt, system policies, or the user's role. If it is not flawed, briefly explain why the sample is valid.",
#   "reasoning_summary": "A shorter rationale for your decision. If the sample is not flawed, just mention that it is not flawed. If it is flawed, specify the issue concisely. e.g., The ground truth books a connecting flight, but the user requested a direct flight.",
#   "error_category": "The category that corresponds to the issue. e.g., \"Incorrect function calls\". If the sample is not flawed, use \"Not Flawed\".",
#   "is_flawed": <true or false>
# }}
# ```
# 
# ## Sample to be evaluated
# 
# ### Category 
# * category: {category}
# * subcategory: {subcategory}
# 
# ### Instruction
# 
# {instruction}
# 
# ### Agent System Prompt
# 
# {system_prompt}
# 
# {missed_function}
# 
# ### Initial Config
# 
# {initial_pwd_description}
# ```json
# {initial_config}
# ```
# 
# ### Tool Default States
# 
# {default_states}
# 
# ### Ground-Truth Function Call(s):
# ```json
# {gt_conv_traj}
# ```
# """
FORMAT_SENSITIVITY_FILTERING_PROMPT = """"""
FORMAT_SENSITIVITY_SCORING_PROMPT = """"""

_DEFAULT_PROMPTS: Dict[LLMJudgeStep, str] = {
    LLMJudgeStep.UNIVERSAL_FILTER: DEFAULT_FILTERING_PROMPT,
    LLMJudgeStep.SPECIFIC_FILTER: DEFAULT_FILTERING_PROMPT,
    LLMJudgeStep.SCORE: DEFAULT_SCORING_PROMPT,
}

_MEMORY_PROMPTS: Dict[LLMJudgeStep, str] = {
    LLMJudgeStep.UNIVERSAL_FILTER: MEMORY_FILTERING_PROMPT,
    LLMJudgeStep.SPECIFIC_FILTER: MEMORY_FILTERING_PROMPT,
    LLMJudgeStep.SCORE: MEMORY_SCORING_PROMPT,
}

_WEB_SEARCH_PROMPTS: Dict[LLMJudgeStep, str] = {
    LLMJudgeStep.UNIVERSAL_FILTER: WEB_SEARCH_FILTERING_PROMPT,
    LLMJudgeStep.SPECIFIC_FILTER: WEB_SEARCH_FILTERING_PROMPT,
    LLMJudgeStep.SCORE: WEB_SEARCH_SCORING_PROMPT,
}

# NOTE: Format sensitivity is not currently used in BFCL V4
# _FORMAT_SENSITIVITY_PROMPTS: Dict[LLMJudgeStep, str] = {
#     LLMJudgeStep.UNIVERSAL_FILTER: FORMAT_SENSITIVITY_FILTERING_PROMPT,
#     LLMJudgeStep.SPECIFIC_FILTER: FORMAT_SENSITIVITY_FILTERING_PROMPT,
#     LLMJudgeStep.SCORE: FORMAT_SENSITIVITY_SCORING_PROMPT,
# }

_DOMAIN_PROMPT_OVERRIDES = {
    "memory": _MEMORY_PROMPTS,
    "web_search": _WEB_SEARCH_PROMPTS,
    # "format_sensitivity": _FORMAT_SENSITIVITY_PROMPTS,  # Not currently used in BFCL V4
}


def build_prompt(question: BFCLv4Question, step: LLMJudgeStep) -> str:
    prompt_map = _DOMAIN_PROMPT_OVERRIDES.get(_infer_domain(question), _DEFAULT_PROMPTS)
    template = prompt_map[step]
    fields = _extract_format_fields(template)
    args = {_field: _render_field(question, _field) for _field in fields}
    return template.format(**args)


def _infer_domain(question: BFCLv4Question) -> str:
    task_name = (question.task_name or "").lower()
    if task_name:
        return task_name
    question_id = (question.question_id or "").lower()
    return question_id.split("_")[0] if question_id else ""


def _render_field(question: BFCLv4Question, field: str) -> str:
    if field == "memory_context":
        return _to_pretty_json(getattr(question, "memory_context", None))
    if field == "memory_source":
        return _to_pretty_json(getattr(question, "sources", None))
    if field == "research_trail":
        return _to_pretty_json(getattr(question, "sources", None))
    if field == "ground_truth":
        return _to_pretty_json(_get_value(question, "ground_truth"))
    if field == "gt_conv_traj":
        return _to_pretty_json(_get_value(question, "gt_conv_traj"))
    if field == "num_hops":
        return _as_text(_get_value(question, "num_hops"), "N/A")
    if field == "format_profile":
        return _as_text(_get_format_metadata(question, "config"), "N/A")
    if field == "base_category":
        return _as_text(_get_format_metadata(question, "base_category"), "N/A")
    if field == "question_turns":
        return _to_pretty_json(getattr(question, "question_turns", None))
    if field == "available_function_list":
        return _to_pretty_json(getattr(question, "available_function_list", []))

    value = _get_value(question, field)
    if isinstance(value, (dict, list)):
        return _to_pretty_json(value)
    return _as_text(value, "N/A")


def _get_value(question: BFCLv4Question, field: str):
    if hasattr(question, field):
        return getattr(question, field)
    meta = getattr(question, "meta", None)
    if isinstance(meta, dict) and field in meta:
        return meta[field]
    return None


def _get_format_metadata(question: BFCLv4Question, key: str):
    meta = getattr(question, "meta", None)
    if isinstance(meta, dict):
        fs_meta = meta.get("format_sensitivity", {})
        if isinstance(fs_meta, dict):
            return fs_meta.get(key)
    return None


def _extract_format_fields(template: str) -> List[str]:
    return re.findall(r"(?<!\{)\{([^{}]+)\}(?!\})", template)


def _to_pretty_json(value: Any) -> str:
    if value is None:
        return "null"
    try:
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _as_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return _to_pretty_json(value)


def _sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[\\/:*?\"<>|]", "_", value)
    return sanitized.replace(" ", "_")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render BFCL v4 judge prompts")
    parser.add_argument(
        "-q",
        "--question-id",
        help="Specific BFCL v4 question ID to render",
    )
    parser.add_argument(
        "-s",
        "--step",
        choices=[step.value for step in LLMJudgeStep],
        default=LLMJudgeStep.SPECIFIC_FILTER.value,
        help="LLM judge step to render (default: specific_filter)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Destination file for the rendered prompt (single-question mode)",
    )
    parser.add_argument(
        "--output-dir",
        default="bfcl_v4_prompts",
        help="Directory to store prompts when exporting the full set",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Export prompts for every BFCL v4 question",
    )
    return parser.parse_args()


def _load_questions() -> List[BFCLv4Question]:
    from src.bench_loaders.bfcl_v4_loader import BfclV4Loader

    loader = BfclV4Loader()
    return loader.load_questions()


def _render_prompt(question: BFCLv4Question, step: LLMJudgeStep) -> str:
    return build_prompt(question, step)


def _write_text(path: str, content: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _export_single_prompt(
    question: BFCLv4Question,
    step: LLMJudgeStep,
    output_path: str,
) -> None:
    prompt_text = _render_prompt(question, step)
    _write_text(output_path, prompt_text)
    print(f"Saved {step.value} prompt for {question.question_id} to {output_path}")


def _export_all_prompts(
    questions: List[BFCLv4Question],
    step: LLMJudgeStep,
    base_dir: str,
) -> None:
    total_written = 0
    for question in questions:
        domain = (question.task_name or "unknown_domain").lower()
        filename = f"{_sanitize_filename(question.question_id)}__{step.value}.txt"
        output_path = os.path.join(base_dir, step.value, domain, filename)
        prompt_text = _render_prompt(question, step)
        _write_text(output_path, prompt_text)
        total_written += 1

    print(
        f"Saved {total_written} prompts for step '{step.value}' under {os.path.join(base_dir, step.value)}"
    )


def _select_question(
    questions: List[BFCLv4Question],
    question_id: str,
) -> BFCLv4Question:
    for question in questions:
        if question.question_id == question_id:
            return question
    raise ValueError(f"Could not find BFCL v4 question with ID '{question_id}'")


def _default_output_filename(step: LLMJudgeStep) -> str:
    return f"bfcl_v4_{step.value}_prompt.txt"


def _main() -> None:
    args = _parse_args()
    step = LLMJudgeStep(args.step)
    questions = _load_questions()

    if args.save_all:
        _export_all_prompts(questions, step, args.output_dir)
        return

    if not args.question_id:
        raise SystemExit("Error: --question-id is required when not using --save-all")

    try:
        selected = _select_question(questions, args.question_id)
    except ValueError as exc:
        raise SystemExit(str(exc))

    output_path = args.output or _default_output_filename(step)
    _export_single_prompt(selected, step, output_path)


if __name__ == "__main__":
    _main()
