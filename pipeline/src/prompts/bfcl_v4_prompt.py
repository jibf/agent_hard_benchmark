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
  Normalization converts text to lowercase and strips punctuation marks such as ,./-\_*^() so that superficial differences (e.g., “eiffel-tower” vs “Eiffel Tower”) do not affect scoring.
* **Reference Sources**: A full of partial excerpt of previous conversation where the required information is located in. 
* **Conversation Memory**: A full multi-turn dialogue history between the user and the model, representing the memory state that the model is expected to recall from.

## Instruction

Go through each of the reference sources, think step-by-step, and judge if the ground-truth answer is reasonable given the information in the memory. 

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

MEMORY_SCORING_PROMPT = """
Score the difficulty of this BFCL v4 **Memory** sample. Consider how much synthesis is required across the stored conversation, ambiguity in the user query, and risk of conflicting memory slots.
Return a JSON array of objects with fields: dimension, reasoning, score (1-5).
"""

WEB_SEARCH_FILTERING_PROMPT = """
You are an expert evaluator for **Berkeley Function Calling Leaderboard (BFCL) V4 Agentic, Part 1: Web Search**, which examines an LLM’s ability to use a web search API to answer knowledge-seeking questions that lie beyond its training data.
Your task is to **determine whether the given sample is fundamentally flawed**, meaning that even a perfect agent with unrestricted access to the internet could not reasonably solve the question as designed.

You will be provided with the following information:

* **Question**: The question that was asked to the LLM model. It often requires multi-hop reasoning and evidence retrieval through web search.
* **Ground-Truth Answer**: The canonical answer string(s) that must be contained (after normalization) in a model’s output for it to be marked correct.  
  Normalization converts text to lowercase and strips punctuation marks such as ,./-\_*^() so that superficial differences (e.g., “eiffel-tower” vs “Eiffel Tower”) do not affect scoring.
* **Reference Sources**: Example URLs or text passages that the benchmark used to construct or validate this question.

## Instruction

Go through each of the reference sources, think step-by-step, and judge if the ground-truth answer is reasonable given the information in the sources. 
Since you are highly likely not to have the information needed to judge each source, do not judge if the information in each source is correct. Instead, only judge if a reasonable model would be able to deduce the ground-truth answer, given the sources.

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

WEB_SEARCH_SCORING_PROMPT = """
Score the difficulty of this BFCL v4 **Web Search** sample. Consider hop count, source reliability requirements, temporal freshness, and ambiguity in the linking facts.
Return a JSON array of objects with fields: dimension, reasoning, score (1-5).
"""

FORMAT_SENSITIVITY_FILTERING_PROMPT = """
You are auditing a BFCL v4 **Format Sensitivity** sample. These cases vary the system prompt and formatting instructions to test whether models obey strict output policies.

Metadata:
- Task ID: {question_id}
- Underlying Category: {base_category}
- Format Profile: {format_profile}

### System Prompt / Format Requirements
{system_prompt}

### User Instruction
{instruction}

### Expected Behaviour (Ground Truth)
{ground_truth}

Return a JSON object with fields:
  * "is_flawed": boolean
  * "reason": concise explanation referencing the format requirements
  * "error_category": one of ["environment", "ground_truth", "instruction", "other"] when flawed, else null
"""

FORMAT_SENSITIVITY_SCORING_PROMPT = """
Score the difficulty of this BFCL v4 **Format Sensitivity** sample. Consider strictness of the formatting policy, conflicting instructions, and ambiguity between the format and actual task.
Return a JSON array of objects with fields: dimension, reasoning, score (1-5).
"""


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

_FORMAT_SENSITIVITY_PROMPTS: Dict[LLMJudgeStep, str] = {
    LLMJudgeStep.UNIVERSAL_FILTER: FORMAT_SENSITIVITY_FILTERING_PROMPT,
    LLMJudgeStep.SPECIFIC_FILTER: FORMAT_SENSITIVITY_FILTERING_PROMPT,
    LLMJudgeStep.SCORE: FORMAT_SENSITIVITY_SCORING_PROMPT,
}

_DOMAIN_PROMPT_OVERRIDES: Dict[str, Dict[LLMJudgeStep, str]] = {
    "memory": _MEMORY_PROMPTS,
    "web_search": _WEB_SEARCH_PROMPTS,
    "format_sensitivity": _FORMAT_SENSITIVITY_PROMPTS,
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
        return _format_memory_context(getattr(question, "memory_context", None))
    if field == "memory_source":
        return _format_memory_source(getattr(question, "sources", None))
    if field == "research_trail":
        return _format_research_trail(getattr(question, "sources", None))
    if field == "ground_truth":
        return _to_pretty_json(_get_value(question, "ground_truth"))
    if field == "gt_conv_traj":
        return _to_pretty_json(_get_value(question, "gt_conv_traj"))
    if field == "num_hops":
        return _as_text(_get_value(question, "num_hops"), "N/A")
    if field == "format_profile":
        return _as_text(_get_format_metadata(question, "format_profile"), "N/A")
    if field == "base_category":
        return _as_text(_get_format_metadata(question, "base_category"), "N/A")

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
        format_meta = meta.get("format_sensitivity", meta)
        if isinstance(format_meta, dict):
            return format_meta.get(key)
    return None


def _format_memory_context(context: Any) -> str:
    if not context:
        return "N/A"
    return _to_pretty_json(context)


def _format_memory_source(source: Any) -> str:
    if source is None:
        return "N/A"
    if isinstance(source, (dict, list)):
        return _to_pretty_json(source)
    return str(source)


def _format_research_trail(sources: Any) -> str:
    if not sources:
        return "N/A"
    return _to_pretty_json(sources)


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
