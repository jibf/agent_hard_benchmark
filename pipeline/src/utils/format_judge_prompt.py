import sys
import os
import json
import re

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.types import FormattedQuestion, Benchmark, LLMJudgeStep
from src.prompts import (
    tau_bench_prompt, tau2_bench_prompt, ace_bench_prompt,
    nexus_bench_prompt, tool_sandbox_prompt, complex_func_bench_prompt,
    drafter_bench_prompt, bfcl_prompt, multi_challenge_prompt, universal_prompt
)


PROMPT_MODULES = {
    Benchmark.TAU_BENCH: tau_bench_prompt,
    Benchmark.TAU2_BENCH: tau2_bench_prompt,
    Benchmark.ACE_BENCH: ace_bench_prompt,
    Benchmark.NEXUS_BENCH: nexus_bench_prompt,
    Benchmark.TOOL_SANDBOX: tool_sandbox_prompt,
    Benchmark.COMPLEX_FUNC_BENCH: complex_func_bench_prompt,
    Benchmark.DRAFTER_BENCH: drafter_bench_prompt,
    Benchmark.BFCL: bfcl_prompt,
    Benchmark.MULTI_CHALLENGE: multi_challenge_prompt
}


def format_judge_prompt(question: FormattedQuestion, step: LLMJudgeStep) -> str:
    if step == LLMJudgeStep.UNIVERSAL_FILTER:
        # Use PromptBuilder for universal filtering
        return universal_prompt.build_filtering_prompt(question)
    else:
        prompt_module = PROMPT_MODULES.get(question.benchmark)
        if not prompt_module:
            raise ValueError(f"Unknown benchmark: {question.benchmark.value}")

        prompt_template = _get_prompt_template(prompt_module, step)
        required_fields = _extract_format_fields(prompt_template)
        format_args = _build_format_args(question, required_fields)

        formatted_prompt = prompt_template.format(**format_args)
        return formatted_prompt


def _get_prompt_template(prompt_module, step: LLMJudgeStep) -> str:
    try:
        if step == LLMJudgeStep.SPECIFIC_FILTER or step == LLMJudgeStep.UNIVERSAL_FILTER:
            template = prompt_module.FILTERING_PROMPT
        elif step == LLMJudgeStep.SCORE:
            template = prompt_module.SCORING_PROMPT
    except:
        raise AttributeError(f"No {step.value.upper()}_PROMPT found in {prompt_module}")
    return template


def _extract_format_fields(prompt_template: str) -> list[str]:
    """Extract all format field names from the prompt template."""
    return re.findall(r'(?<!\{)\{([^{}]+)\}(?!\})', prompt_template)


def _build_format_args(question: FormattedQuestion, required_fields: list[str]) -> dict:
    """Build format arguments by extracting required fields from question."""
    format_args = {}
    for field in required_fields:
        if hasattr(question, field):
            value = getattr(question, field)
            format_args[field] = _serialize_value(value)
        else:
            raise ValueError(f"Required field '{field}' not found in question data")
    
    return format_args


def _serialize_value(value):
    """Serialize value for template formatting."""
    if isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, (dict, list)):
        return json.dumps(value, indent=2)
    elif isinstance(value, Benchmark):
        return value.value
    else:
        return str(value)


