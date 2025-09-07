import sys
import os
import json
import re
from typing import Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.types import FormattedQuestion, Benchmark
from src.prompts import (
    tau_bench_prompt, tau2_bench_prompt, ace_bench_prompt,
    nexus_bench_prompt, tool_sandbox_prompt, complex_func_bench_prompt,
    drafter_bench_prompt, bfcl_prompt, multi_challenge_prompt
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


def format_judge_prompt(question: FormattedQuestion, eval_type: str) -> Tuple[str, str]:
    prompt_module = PROMPT_MODULES.get(question.benchmark)
    if not prompt_module:
        raise ValueError(f"Unknown benchmark: {question.benchmark.value}")
    
    prompt_template = _get_prompt_template(prompt_module, eval_type)
    format_data = _extract_format_data(question, question.benchmark)
    format_args = _build_format_args(prompt_template, format_data)
    
    formatted_prompt = prompt_template.format(**format_args)
    return formatted_prompt


def _get_prompt_template(prompt_module, eval_type: str) -> str:
    if eval_type == 'filtration':
        template = getattr(prompt_module, 'FILTERING_PROMPT', None)
    elif eval_type == 'scoring':
        template = getattr(prompt_module, 'SCORING_PROMPT', None)
    else:
        raise ValueError(f"Unknown eval_type: {eval_type}. Must be 'filtration' or 'scoring'")
    
    if template is None:
        raise AttributeError(f"No {eval_type.upper()}_PROMPT found in {prompt_module}")
    
    return template


def _build_format_args(prompt_template: str, format_data: dict) -> dict:
    format_fields = re.findall(r'(?<!\{)\{([^{}]+)\}(?!\})', prompt_template)
    format_args = {}
    
    for field in format_fields:
        if field in format_data:
            format_args[field] = format_data[field]
        else:
            raise ValueError(f"Required field '{field}' not available in question data")
    
    return format_args


def _extract_format_data(question: FormattedQuestion, benchmark: Benchmark) -> dict:
    format_data = {
        'benchmark': benchmark.value,
        'user_prompt': question.instruction,
        'available_function_list': json.dumps(question.available_function_list, indent=2),
        'conversations': json.dumps(question.gt_conv_traj, indent=2),
        'question_id': question.question_id
    }
    
    _add_specific_attributes(question, format_data)
    _add_meta_data(question, format_data)
    _add_all_attributes(question, format_data)
    
    return format_data


def _add_specific_attributes(question: FormattedQuestion, format_data: dict):
    if hasattr(question, 'agent_system_prompt'):
        format_data['agent_system_prompt'] = question.agent_system_prompt


def _add_meta_data(question: FormattedQuestion, format_data: dict):
    if not question.meta:
        return
        
    for key, value in question.meta.items():
        if isinstance(value, (str, int, float, bool)):
            format_data[f'meta_{key}'] = value
        elif isinstance(value, (dict, list)):
            format_data[f'meta_{key}'] = json.dumps(value, indent=2)


def _add_all_attributes(question: FormattedQuestion, format_data: dict):
    for attr_name in dir(question):
        if attr_name.startswith('_') or attr_name in format_data:
            continue
            
        try:
            attr_value = getattr(question, attr_name)
            if isinstance(attr_value, (str, int, float, bool)):
                format_data[attr_name] = attr_value
            elif isinstance(attr_value, (dict, list)):
                format_data[attr_name] = json.dumps(attr_value, indent=2)
        except:
            pass