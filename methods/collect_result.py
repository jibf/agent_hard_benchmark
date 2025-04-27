import sys
import os
sys.path.append(os.path.abspath(".."))

from Drafter_Bench.utils.types import Score_builder


def collect_result(
    task_set: str,
    index: int,
    instruction: str,
    ground_code: str,
    generated_code: str,
    code_quality: Score_builder,
):
    return {
        "task_set": task_set,
        "task_id": index,
        "instruction": instruction,
        "ground_truth": ground_code,
        "output_code": generated_code,
        "task_score": {
            "task_score": code_quality.total_score,
            "success_arguments_define": code_quality.arguments_count,
            "total_arguments_define": code_quality.total_arguments,
            "success_variable_transfer": code_quality.variables_transfer_count,
            "total_variable_transfer": code_quality.total_variables_transfer,
            "success_function_calling": code_quality.function_calls_count,
            "total_function_calling": code_quality.total_function_calls,
            "success_single_tool_selection": code_quality.single_tool_calls_count,
            "total_single_tool_selection": code_quality.total_single_tool_calls,
            "success_multi_tool_selection": code_quality.multi_tool_calls_count,
            "total_multi_tool_selection": code_quality.total_multi_tool_calls,
            "intersected_plan_execution": code_quality.intersected_plan_count,
            "total_plans_appeared": code_quality.all_plan,
            "ground_plan_execution": code_quality.total_plan_execution,
        },
    }
