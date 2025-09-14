import json
import typing
from src.utils.types import Benchmark, FormattedQuestion

BENCHMARK_PROPERTIES = {
    Benchmark.TAU_BENCH: {
        'provides_gt': True,
        'milestone_only_gt': True,
        'has_db': True,
        'is_multi_turn': True,
        'provides_function_call_results': True,
    },
    Benchmark.TAU2_BENCH: {
        'provides_gt': True,
        'milestone_only_gt': True,
        'has_db': True,
        'is_multi_turn': True,
        'provides_function_call_results': True,
    },
    Benchmark.COMPLEX_FUNC_BENCH: {
        'provides_gt': True,
        'milestone_only_gt': False,
        'has_db': False,
        'is_multi_turn': True,
        'provides_function_call_results': True,
    },
    Benchmark.ACE_BENCH: {
        'provides_gt': True,
        'milestone_only_gt': False,
        'has_db': False,
        'is_multi_turn': True,
        'provides_function_call_results': False,
    },
    Benchmark.NEXUS_BENCH: {
        'provides_gt': True,
        'milestone_only_gt': True,
        'has_db': False,
        'is_multi_turn': True,
        'provides_function_call_results': False,
    },  # many fields are duplicate in NexusBenchQuestion; only three fields: instruction, gt_conv_traj, and available_function_list are actually used 
    Benchmark.DRAFTER_BENCH: {
        'provides_gt': True,
        'milestone_only_gt': False,
        'has_db': False,
        'is_multi_turn': False,
        'provides_function_call_results': False
    }   # avilable_function_list is empty, and the list of available functions is in the agent system prompt
}


class UniversalPromptBuilder:
    def __init__(self):
        self.sections = []

    def add_header(self, benchmark: Benchmark):
        header = f"""You are an expert evaluator designed to assess the quality and reliability of samples in {benchmark.value}, a benchmark that assesses the agentic task solving capabilities of LLMs.
Your task is to identify if the sample has a fundamental flaw in its ground-truth, which would make it an unreliable sample for evaluation."""
        self.sections.append(header)
        return self

    def add_information_overview(self, question: FormattedQuestion):
        overview_parts = ["You will be provided with the following information about the sample:"]
        overview_parts.append("- Task Description/Instructions: The prompt or scenario given to the model")

        # Common fields
        if hasattr(question, 'available_function_list') and question.available_function_list:
            overview_parts.append("- Available Functions: A list of functions/tools available to the agent and their schemas")
        if hasattr(question, 'agent_system_prompt') and question.agent_system_prompt:
            overview_parts.append("- System Prompt Given to the Agent: This may be a specific instruction on the answer style, or a domain-specific policy that the agent must follow. ")
        if hasattr(question, 'user_context') and question.user_context:
            overview_parts.append("- User Context: Information about the user and relevant background details, such as related database contents")
        
        # Benchmark-specific fields
        if hasattr(question, 'user_system_prompt') and question.user_system_prompt:
            overview_parts.append("- System Prompt Given to the User: This is a system prompt provided to the model simulating the user. It may contain instructions, context, or previous conversation trajectory that the user-simulating model can use as a context.")

        if hasattr(question, 'initial_config') and question.initial_config:                             # For ACEBench
            overview_parts.append("- Initial Configuration: The initial setup or configuration provided to the agent at the start of the task")

        if hasattr(question, 'available_user_function_list') and question.available_user_function_list: # For Tau2Bench
            overview_parts.append("- Available Functions for the User Model: In this benchmark, user-simulating models can also make function calls by itself. This provides a list of functions/tools available to the user-simulating model and their schemas")

        if hasattr(question, 'initial_state') and question.initial_state:                               # For Tau2Bench
            overview_parts.append("- Initial State: The initial environment setup and conditions before the task begins. This defines the starting state of the system.")

        if hasattr(question, 'evaluation_criteria') and question.evaluation_criteria:                   # For Tau2Bench
            overview_parts.append("- Complete Evaluation Criteria: The complete evaluation criteria including milestone ground-truth actions (function calls), natural language assertions, and final environment state assertion to validate. The sample is considered to be flawed if one or more of these criteria is unachievable.")

        # Ground-Truth Trajectory needs to be at the end
        if hasattr(question, 'gt_conv_traj'):
            milestone_only_gt = BENCHMARK_PROPERTIES[question.benchmark]['milestone_only_gt']
            call_results_included = BENCHMARK_PROPERTIES[question.benchmark]['provides_function_call_results']
            overview_parts.append(f"- Ground-Truth Trajectory: The provided reference function call sequence that the agent is expected to follow.")
            overview_parts.append(f"\t- This is what you will evaluate.")
            if call_results_included:
                overview_parts.append("\t- Note that messages with \"role\": \"observation\", are the results of the function call right before")
            if milestone_only_gt:
                overview_parts.append("\t- Important Note: The ground-truth trajectory contains only milestone function calls, which are the function calls that are most crucial in solving the task. Thus, some of the less important function calls may have been omitted although they are necessary to solve the task.")

        self.sections.append("\n".join(overview_parts))
        return self

    def add_instruction_and_error_categories(self, question: FormattedQuestion):
        is_gt_milestone_only = BENCHMARK_PROPERTIES.get(question.benchmark, {}).get('milestone_only_gt', False)
        has_separate_evaluation_criteria = hasattr(question, 'evaluation_criteria') and question.evaluation_criteria    # True if the benchmark has separate evaluation criteria other than function call


        INSTRUCTION = f"""## Instruction for Evaluation\n
Follow this hierarchical process to identify flaws in ground truth. You must check for errors in the order presented below. If a sample fails any check, it is considered flawed, and you should stop the evaluation and report the specific error category.\n
**A. Malformed Function Calls**
First, check if the function calls are syntactically correct and adhere to the API schema. The sample is flawed if any function call is malformed.\n
- Schema Violations: Function calls that violate the provided API schema (wrong parameter types, missing required parameters, invalid function names)
- Format Errors: Parameters with incorrect data types or does not conform to expected formats (e.g., invalid date format)
- Invalid Parameter Values: String-typed parameter values that are clearly misspelled or do not belong to the expected set of values.\n
If all function calls are syntactically correct, proceed to the next check.\n
**B. Incorrect Function Calls**
Next, evaluate if the function calls are logically correct and grounded in the user's request and available context. The sample is flawed if any function call contains ungrounded or contradictory information.\n
- Unjustified/Hallucinated Parameters: Parameter values that lack supporting context from the task description, previous actions, or provided information.
- Contradictory to User Prompt/Policy: Function calls or parameter values that directly contradict explicit constraints or requirements stated by the user.\n
If all function calls are logically sound, proceed to the next check.\n
**C. Function Call Selection Ambiguity**
Third, determine if the function call sequence is a single, unambiguous correct choice. The sample is flawed if there are multiple equally valid ways to complete the task, but the ground-truth trajectory only accepts one specific path.\n
- Ambiguous Parameter/Tool Selection: Cases where multiple valid approaches exist, but only one specific approach is accepted as correct by the ground truth.\n
If the trajectory is not subject to ambiguity, proceed to the final check.\n"""

        if not is_gt_milestone_only:        # Add section D only for benchmarks that provide the full trajectory
            INSTRUCTION += """**D. Redundant/Incomplete Function Call Trajectory**
Finally, check if the trajectory is efficient and complete. The sample is flawed if the sequence is either too long or missing crucial steps.\n
- Redundant Function Calls: The trajectory includes unnecessary function calls that do not contribute to solving the task.
- Incomplete Function Calls: The trajectory is missing crucial function calls needed to properly solve the task.\n"""

        if has_separate_evaluation_criteria and question.evaluation_criteria: 
            INSTRUCTION += "**E. Unachievable Evaluation Criteria**\n"
            INSTRUCTION += "For this sample, you are also provided with a complete set of evaluation criteria that includes natural language assertions and/or final environment state assertions in addition to the milestone function calls. Carefully check if these criteria is actually achievable by the agent.\n"
            INSTRUCTION += "If there are specific values of amounts mentioned in the criteria, you need to verify how those values are deduced. If any of these values cannot be logically inferred from the plausible conversation trajectory consistent with the ground-truth, then it is a flaw.\n"


        if is_gt_milestone_only:            # Add important notes for milestone-only benchmarks
            INSTRUCTION += "### CRUCIAL RULE: Make plausible assumptions\n"
            INSTRUCTION += "As mentioned previously, the ground-truth trajectory contains only key milestone function calls and intentionally omits natural language conversation between user and agent (e.g., user confirmations, clarifications, or follow-up questions)."
            INSTRUCTION += "Imagine possible conversation history that would justify the ground truth milestone function call trajectory."
            INSTRUCTION += "If a sequence of function calls can be justified by a sequence of plausible, un-shown conversation and tool calls that does not contradict the user scenario or system prompts, then it is NOT a flaw. "
            INSTRUCTION += "Flag a sample as flawed ONLY if a function call is impossible to justify, even with a hypothetical conversation.)\n"

        self.sections.append(INSTRUCTION)
        return self

    def add_output_format(self):
        output_format = """## Evaluation and Output Format\n
Carefully analyze the provided sample step-by-step, following the hierarchical evaluation process outlined above.\n
Your final output must be a JSON object with the following structure, with NO additional commentary:\n
```json
{
  "reasoning": "Provide a clear, step-by-step explanation for your decision. If the sample is flawed, specify what is incorrect and why it violates the criteria above. If it is not flawed, briefly explain why the sample is valid and reliable for evaluation.",
  "reasoning_summary": "A concise summary of your decision. If not flawed, state 'Sample is valid for evaluation.' If flawed, provide a brief description of the main issue.",
  "error_category": "The error category identifier (e.g., 'B'). If no errors are found, state 'None'.",
  "is_flawed": <true or false>
}
```"""
        self.sections.append(output_format)
        return self

    def add_sample_data(self, question):
        sample_section = ["## Sample to be evaluated"]

        # Task Description
        sample_section.append("### Task Description/Instructions")
        sample_section.append(getattr(question, 'instruction', ''))

        # Available Functions
        available_functions = getattr(question, 'available_function_list', None)
        if available_functions:
            sample_section.append("\n### Available Functions and Schemas")
            sample_section.append("```json")
            sample_section.append(json.dumps(available_functions, indent=2))
            sample_section.append("```")

        # System Constraints/Policies
        agent_system_prompt = getattr(question, 'agent_system_prompt', None)
        if agent_system_prompt:
            sample_section.append("\n### System Prompt Given to the Agent")
            sample_section.append(agent_system_prompt)

        # User Context
        user_context = getattr(question, 'user_context', None)
        if user_context:
            sample_section.append("\n### User Context and Relevant Information")
            sample_section.append(user_context)

        # User System Prompt (For ACEBench)
        user_system_prompt = getattr(question, 'user_system_prompt', None)
        if user_system_prompt:
            sample_section.append("\n### System Prompt Given to the User")
            sample_section.append(user_system_prompt)

        # Initial Configuration (For ACEBench)
        initial_config = getattr(question, 'initial_config', None)
        if initial_config:
            sample_section.append("\n### Initial Configuration")
            sample_section.append("```json")
            sample_section.append(json.dumps(initial_config, indent=2))
            sample_section.append("```")
        
        # Available Functions for User-Simulating Model (For Tau2Bench)
        available_user_function_list = getattr(question, 'available_user_function_list', None)
        if available_user_function_list:
            sample_section.append("\n### Available Functions for the User-Simulating Model and Schemas")
            sample_section.append("```json")
            sample_section.append(json.dumps(available_user_function_list, indent=2))
            sample_section.append("```")
        
        # Initial State (For Tau2Bench)
        initial_state = getattr(question, 'initial_state', None)
        if initial_state:
            sample_section.append("\n### Initial State")
            sample_section.append("```json")
            sample_section.append(json.dumps(initial_state, indent=2))
            sample_section.append("```")

        # Evaluation Criteria (For Tau2Bench)
        evaluation_criteria = getattr(question, 'evaluation_criteria', None)
        if evaluation_criteria:
            sample_section.append("\n### Complete Evaluation Criteria")
            sample_section.append("```json")
            sample_section.append(json.dumps(evaluation_criteria, indent=2))
            sample_section.append("```")

        # Ground-Truth Trajectory
        sample_section.append("\n### Ground-Truth Trajectory")
        sample_section.append("```json")
        sample_section.append(json.dumps(getattr(question, 'gt_conv_traj', []), indent=2))
        sample_section.append("```")

        self.sections.append("\n".join(sample_section))
        return self

    def build(self):
        return "\n\n".join(self.sections)

def build_filtering_prompt(question) -> str:
    """Build universal filtering prompt using PromptBuilder"""
    builder = UniversalPromptBuilder()
    return (builder
            .add_header(question.benchmark)
            .add_information_overview(question)
            .add_instruction_and_error_categories(question)
            .add_output_format()
            .add_sample_data(question)
            .build())

# For backward compatibility
FILTERING_PROMPT = None  # Will be built dynamically using the builder


# Test
if __name__ == "__main__":
    from src.utils.types import Benchmark
    from src.bench_loaders import get_bench_loader

    # # Test CFB
    # cfb_loader = get_bench_loader(Benchmark.COMPLEX_FUNC_BENCH)()
    # cfb_questions = cfb_loader.load_questions()
    # cfb_sample = cfb_questions[0]

    # # Generate and save CFB prompt
    # cfb_prompt = build_filtering_prompt(cfb_sample)
    # with open("cfb_universal_prompt.txt", "w", encoding="utf-8") as f:
    #     f.write("==== Prompt for CFB ====\n")
    #     f.write(cfb_prompt)

    # # Test Tau Bench
    # tau_loader = get_bench_loader(Benchmark.TAU_BENCH)()
    # tau_questions = tau_loader.load_questions()
    # tau_sample = tau_questions[34]
    # print(tau_sample.question_id)

    # # Generate and save Tau Bench prompt
    # tau_prompt = build_filtering_prompt(tau_sample)
    # with open("tau_bench_universal_prompt.txt", "w", encoding="utf-8") as f:
    #     f.write("==== Prompt for TAU_BENCH ====\n")
    #     f.write(tau_prompt)
    
    # # Test Tau2 Bench
    # tau2_loader = get_bench_loader(Benchmark.TAU2_BENCH)()
    # tau2_questions = tau2_loader.load_questions()
    # tau2_sample = tau2_questions[14]
    # print(tau2_sample.question_id)

    # # Generate and save Tau2 Bench prompt
    # tau2_prompt = build_filtering_prompt(tau2_sample)
    # with open("tau2_bench_universal_prompt.txt", "w", encoding="utf-8") as f:
    #     f.write("==== Prompt for TAU2_BENCH ====\n")
    #     f.write(tau2_prompt)

    # # Test ACE Bench
    # ace_loader = get_bench_loader(Benchmark.ACE_BENCH)()
    # ace_questions = ace_loader.load_questions()
    # ace_sample = ace_questions[0]
    # print(ace_sample.question_id)
    # # Generate and save ACE Bench prompt
    # ace_prompt = build_filtering_prompt(ace_sample)
    # with open("ace_bench_universal_prompt.txt", "w", encoding="utf-8") as f:
    #     f.write("==== Prompt for ACE_BENCH ====\n")
    #     f.write(ace_prompt)

    # Test DrafterBench
    drafter_loader = get_bench_loader(Benchmark.DRAFTER_BENCH)()
    drafter_questions = drafter_loader.load_questions()
    drafter_sample = drafter_questions[0]
    print(drafter_sample.question_id)
    # Generate and save Drafter Bench prompt
    drafter_prompt = build_filtering_prompt(drafter_sample)
    with open("drafter_bench_universal_prompt.txt", "w", encoding="utf-8") as f:
        f.write("==== Prompt for DRAFTER_BENCH ====\n")
        f.write(drafter_prompt)

    print("Prompts saved")