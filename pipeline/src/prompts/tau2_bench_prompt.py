# Please fill in the prompts to resolve the identified issue. You can refer to the prompt in src/prompts/complex_func_bench_prompt.py
# Make sure the output format is as follows. Beware the `reasoning` attribute needs to preceed the results (`is_flawed` or `score`) to encourage the model's chain-of-thought reasoning.


# FILTERING: 
# {{
#   "reasoning": "Provide a clear, step-by-step explanation for your decision. If the ground-truth is flawed, specify which argument is incorrect and why it contradicts the prompt or schema. If it is not flawed, briefly explain why the ground-truth is a correct interpretation of the user's request."
#   "reasoning_summary": "A shorter rationale for your decision. If the ground-truth is not flawed, just mention that it is not flawed. If the ground-truth is flawed, specify the issue concisely. e.g., The argument `search_type` in the function call `Search_Hotels` is supposed to be `district`, but is misspelled as `dustrict`.",
#   "error_category": "<Argument Value Mismatch | Argument Type Mismatch | Unjustified Assumption | Misspelling | Not Flawed>",
#   "is_flawed": <true_or_false>,
# }}

# SCORING: 
# [
#     {{
#     "dimension": "tool necessity",
#     "reasoning": "The user's goal of booking a flight and a taxi involves interacting with external reservation systems. This is fundamentally impossible to achieve with only the model's internal knowledge. However, small sub-tasks such as identifying the closest airport from the user's location could be handled without external APIs.",
#     "score": 3
#     }},
#     {{
#     "dimension": "planning and context depth",
#     "reasoning": "The task requires a sequence: 1. Search for a flight, 2. Use the flight's arrival airport to book a taxi. This is a **standard multi-step plan with a clear, linear dependency**. However, it does not require **complex, non-linear planning or adaptation to unexpected results**, which would be necessary for a score of 5.",
#     "score": 4
#     }},
#     {{
#     "dimension": "parameter generation",
#     "reasoning": "Assuming the user prompt mentioned 'tomorrow', the agent needs to calculate the exact date. This is a **form of basic reasoning**, fitting the 3-point criteria. It does not require **deep semantic inference or the generation of a long, complex value** (like a full JSON object for filtering).",
#     "score": 3
#     }},
#     {{
#     "dimension": "tool selection difficulty",
#     "reasoning": "The user's intent to 'search for a flight' and 'book a taxi' maps directly to tools like `search_flights` and `book_taxi`. There are **no plausible or confusing distractor tools** mentioned. The choice is obvious and straightforward.",
#     "score": 2
#     }},
#     {{
#     "dimension": "real-world applicability",
#     "reasoning": "Booking a flight and then arranging for transportation from the airport is a very common and practical real-world scenario for travelers. However, some of the conditions that the user demands are a bit unrealistic.",
#     "score": 3
#     }}
# ]
#



FILTERING_PROMPT = """
You are an expert evaluator for Tau-2-Bench, a benchmark designed to assess an agent's ability to follow complex rules and interact with a simulated user.
Your task is to determine if a given benchmark sample has a fundamental flaw in its user prompt, environment, or ground-truths, which would make it unable to be incorporated in the evaluation.


You will be provided with the following information:
* **Task Description/Instructions**: The prompt or scenario given to the model that simulates user. You need to expect how the model that simulates the user would behave given this instruction.
* **System Policy**: Domain-specific rules that the agent model needs to obey. This will be given as the system prompt for agent models.
* **User Context and Relevant Information**: a brief information of the user and relevant information. This may be in a form of system message given to the user-simulating model.
* **Initial State**: The initial environment setup and conditions before the task begins. This defines the starting state of the system.
* **Functions available to the agent**: a list of functions available for the agents and their schema.
* **Functions available to the user**: a list of functions available for the user-simulating models and their schema. When the user cannot directly call any functions, this is set to be empty.
* **Complete Evaluation Criteria**: The complete evaluation criteria including milestone ground-truth actions (function calls), natural language assertions, and final environment state assertion to validate. The sample is considered to be flawed if one or more of these criteria is unachievable. 
* **Ground-Truth Milestone Function Calls**: the provided ground-truth trajectory. Note that this is not a complete log of all function calls. Instead, it is a curated list containing only the key milestone function calls required to solve the task. Note that messages with `"role": "observation"` are the outputs of the immediately preceding function call.
A sample is **flawed** if it exhibits one or more of the issues described below.

## Flaw Categories

Below is the categorization of benchmark issues, outlined according to its **relevant benchmark component**. A sample is considered flawed if it has one or more of the issues below.

### Environment

This category covers flaws within the agent's operating environment—the tools and API results—which can make a task unsolvable regardless of the agent's logic.

* Flawed function design: the naming or the description of an available function is misleading or contradicts its actual functionality.

### Ground-Truth

This category addresses errors in the provided ground-truth trajectory, where the supposed correct solution is itself incorrect, forcing any correct agent to fail the evaluation.


* Malformed function calls: A technical error where a ground-truth function call violates the provided API schema.
  * Example: A parameter requires a string but is given a number , a required parameter is missing, the function name is wrong, or a parameter value is misspelled .

* Incorrect function calls: A function call is syntactically valid but logically flawed. The function choice or a parameter value contradicts the user's request or the context from previous steps.
  * Unjustified/Hallucinated Parameters: A value that appears without any grounding context. For example, searching for a hotel on a date that was not returned by a preceding flight search.
  * Contradictory: A value that directly contradicts a constraint in the user's prompt. However, it is NOT a flaw if there is any chance that the agent's action was a necessary alternative due to constraints like an insufficient budget or a lack of available seats.
  * Policy Violation: A function call in the ground truth trajectory directly violates the provided system policy. Example: The ground truth where the agent calls a specific function twice, although it is mentioned in the system policy that the function can only be called once.
  * Misspelled or Incorrectly Identified Parameter Values: A misspelled name or an ID/slug that points to the wrong entity (e.g., selecting the wrong airport ID).

* Redundant/ungrounded function calls: The ground truth function call trajectory consists of function calls that are redundant in solving the task, ungrounded by the context, or irrelevant in solving the task.
  * Irrelevant tool call: A function call in the ground truth trajectory is totally irrelevant to the task or belongs to a completely different domain. Example: agent calls a function to reserve a flight, though it was asked to process product exchange.
  * Redundant tool call: A function call that is not necessary in solving the task. Example: the agent is asked to search for attractions until it finds one that meets a certain condition; However, the agent performs the search in an arbitrary order, resulting in an excessive number of function calls.


## Crucial Rule: Actively Reconstruct the Conversation

The ground-truth trajectory only contains key milestone function calls. It intentionally omits function calls that are less important for evaluation and the natural language conversation between the user and the agent (e.g., user confirmations, request, clarifications, or follow-up questions).
Your task is to find undeniable flaws. Therefore, you MUST operate under the following assumption:

* For example, the ground truth milestone sequence may not contain a call that authenticates the user identity. It may have been intentionallly omitted from the milestone sequence, since it is considered less important than calls that explicitly process user requests. Therefore, lack of authentication, user's confirmation or request, clarification should NOT be the sole reason to judge a sample as flawed.
* If a sequence of function calls can be justified by a plausible, un-shown conversation that does not contradict the User Scenario or System Policy, then it is NOT a flaw. The agent would have explained the user why it cannot process his request, although it is not shown in the milestone trajectory.
* In other words, imagine a possible conversation history that would justify the ground truth milestone function call trajectory. When you contemplate of a plausible trajectory, note that the user can make a request that is not mentioned in the prompt, guided by the agent. Flag a sample as flawed ONLY if a function call is impossible to justify, even with a hypothetical conversation. Do NOT infer a flaw from missing conversational steps.


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

### Task Description/Instructions

```
{instruction}
```

### System Policy

{agent_system_prompt}

### User context and relevant information

{user_context}

### Initial Status

{initial_state}

### Functions available to the agent and their schema

```json
{available_function_list}
``` 

### Functions available to the user and their schema

```json
{available_user_function_list}
```

### Complete Evaluation Criteria

```json
{evaluation_criteria}
```

### Ground-Truth Milestone Function Calls
* Note that messages with "role": "observation" are the results of the function call right before.

```json
{gt_conv_traj}
```
"""

SCORING_PROMPT = ""

# Test
if __name__ == "__main__":
    from src.utils.types import Benchmark
    from src.bench_loaders import get_bench_loader
    import argparse
    import sys


    parser = argparse.ArgumentParser(description="Generate formatted prompt for Tau Bench")
    parser.add_argument("-q", "--question_id", type=str,
                       help="Question ID to format (e.g., etail-132)")
    parser.add_argument("--save-all", action="store_true",
                       help="Save all prompts to tau2_bench_prompts folder with format task_name-question_id.txt")
    args = parser.parse_args()

    # Test Tau2 Bench
    tau2_loader = get_bench_loader(Benchmark.TAU2_BENCH)()
    tau2_questions = tau2_loader.load_questions()

    if args.save_all:
        # Save all prompts to tau2_bench_prompts folder
        import os
        os.makedirs("tau2_bench_prompts", exist_ok=True)

        for question in tau2_questions:
            tau2_prompt = FILTERING_PROMPT.format(
                instruction=question.instruction,
                agent_system_prompt=question.agent_system_prompt,
                user_context=question.user_context,
                initial_state=getattr(question, 'initial_state', ''),
                available_function_list=question.available_function_list,
                available_user_function_list=getattr(question, 'available_user_function_list', []),
                evaluation_criteria=getattr(question, 'evaluation_criteria', {}),
                gt_conv_traj=question.gt_conv_traj
            )

            filename = f"tau2_bench_prompts/{question.task_name}-{question.question_id}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(tau2_prompt)

        print(f"Saved {len(tau2_questions)} prompts to tau2_bench_prompts/ folder")
    else:
        # Original single question functionality
        if not args.question_id:
            print("Error: --question_id is required when not using --save-all")
            sys.exit(1)

        task_name, question_id = args.question_id.split("-")

        tau2_sample = None
        for question in tau2_questions:
            if question.question_id == question_id and question.task_name == task_name:
                tau2_sample = question
                break

        if tau2_sample is None:
            print(f"Error: Could not find question with ID {args.question_id}")
            sys.exit(1)

        print(tau2_sample.question_id)

        # Generate formatted prompt using the FILTERING_PROMPT template
        tau2_prompt = FILTERING_PROMPT.format(
            instruction=tau2_sample.instruction,
            agent_system_prompt=tau2_sample.agent_system_prompt,
            user_context=tau2_sample.user_context,
            initial_state=getattr(tau2_sample, 'initial_state', ''),
            available_function_list=tau2_sample.available_function_list,
            available_user_function_list=getattr(tau2_sample, 'available_user_function_list', []),
            evaluation_criteria=getattr(tau2_sample, 'evaluation_criteria', {}),
            gt_conv_traj=tau2_sample.gt_conv_traj
        )

        with open("tau2_bench_formatted_prompt.txt", "w", encoding="utf-8") as f:
            f.write(tau2_prompt)

        print("Tau2 Bench formatted prompt saved to tau2_bench_formatted_prompt.txt")