# Please fill in the prompts to resolve the identified issue. You can refer to the prompt in src/prompts/complex_func_bench_prompt.py
# Make sure the output format is as follows. Beware the `reasoning` attribute needs to preceed the results (`is_flawed` or `score`) to encourage the model's chain-of-thought reasoning.


FILTERING_PROMPT = """
You are an expert evaluator for Tau-Bench, a benchmark designed to assess an agent's ability to follow complex rules and interact with a simulated user.
Your task is to determine if a given benchmark sample has a fundamental flaw in its user prompt, environment, or ground-truths, which would make it unable to be incorporated in the evaluation.


You will be provided with the following information:
* **User Scenario**: The system prompt given to the model that simulates user. You need to expect how the user model would behave based on this.
* **System Policy**: Domain-specific rules that the agent model needs to obey. This will be given as the system prompt for agent models.
* **Available Function List**: a list of functions available for the agents and their schema
* **User Context and Relevant Information**: a brief information of the user and relevant information.
* **Ground-Truth Milestone Function Calls**: the provided ground-truth trajectory. Note that this is not a complete log of all function calls. Instead, it is a curated list containing only the key milestone function calls required to solve the task. Note that messages with `"role": "observation"` are the outputs of the immediately preceding function call. 

A sample is **flawed** if it exhibits one or more of the issues described below.

## Flaw Categories

Below is the categorization of benchmark issues, outlined according to its **relevant benchmark component**. A sample is considered flawed if it has one or more of the issues below.

### Environment

This category covers flaws within the agent's operating environment—the tools and API results—which can make a task unsolvable regardless of the agent's logic.

* Flawed function design: the naming or the description of an available function is misleading or contradicts its actual functionality.
  * Example: A function named `vt_get_votes_on_ip_address` provides "example.com" as an example for its argument value in its schema. 

### Ground-Truth

This category addresses errors in the provided ground-truth trajectory, where the supposed correct solution is itself incorrect, forcing any correct agent to fail the evaluation.


* Malformed function calls: A technical error where a ground-truth function call violates the provided API schema.
  * Example: A parameter requires a string but is given a number (e.g., dest_id: 123 instead of dest_id: "123"), a required parameter is missing, the function name is wrong, or a parameter value is misspelled (e.g., sort_by: "popularitye" instead of "popularity").

* Incorrect function calls: A function call is syntactically valid but logically flawed. The function choice or a parameter value contradicts the user's request or the context from previous steps.
  * Unjustified/Hallucinated Parameters: A value (e.g., a date, a coordinate) that appears without any grounding context. For example, searching for a hotel on a date that was not returned by a preceding flight search.
  * Contradictory: A value that directly contradicts a constraint in the user's prompt. However, it is NOT a flaw if there is any chance that the agent's action was a necessary alternative due to constraints like an insufficient budget or a lack of available seats.
  * Policy Violation: A function call in the ground truth trajectory directly violates the provided system policy. Example: The ground truth where the agent calls a specific function twice, although it is mentioned in the system policy that the function can only be called once.
  * Misspelled or Incorrectly Identified Parameter Values: A misspelled name or an ID/slug that points to the wrong entity (e.g., selecting the wrong airport ID).

* Redundant/ungrounded function calls: The ground truth function call trajectory consists of function calls that are redundant in solving the task, ungrounded by the context, or irrelevant in solving the task.
  * Irrelevant tool call: A function call in the ground truth trajectory is totally irrelevant to the task or belongs to a completely different domain. Example: agent calls a function to reserve a flight, though it was asked to process product exchange.
  * Redundant tool call: A function call that is not necessary in solving the task. However, note that function calls that does not change the database status are tolerable even if they are redundant. EXAMPLE: a sample is not flawed even if calls `get_order_details` for irrelevant items. 


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


## Target Sample

### User Scenario

```
{instruction}
```

### System Policy

{agent_system_prompt}


### User context and relevant information
{user_context}

### List of available functions and their schema

```json
{available_function_list}
```


### Ground-Truth Milestone Function Calls 
* Note that messages with "role": "observation" are the results of the function call right before.

```json
{gt_conv_traj}
```
"""


SCORING_PROMPT = """
You are an expert evaluator for agentic benchmarks, tasked with assessing the quality of a given question sample. Your goal is to provide a multi-dimensional score for each sample, reflecting key aspects of a good agentic prompt. 
The final score will be a sum of the scores from each dimension.

You will be provided with the following information:
* User Scenario: the prompt given to the model that simulates user. 
* System Policy: domain-specific rules that the agent model needs to obey.
* Available Function List: a list of functions available for the agents and their schema
* User Context and Relevant Information: a brief information of the user and relevant information.
* Ground-Truth Milestone Function Calls: the provided ground-truth trajectory. Note that this is not a complete log of all function calls. Instead, it is a curated list containing only the key milestone function calls required to solve the task. Following each function call is its response, designated as `"role": "observation"`.


## Scoring Dimensions

Evaluate the sample on each of the following dimensions using a 1-5 point scale. Below are example descriptions for scores 1, 3, and 5. You are veryencouraged to use scores 2 and 4 for cases that fall between these descriptions, since most real samples will likely fall somewhere between the anchor points described below. Provide a clear, critical reasoning for every score.

1. Tool Necessity
* 5 points: Every single step of the sub-task required to solve the given task is fundamentally impossible without the specific tools provided.
* 3 points: The core task requires tools to complete, but small peripheral aspects or subtasks could be handled using internal knowledge of model intensively trained on up-to-date data. e.g., identifying the airport name given the city
* 1 points: A model intensively trained on up-to-date data could potentially solve the task without any tools, making the tool calls feel optional or of limited value.

2. Planning and Context Depth 
* 5 points: Requires highly complex, non-linear planning with multiple dependencies between tool calls. The agent must track a long and detailed context to decide every next function call.
* 3 points: Requires a standard multi-step plan where the output of one step informs the next.
* 1 points: Requires only a single tool call or a static, predefined sequence of calls. Context is not important.

3. Parameter Generation
* 5 points: Generating the correct parameters for function calls requires deep semantic understanding of user intent. Some of the function calls requires a long, complex value (e.g., tokens).
* 3 points: Requires some basic reasoning or extraction from context (e.g., calculating a date from "tomorrow").
* 1 points: Parameters are simple values copied directly from the user prompt.

4. Tool Selection Difficulty
* 5 points: The toolset contains highly plausible and confusing distractors (e.g., such as similarly named tools). The task is design to actively tempt an agent into making the wrong choice, which results in the failure of the task.
* 3 points: The toolset contains a few distinct but related options, requiring the agent to discern subtle differences to make the correct choice based on the context and correct understanding of the user's intention.
* 1 points: The tool choice is obvious every step. The selection is straightforward and does not require deep reasoning or understanding of the context.

5. Real-World Applicability
* 5 points: Represents an extremely common, daily scenario that millions of users encounter with identical specificity. Every detail reflects typical user behavior patterns and natural language use.
* 3 points: Based on realistic, common scenarios that people do encounter, but with some specific requirements or constraints that are slightly artificial or less typical in practice.
* 1 points: Clearly synthetic or academic in nature - designed for evaluation rather than reflecting genuine user needs.

## Final Output Format

Carefully analyze the provided sample using the dimensions above. Your final output must be a JSON object with the following structure, with no additional commentary:

```json
{{
  "justifications": {{
    "tool_necessity": "The user's goal of booking a flight and a taxi involves interacting with external reservation systems. This is fundamentally impossible to achieve with only the model's internal knowledge. However, small sub-tasks such as identifying the closest airport from the user's location could be handled without external APIs."
    "planning_and_context_depth": "The task requires a sequence: 1. Search for a flight, 2. Use the flight's arrival airport to book a taxi. This is a **standard multi-step plan with a clear, linear dependency**. However, it does not require **complex, non-linear planning or adaptation to unexpected results**, which would be necessary for a score of 5.",
    "parameter_generation": "Assuming the user prompt mentioned 'tomorrow', the agent needs to calculate the exact date. This is a **form of basic reasoning**, fitting the 3-point criteria. It does not require **deep semantic inference or the generation of a long, complex value** (like a full JSON object for filtering).",
    "tool_selection_difficulty": "The user's intent to 'search for a flight' and 'book a taxi' maps directly to tools like `search_flights` and `book_taxi`. There are **no plausible or confusing distractor tools** mentioned. The choice is obvious and straightforward.",
    "real_world_applicability": "Booking a flight and then arranging for transportation from the airport is a very common and practical real-world scenario for travelers. However, some of the conditions that the user demands are a bit unrealistic."
  }},
  "scores": {{
    "tool_necessity": 3,
    "planning_and_context_depth": 4,
    "parameter_generation": 3,
    "tool_selection_difficulty": 2, 
    "real_world_applicability": 3
  }},
}}
```

## Target Sample

### User Scenario

```
{instruction}
```

### System Policy

{agent_system_prompt}


### User context and relevant information
{user_context}

### List of available functions and their schema

```json
{available_function_list}
```

### Ground-Truth Milestone Function Calls
* Note that messages with "role": "observation" are the results of the function call right before.

```json
{gt_conv_traj}
```
"""

# Test
if __name__ == "__main__":
    import argparse
    from src.utils.types import Benchmark
    from src.bench_loaders import get_bench_loader

    parser = argparse.ArgumentParser(description="Generate formatted prompt for Tau Bench")
    parser.add_argument("-q", "--question_id", type=str, 
                       help="Question ID to format (e.g., etail-132)")
    args = parser.parse_args()

    # Parse the question ID format: TaskName-QuestionNumber
    tau_loader = get_bench_loader(Benchmark.TAU_BENCH)()
    tau_questions = tau_loader.load_questions()
    tau_sample = None

    for question in tau_questions:
        if question.question_id == args.question_id:
            tau_sample = question
            break

    if tau_sample:
        print(f"Tau Bench sample found - Task: {tau_sample.task_name}, ID: {tau_sample.question_id}")

        # Generate formatted prompt using the FILTERING_PROMPT template
        tau_filtering_prompt = FILTERING_PROMPT.format(
            instruction=tau_sample.instruction,
            agent_system_prompt=tau_sample.agent_system_prompt,
            user_context=tau_sample.user_context,
            available_function_list=tau_sample.available_function_list,
            gt_conv_traj=tau_sample.gt_conv_traj
        )

        output_filename = f"tau_bench_formatted_prompt.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(tau_filtering_prompt)

        print(f"Tau Bench formatted prompt saved to {output_filename}")
    else:
        print(f"No Tau Bench sample found with ID '{args.question_id}'")