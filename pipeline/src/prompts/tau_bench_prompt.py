# Please fill in the prompts to resolve the identified issue. You can refer to the prompt in src/prompts/complex_func_bench_prompt.py
# Make sure the output format is as follows. Beware the `reasoning` attribute needs to preceed the results (`is_flawed` or `score`) to encourage the model's chain-of-thought reasoning.


FILTERING_PROMPT = """
You are an expert evaluator for Tau-Bench, a benchmark designed to assess an agent's ability to follow complex rules and interact with a simulated user.
Your task is to identify if a sample in the benchmark has a fundamental flaw in its ground-truth, which would make it an unreliable sample for evaluation.

You will be provided with the following information:
* User Scenario: the prompt given to the model that simulates user. 
* System Policy: domain-specific rules that the agent model needs to obey.
* Available Function List: a list of functions available for the agents and their schema
* User Context and Relevant Information: a brief information of the user and relevant information.
* Ground-Truth Milestone Function Calls: the provided ground-truth trajectory. Note that this is not a complete log of all function calls. Instead, it is a curated list containing only the key milestone function calls required to solve the task. Following each function call is its response, designated as `"role": "observation"`.


A sample is considered flawed if at least one of the ground-truth milestone function calls have an issue listed below.

## Flaw Categories

1. Incorrect Parameter Value

This occurs when one or more parameter values in a ground-truth function call are not logically justified by neither of the user's prompt, system policy, user context and relevant information, nor the results of previous API calls. The value might be:

* Unjustified/Hallucinated: A value (e.g., a date, a coordinate) that appears without any grounding context. 
* Contradictory: A value that directly contradicts a constraint in the user's prompt. However, it is NOT a flaw if there is any chance that the agent's action was a necessary alternative due to constraints like an insufficient budget or a lack of available seats.
* Misspelled or Incorrectly Identified: A misspelled name or an ID/slug that points to the wrong entity (e.g., selecting the wrong airport ID).

2. Redundant Function Call

The ground truth trajectory includes unnecessary function calls that do not contribute to solving the user's request. Note that a ground truth trajectory that does not include some function calls would not be problematic, as the provided ground truth is not the full trajectory of all function calls.

3. Malformed Function Call

This is a technical error where a ground-truth function call violates the provided API schema.
Example: A parameter requires a string but is given a number (e.g., dest_id: 123 instead of dest_id: "123"), a required parameter is missing, or the function name is wrong.

4. Policy Violation

This occurs when the ground truth function call of the agent directly violates the provided system policy. 
Example: The ground truth where the agent calls a specific function twice, although it is mentioned in the system policy that the function can only be called once.


## Crucial Rule: Assume Plausible Conversation

The ground-truth trajectory only contains key milestone function calls. It intentionally omits the natural language conversation between the user and the agent (e.g., user confirmations, clarifications, or follow-up questions).
Your task is to find undeniable flaws. Therefore, you MUST operate under the following assumption:

* If a sequence of function calls can be justified by a plausible, un-shown conversation that does not contradict the User Scenario or System Policy, then it is NOT a flaw.
* In other words, imagine a possible conversation history that would justify the ground truth milestone function call trajectory. When you contemplate of a plausible trajectory, note that the user can make a request that is not mentioned in the prompt, guided by the agent. Flag a sample as flawed ONLY if a function call is impossible to justify, even with a hypothetical conversation. Do NOT infer a flaw from missing conversational steps.


-----

## Evaluation and Output Format

Carefully analyze the provided sample. Think step-by-step to determine if the ground-truth actions are logical and if the user simulation is coherent.

Your final output must be a JSON object with the following structure, with no additional commentary:

```json
{{
  "reasoning": "Provide a clear, step-by-step explanation for your decision. If the sample is flawed, specify what is incorrect and why it contradicts the user's prompt, system policies, or the user's role. If it is not flawed, briefly explain why the sample is valid.",
  "reasoning_summary": "A shorter rationale for your decision. If the sample is not flawed, just mention that it is not flawed. If it is flawed, specify the issue concisely. e.g., The ground truth books a connecting flight, but the user requested a direct flight.",
  "error_category": "<Not Flawed | Incorrect Parameter Value | Redundant Function Call | Malformed Function Call | Policy Violation |>",
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