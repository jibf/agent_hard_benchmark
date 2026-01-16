FILTERING_PROMPT = """
You are an expert evaluator for ComplexFuncBench, a benchmark designed to assess the complex function-calling capabilities of LLMs. 
Your task is to determine if a given benchmark sample has a fundamental flaw in its user prompt, environment, or ground-truths, which would make it unable to be incorporated in the evaluation.

You will be provided with:

* **User Prompt:** The user's task description and instructions.
* **Available Functions:** A complete list of functions the agent can use, including their JSON schemas.
* **Ground-Truth Trajectory:** The reference sequence of function calls. Note that messages with `"role": "observation"` are the outputs of the immediately preceding function call. 
  
A sample is **flawed** if it exhibits one or more of the issues described below.

## Flaw Categories

Below is the categorization of benchmark issues, outlined according to its **relevant benchmark component**. A sample is considered flawed if it has one or more of the issues below.

### User

* Vague instruction: The user's prompt is too ambiguous or underspecified for a single, correct function call, yet the benchmark expects one.


### Environment

This category covers flaws within the agent's operating environment—the tools and API results—which can make a task unsolvable regardless of the agent's logic.

* Flawed function response: The pre-computed API response provided in the benchmark is incorrect, misleading, or doesn't contain the information needed to fulfill the user's request. Since the agent relies on these responses, a flawed API response makes the task unsolvable.
  * Look for:
    * Incorrect resolution: An ambiguous name in the function call is resolved to the wrong entity in the response.
    * Irrelevant results: The API returns a list of items that are completely irrelevant to the user's request 

* Insufficient toolsets: the environment does not provide the necessary tools (functions), making the agent impossible to solve the task even with a combination of multiple tools and reasoning.

* Flawed function design: the naming or the description of an available function is misleading or contradicts its actual functionality.


### Ground-Truth

This category addresses errors in the provided ground-truth trajectory, where the supposed correct solution is itself incorrect, forcing any correct agent to fail the evaluation.


* Malformed function calls: A technical error where a ground-truth function call violates the provided API schema.

* Incorrect function calls: A function call is syntactically valid but logically flawed. The function choice or a parameter value contradicts the user's request or the context from previous steps.
  * Unjustified/Hallucinated Parameters: A value (e.g., a date, a coordinate) that appears without any grounding context. 
  * Contradictory Parameter Values: A value that directly contradicts a constraint in the user's prompt. 
  * Misspelled or Incorrectly Identified Parameter Values: A misspelled name or an ID/slug that points to the wrong entity.

* Redundant/ungrounded function calls: The ground truth function call trajectory consists of function calls that are redundant in solving the task, ungrounded by the context, or irrelevant in solving the task.
  * Irrelevant tool call: A function call in the ground truth trajectory is totally irrelevant to the task or belongs to a completely different domain. 
  * Redundant tool call: A function call that is not necessary in solving the task. 

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


## Sample to be evaluated

### User's Prompt

```
{instruction}
```

### List of available functions and their schema

```json
{available_function_list}
```

### Ground-truth function call trajectory 
* Note that messages with "role": "observation" are the results of the function call right before.

```json
{gt_conv_traj}
```
"""

SCORING_PROMPT = """
You are an expert AI assistant specializing in the meticulous evaluation of function-calling benchmarks. Your task is to assess how effectively a given benchmark sample measures the capabilities of AI agents.

This evaluation is for the {benchmark} benchmark.

You will be given three pieces of information:

1.  User Prompt: This could be either:
    - For ComplexFuncBench: The original request from the user.
    - For Tau-bench: The instruction given to the AI model (acting as a customer service representative). In this case, the "user prompt" actually describes the persona and scenario the AI should simulate, not a direct user request.
2.  Available Function List: The JSON schema of tools the agent can use.
3.  Ground-Truth Conversation: The sequence of assistant and function call result (marked as "role": "observation") messages. Note that whenever an assistant makes a function call, the result will be in the subsequent "observation" message.


-----

Evaluation Criteria:

Evaluate the sample on each of the following dimensions using a 1-5 point scale. Below are example descriptions for scores 1, 3, and 5. You are veryencouraged to use scores 2 and 4 for cases that fall between these descriptions, since most real samples will likely fall somewhere between the anchor points described below. Provide a clear, critical reasoning for every score.

1. function Necessity
* 5 points: Every single step of the sub-task required to solve the given task is fundamentally impossible without the specific tools provided.
* 3 points: The core task requires tools to complete, but small peripheral aspects or subtasks could be handled using internal knowledge of model intensively trained on up-to-date data. e.g., identifying the airport name given the city
* 1 points: A model intensively trained on up-to-date data could potentially solve the task without any tools, making the function calls feel optional or of limited value.

2. Planning and Context Depth 
* 5 points: Requires highly complex, non-linear planning with multiple dependencies between function calls. The agent must track a long and detailed context to decide every next function call.
* 3 points: Requires a standard multi-step plan where the output of one step informs the next.
* 1 points: Requires only a single function call or a static, predefined sequence of calls. Context is not important.

3. Parameter Generation
* 5 points: Generating the correct parameters for function calls requires deep semantic understanding of user intent. Some of the function calls requires a long, complex value (e.g., tokens).
* 3 points: Requires some basic reasoning or extraction from context (e.g., calculating a date from "tomorrow").
* 1 points: Parameters are simple values copied directly from the user prompt.

4. function Selection Difficulty
* 5 points: The toolset contains highly plausible and confusing distractors (e.g., such as similarly named tools). The task is design to actively tempt an agent into making the wrong choice, which results in the failure of the task.
* 3 points: The toolset contains a few distinct but related options, requiring the agent to discern subtle differences to make the correct choice based on the context and correct understanding of the user's intention.
* 1 points: The function choice is obvious every step. The selection is straightforward and does not require deep reasoning or understanding of the context.

5. Real-World Applicability
* 5 points: Represents an extremely common, daily scenario that millions of users encounter with identical specificity. Every detail reflects typical user behavior patterns and natural language use.
* 3 points: Based on realistic, common scenarios that people do encounter, but with some specific requirements or constraints that are slightly artificial or less typical in practice.
* 1 points: Clearly synthetic or academic in nature - designed for evaluation rather than reflecting genuine user needs.

-----

Output Format:

Based on your evaluation, aggregate the scores of each dimension in the jsonl format as follows. 
Note that the dimensions must be arranged in the order listed above, and ensure that no dimensions are skipped.
Do not include any additional comments or explanations, and only include the JSONL output. That is, your response should start directly with [ and end with ].

Example:
[
    {{
    "dimension": "function necessity",
    "reasoning": "The user's goal of booking a flight and a taxi involves interacting with external reservation systems. This is fundamentally impossible to achieve with only the model's internal knowledge. However, small sub-tasks such as identifying the closest airport from the user's location could be handled without external APIs.",
    "score": 3
    }},
    {{
    "dimension": "planning and context depth",
    "reasoning": "The task requires a sequence: 1. Search for a flight, 2. Use the flight's arrival airport to book a taxi. This is a **standard multi-step plan with a clear, linear dependency**. However, it does not require **complex, non-linear planning or adaptation to unexpected results**, which would be necessary for a score of 5.",
    "score": 4
    }},
    {{
    "dimension": "parameter generation",
    "reasoning": "Assuming the user prompt mentioned 'tomorrow', the agent needs to calculate the exact date. This is a **form of basic reasoning**, fitting the 3-point criteria. It does not require **deep semantic inference or the generation of a long, complex value** (like a full JSON object for filtering).",
    "score": 3
    }},
    {{
    "dimension": "function selection difficulty",
    "reasoning": "The user's intent to 'search for a flight' and 'book a taxi' maps directly to tools like `search_flights` and `book_taxi`. There are **no plausible or confusing distractor tools** mentioned. The choice is obvious and straightforward.",
    "score": 2
    }},
    {{
    "dimension": "real-world applicability",
    "reasoning": "Booking a flight and then arranging for transportation from the airport is a very common and practical real-world scenario for travelers. However, some of the conditions that the user demands are a bit unrealistic.",
    "score": 3
    }}
]


-----

User Input:

### User Prompt

```
{instruction}
```

### Available Function List

```json
{available_function_list}
```

### Ground-truth conversation

```json
{gt_conv_traj}
```


"""

# Test
if __name__ == "__main__":
    import argparse
    from src.utils.types import Benchmark
    from src.bench_loaders import get_bench_loader

    parser = argparse.ArgumentParser(description="Generate formatted prompt for ComplexFuncBench")
    parser.add_argument("-q", "--question_id", type=str, default="1",
                       help="Question ID to format (e.g., Hotels-69, default: 1)")
    args = parser.parse_args()

    # Test ComplexFuncBench
    cfb_loader = get_bench_loader(Benchmark.COMPLEX_FUNC_BENCH)()
    cfb_questions = cfb_loader.load_questions()
    cfb_sample = None

    domain = '-'.join(args.question_id.split("-")[:-1])
    question_id = args.question_id.split("-")[-1]

    for question in cfb_questions:
        if question.question_id == question_id and question.task_name == domain:
            cfb_sample = question
            break

    if cfb_sample:
        print(f"ComplexFuncBench sample ID: {cfb_sample.question_id}")

        # Generate formatted prompt using the FILTERING_PROMPT template
        cfb_filtering_prompt = FILTERING_PROMPT.format(
            instruction=cfb_sample.instruction,
            available_function_list=cfb_sample.available_function_list,
            gt_conv_traj=cfb_sample.gt_conv_traj
        )

        output_filename = f"complex_func_bench_formatted_prompt.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(cfb_filtering_prompt)

        print(f"ComplexFuncBench formatted prompt saved to {output_filename}")
    else:
        print(f"No ComplexFuncBench sample found with ID '{args.question_id}'")