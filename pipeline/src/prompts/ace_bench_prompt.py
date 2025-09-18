
# NOTE: in ACEBench, agent system prompt includes available function list


FILTERING_PROMPT = """You are an expert evaluator for ACEBench, a benchmark designed to assess an agent's ability to perform tool usage (function calling) across scenarios of increasing complexity and realism.

Your task is to identify if a sample in the benchmark has a fundamental flaw in its ground-truth, which would make it an unreliable sample for evaluation.

You will be provided with the following information:
* **Conversation History**: The conversation history between the user and the agent model. The agent's next function call immediately following this history is what will be evaluated.
* **Agent System Prompt**: the system prompt used to initialize the agent model. this may be a specific instruction on the answer style, domain-specific policy that the agent needs to follow, etc. 
* **API Specification**: a list of functions available for the agents and their schema
* **Ground-Truth Function Calls**: the provided ground-truth trajectory of function calls. When this is empty, it means that the agent needs to call nothing to be scored as correct. Note that the list includes only the function calls; it does not include the natural language response that the agent generates for the user.

A sample is **flawed** if it exhibits one or more of the issues described below.

## Flaw Categories

Below is the categorization of benchmark issues, outlined according to its **relevant benchmark component**. A sample is considered flawed if it has one or more of the issues below.


### User

* Vague instruction: The user's prompt is too ambiguous or underspecified for a single, correct function call, yet the benchmark expects one.
  * Example: The user asks "Where's the Golden Gate Bridge?", but the ground truth arbitrarily expects a distance calculation from a specific, unmentioned point.

### Environment

This category covers flaws within the agent's operating environment—the tools and API results—which can make a task unsolvable regardless of the agent's logic.

* Insufficient toolsets: the environment does not provide the necessary tools (functions), making the agent impossible to solve the task even with a combination of multiple tools and reasoning.
  * Example: A user asks for an advanced file manipulation, while the environment only provides basic tools like `mk` or `ls`.

* Flawed function design: the naming or the description of an available function is misleading or contradicts its actual functionality.
  * Example: A function named `vt_get_votes_on_ip_address` provides "example.com" as an example for its argument value in its schema.


### Ground-Truth

This category addresses errors in the provided ground-truth trajectory, where the supposed correct solution is itself incorrect, forcing any correct agent to fail the evaluation.


* Malformed function calls: A technical error where a ground-truth function call violates the provided API schema.
  * Example: A parameter requires a string but is given a number (e.g., dest_id: 123 instead of dest_id: "123"), a required parameter is missing, the function name is wrong, or a parameter value is misspelled (e.g., sort_by: "popularitye" instead of "popularity").

* Incorrect function calls: A function call is syntactically valid but logically flawed. The function choice or a parameter value contradicts the user's request or the context from previous steps.
  * Unjustified/Hallucinated Parameters: A value (e.g., a date, a coordinate) that appears without any grounding context. For example, searching for a hotel on a date that was not returned by a preceding flight search.
  * Contradictory: A value that directly contradicts a constraint in the user's prompt. However, it is NOT a flaw if there is any chance that the agent's action was a necessary alternative due to constraints like an insufficient budget or a lack of available seats.
  * Policy Violation: A function call in the ground truth trajectory directly violates the policy specified in the provided system prompt. Example: The ground truth where the agent calls a specific function twice, although it is mentioned in the system policy that the function can only be called once.
  * Misspelled or Incorrectly Identified Parameter Values: A misspelled name or an ID/slug that points to the wrong entity (e.g., selecting the wrong airport ID).

* Redundant/ungrounded function calls: The ground truth function call trajectory consists of function calls that are redundant in solving the task, ungrounded by the context, or irrelevant in solving the task.
  * Irrelevant tool call: A function call in the ground truth trajectory is totally irrelevant to the task or belongs to a completely different domain. Example: agent calls a function to reserve a flight, though it was asked to process product exchange.
  * Redundant tool call: A function call that is not necessary in solving the task. Example: the agent is asked to search for attractions until it finds one that meets a certain condition; However, the agent performs the search in an arbitrary order, resulting in an excessive number of function calls.


## Evaluation and Output Format

Carefully analyze the provided sample. Think step-by-step to determine if the ground-truth function calls are logical and if the parameter values are properly justified.

Your final output must be a JSON object with the following structure, with no additional commentary:

```json
{{
  "reasoning": "Provide a clear, step-by-step explanation for your decision. If the sample is flawed, specify what is incorrect and why it contradicts the user's prompt, API schema, or context. If it is not flawed, briefly explain why the sample is valid.",
  "reasoning_summary": "A shorter rationale for your decision. If the sample is not flawed, just mention that it is not flawed. If it is flawed, specify the issue concisely. e.g., The parameter 'dest_id' requires a string but is given a number.",
  "error_category": "The category that corresponds to the issue. e.g., \"Flawed function response\". If the sample is not flawed, use \"Not Flawed\".",
  "is_flawed": <true_or_false>

}}
```

## Sample to be evaluated

### Conversation History 

{previous_conversation_history}

### Agent System Prompt

{agent_system_prompt}

### Ground-truth function call trajectory

```json
{gt_conv_traj}
```

"""

SCORING_PROMPT = """You are an expert evaluator for ACEBench, a benchmark for evaluating LLMs' ability to perform tool usage (function calling) across scenarios of increasing complexity and realism.

Your task is to score the given sample across 5 dimensions, each rated from 1-5 where:
- 1: Poor performance, fundamental errors
- 2: Below average, significant issues
- 3: Average performance, some errors but generally functional
- 4: Good performance, minor issues
- 5: Excellent performance, robust and reliable

## Sample to be evaluated

### Conversation History

{previous_conversation_history}

### Agent System Prompt

```
{agent_system_prompt}
```

### Ground-truth function call trajectory

```json
{gt_conv_traj}
```

## Scoring Dimensions

1. **Parameter Accuracy** (1-5): How accurately does the model generate parameter values? Consider semantic correctness, proper formatting, and consistency with expected schema.

2. **Function Call Correctness** (1-5): Does the model make the correct function calls with appropriate parameters? Consider tool selection, parameter completeness, and adherence to API specifications.

3. **Error Handling** (1-5): How well does the model handle imperfect instructions, missing parameters, or invalid requests? Consider robustness and appropriate error responses.

4. **Multi-turn Reasoning** (1-5): For multi-turn tasks, how well does the model maintain context, track state changes, and make coherent sequences of function calls?

5. **Real-world Applicability** (1-5): How realistic and practical are the model's responses for real-world tool usage scenarios?

## Output Format

Provide your evaluation in the following JSON format:
[
  {{
    "dimension": "parameter accuracy",
    "reasoning": "Evaluate how accurately the model generates parameter values. Consider whether values are semantically correct, properly formatted, and consistent with the expected schema.",
    "score": <1-5>
  }},
  {{
    "dimension": "function call correctness",
    "reasoning": "Assess whether the model makes the correct function calls with appropriate parameters. Consider tool selection, parameter completeness, and adherence to API specifications.",
    "score": <1-5>
  }},
  {{
    "dimension": "error handling",
    "reasoning": "Evaluate how well the model handles imperfect instructions, missing parameters, or invalid requests. Consider robustness and appropriate error responses.",
    "score": <1-5>
  }},
  {{
    "dimension": "multi-turn reasoning",
    "reasoning": "For multi-turn tasks, assess the model's ability to maintain context, track state changes, and make coherent sequences of function calls.",
    "score": <1-5>
  }},
  {{
    "dimension": "real-world applicability",
    "reasoning": "Evaluate how realistic and practical the model's responses are for real-world tool usage scenarios.",
    "score": <1-5>
  }}
]

Provide detailed reasoning for each score, considering the specific context and requirements of the task."""

# Test
if __name__ == "__main__":
    from src.utils.types import Benchmark
    from src.bench_loaders import get_bench_loader

    # Test ACE Bench
    ace_loader = get_bench_loader(Benchmark.ACE_BENCH)()
    ace_questions = ace_loader.load_questions()

    for ace_sample in ace_questions:
      if "normal_multi_turn" in ace_sample.question_id:
        break
    
    if ace_sample:
        print(f"ACE Bench sample ID: {ace_sample.question_id}")

        # Generate formatted prompt using the FILTERING_PROMPT template
        ace_filtering_prompt = FILTERING_PROMPT.format(
            previous_conversation_history=ace_sample.previous_conversation_history,
            agent_system_prompt=ace_sample.agent_system_prompt,
            gt_conv_traj=ace_sample.gt_conv_traj
        )

        with open("ace_bench_formatted_prompt.txt", "w", encoding="utf-8") as f:
            f.write(ace_filtering_prompt)

        print("ACE Bench formatted prompt saved to ace_bench_formatted_prompt.txt")
    else:
        print("No ACE Bench sample found with ID '1'")