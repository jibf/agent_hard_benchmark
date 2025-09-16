
# NOTE: in ACEBench, agent system prompt includes available function list


FILTERING_PROMPT = """You are an expert evaluator for ACEBench, a benchmark designed to assess an agent's ability to perform tool usage (function calling) across scenarios of increasing complexity and realism.

Your task is to identify if a sample in the benchmark has a fundamental flaw in its ground-truth, which would make it an unreliable sample for evaluation.

You will be provided with the following information:
* Conversation History: The conversation history between the user and the agent model. The agent's next function call immediately following this history is what will be evaluated.
* Agent System Prompt: the system prompt used to initialize the agent model. this may be a specific instruction on the answer style, domain-specific policy that the agent needs to follow, etc. 
* API Specification: a list of functions available for the agents and their schema
* Ground-Truth Function Calls: the provided ground-truth trajectory of function calls. When this is empty, it means that the agent needs to call nothing to be scored as correct.

A sample is considered flawed if at least one of the ground-truth function calls has an issue listed below.

## Flaw Categories

1. Format Mismatch - Non-canonical Parameter Values

This occurs when one or more parameter values in a ground-truth function call use non-canonical formats that would cause API errors, even though the semantic meaning is correct. The value might be:

* Concatenated Tokens: Values that should be separated are joined together (e.g., "$2000monthly" instead of "$2000 monthly", "24months" instead of "24 months").
* Malformed Ranges: Incorrectly formatted range specifications (e.g., "1-5" instead of "1 to 5" when the API expects the latter).
* Inconsistent Casing: Values that don't match the expected case format (e.g., "JSON" instead of "json" when the API is case-sensitive).

2. Ambiguous Ground Truth - Incorrect Parameter Value

This occurs when one or more parameter values in a ground-truth function call are not logically justified by the user's prompt, system policy, user context, or the results of previous API calls. The value might be:

* Unjustified/Hallucinated: A value (e.g., a date, a coordinate, an ID) that appears without any grounding context.
* Contradictory: A value that directly contradicts a constraint in the user's prompt or context.
* Misspelled or Incorrectly Identified: A misspelled name or an ID/slug that points to the wrong entity.

3. Format Mismatch - Value Format Mismatch

This occurs when parameter values are semantically correct but formatted incorrectly according to the API schema requirements. The issue might be:

* Type Mismatch: A parameter requires a string but is given a number (e.g., dest_id: 123 instead of dest_id: "123").
* Missing Required Parameters: Required parameters are omitted from the function call.
* Wrong Function Name: The function name is misspelled or incorrect.

4. Optional - Too-easy Single-turn Tasks

This occurs when the task is too simple and can be completed with a single, straightforward function call without requiring complex reasoning or multi-step planning. However, this should only be flagged if the task provides no meaningful evaluation value.

## Crucial Rule: Assume Plausible Context

The ground-truth trajectory represents a realistic tool usage scenario. Your task is to find undeniable flaws in the function calls, not in the overall task design.

* If a parameter value can be reasonably justified by the user's prompt, context, or previous API results, then it is NOT a flaw.
* Flag a sample as flawed ONLY if the function call is impossible to justify or would cause a clear API error.

-----

## Evaluation and Output Format

Carefully analyze the provided sample. Think step-by-step to determine if the ground-truth function calls are logical and if the parameter values are properly justified.

Your final output must be a JSON object with the following structure, with no additional commentary:

```json
{{
  "reasoning": "Provide a clear, step-by-step explanation for your decision. If the sample is flawed, specify what is incorrect and why it contradicts the user's prompt, API schema, or context. If it is not flawed, briefly explain why the sample is valid.",
  "reasoning_summary": "A shorter rationale for your decision. If the sample is not flawed, just mention that it is not flawed. If it is flawed, specify the issue concisely. e.g., The parameter 'dest_id' requires a string but is given a number.",
  "error_category": "<Not Flawed | Format Mismatch - Non-canonical Parameter Values | Ambiguous Ground Truth - Incorrect Parameter Value | Format Mismatch - Value Format Mismatch | Optional - Too-easy Single-turn Tasks>",
  "is_flawed": <true_or_false>

}}
```

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