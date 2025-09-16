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

# IN BFCL, agent system prompt includes available_function_list

FILTERING_PROMPT = """
You are an expert evaluator for evaluating a function-calling benchmark named BFCL.
Your task is to determine if the provided ground-truth function call(s) are flawed.

You will be given:
* Instruction: The description of the task given to the agent. This can be either single-turn or multi-turn.
* Agent System Prompt: the system prompt used to initialize the agent model. This may contain a specific instruction on the answer style, domain-specific policy that the agent needs to follow, a list of available functions and their schema (in JSON format), etc.
* Ground-Truth Function Call Trajectory: the provided ground-truth trajectory of function calls. When this is empty or None, it means that the agent needs to call nothing to be scored as correct.

## Definition of a flawed ground truth (any one is sufficient):

1. Argument Type Mismatch: a parameter value violates its declared type in the function schema.
2. Argument Value Mismatch: the value contradicts, or is not justified by, the user instruction (e.g., wrong id, date, or enum value).
3. Missing/Unexpected Argument: a required parameter is missing, or an argument not declared in the schema is present.
4. Invalid Function Name: the function name is not among the available functions.
5. Irrelevant Call: the ground truth includes a function call when it should be empty (e.g., irrelevance turns, miss_param/miss_func turns in multi-turn).
6. Misspelling that changes meaning (e.g., incorrect enum name or schema key).

## Important guidance:

- Judge only on undeniable flaws. Do not penalize harmless formatting differences.
- If multiple function calls are shown, evaluate them collectively. For multi-turn, evaluate the sequence across turns; turns with empty ground-truth imply no valid call should be made in that turn.
- If a plausible reading of the instruction would justify the value, do not flag it as flawed.

## Output Format

Output a single JSON object with exactly these fields and no extra text:
```json
{{
  "reasoning": "Step-by-step rationale. Reference specific args/fields if flawed.",
  "reasoning_summary": "Concise reason. If not flawed, say Not Flawed. For multi-turn, mention if any turn violates irrelevance or parameter requirements.",
  "error_category": "<Not Flawed | Argument Value Mismatch | Argument Type Mismatch | Missing/Unexpected Argument | Invalid Function Name | Irrelevant Call | Misspelling>",
  "is_flawed": <true or false>
}}
```

## Sample to be evaluated

### Category 
* category: {category}
* subcategory: {subcategory}

### Instruction

{instruction}

### Agent System Prompt

{system_prompt}

- Ground-Truth Function Call(s):
```json
{ground_truth}
```
"""

# Optional: scoring template (only used if you run with scoring enabled)
SCORING_PROMPT = """
You are scoring the task difficulty of a BFCL single-turn sample.
Consider: clarity of mapping from instruction to tool(s), parameter complexity, and ambiguity.
Return a JSON array of objects with fields: dimension, reasoning, score (1-5).

Example output:
[
  {{"dimension": "tool selection difficulty", "reasoning": "…", "score": 3}},
  {{"dimension": "parameter complexity", "reasoning": "…", "score": 3}}
]
"""
