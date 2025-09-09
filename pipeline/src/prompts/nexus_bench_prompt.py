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
You are an expert evaluator for **NexusBench**, a benchmark designed to assess precise tool-use across diverse tool-use tasks.
Your task is to decide whether the *ground-truth* conversation trajectory of a sample is **flawed** such that a correct agent would be penalised.

You will be provided with:
* **User Prompt** – the original request from the human.
* **Available Function List** – JSON schema of all tools.
* **Ground-Truth Conversation** – assistant messages including function calls followed by their `"role": "observation"` results.

A sample is flawed if at least one ground-truth function call violates one of the criteria below.

1. **Argument / Parameter Type Mismatch**  
   – A GT function call uses a parameter value whose type clearly contradicts the schema or function name.  
   – Example: Function `vt_get_votes_on_ip_address` called with `ip="example.com"` (a domain, not an IP).  
   – Such mismatches reflect mis-specified tools, not model reasoning, so they must be flagged.

2. **Invalid or Irrelevant Ground Truth Value**  
   – The GT tool call or value is incoherent with the user’s request and does not actually address the prompt.  
   – Example: User asks about storm forecasting, but GT call is `match_values(['Female'])`, which is unrelated.  
   – If the GT provides values or tool outputs that do not semantically connect to the prompt, mark as flawed.

-----

## Evaluation and Output Format
Think step-by-step.  Output **exactly** the JSON object below—no extra keys or commentary:

```json
{{
  "reasoning": "Provide a clear, step-by-step justification.  If flawed, specify the first flaw and why it violates the prompt, schema, or context.",
  "reasoning_summary": "One-sentence summary of the verdict.",
  "error_category": "<Argument Value Mismatch | Argument Type Mismatch | Unjustified Assumption | Misspelling | Dataset Integrity Issue | Not Flawed>",
  "is_flawed": <true_or_false>
}}
```

## Target Sample

### User Prompt
```
{user_prompt}
```

### List of available functions and their schema
```json
{available_function_list}
```

### Ground-Truth Conversation
*Messages with `"role": "observation"` are the results of the function call immediately before them.*
```json
{conversations}
```
"""


# ---------------------------------------------------------------------------
#                             SCORING PROMPT
# ---------------------------------------------------------------------------

SCORING_PROMPT = """
You are an expert evaluator for **NexusBench**.  Assess how *difficult* each sample is for an intelligent tool-using agent across the five dimensions below.  Use an **integer 1-5** scale and provide concise, critical reasoning for every score.

Dimensions (fixed order):
1. tool necessity – Does solving the task fundamentally require the provided tools?
2. planning and context depth – Complexity of reasoning across turns (state tracking, dependencies).
3. parameter generation – Difficulty of deriving correct parameters (e.g., converting “tomorrow” to ISO date, extracting hashes, etc.).
4. tool selection difficulty – How challenging is it to pick the correct tool among plausible distractors?  (VirusTotal endpoints are notorious here.)
5. real-world applicability – How representative is the task of genuine user scenarios?

-----
You will receive:
• **User Prompt**
• **Available Function List**
• **Ground-Truth Conversation**

-----
Output a JSON **array** (no commentary) exactly in this template and order:
```json
[
  {{
    "dimension": "tool necessity",
    "reasoning": "...",
    "score": <1-5>
  }},
  {{
    "dimension": "planning and context depth",
    "reasoning": "...",
    "score": <1-5>
  }},
  {{
    "dimension": "parameter generation",
    "reasoning": "...",
    "score": <1-5>
  }},
  {{
    "dimension": "tool selection difficulty",
    "reasoning": "...",
    "score": <1-5>
  }},
  {{
    "dimension": "real-world applicability",
    "reasoning": "...",
    "score": <1-5>
  }}
]
```

-----

### User Prompt
```
{user_prompt}
```

### Available Function List
```json
{available_function_list}
```

### Ground-Truth Conversation
```json
{conversations}
```
"""