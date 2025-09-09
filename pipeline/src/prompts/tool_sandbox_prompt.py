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
You are a rigorous evaluator for the **ToolSandbox** benchmark (Apple, 2024), a stateful, multi-turn tool-use benchmark.  
Your task is to decide if the provided *ground-truth* conversation trajectory is
**flawed** given the user's intent and the available tool schemas.

ToolSandbox specifics to keep in mind:
• Tasks are evaluated via **milestones** (essential actions) and **minefields** (undesirable actions), not just final text.
• Scenarios are **stateful**: tools can mutate world/DB state and later steps may depend on earlier ones.

────────────────────────────────────────────────────────────────────────
You will receive **three** inputs:

1. **User Prompt** – the first instruction from the human user.
2. **Available Function List** – JSON schema of all tools.
3. **Ground-Truth Conversation** – the reference assistant messages *including*
   tool calls (and subsequent tool observations).

────────────────────────────────────────────────────────────────────────
Your task is to decide whether the **ground-truth (GT) milestone trajectory** for a scenario is **flawed**, focusing these issues:

1. **Ambiguous User Query / Over-Specific Ground Truth**  
   – The user’s request is vague or underspecified
   – The GT enforces an answer at a **more specific level** than requested (e.g., requires a numeric distance when the user just asked *“Where is X?”*).
   – In such cases, multiple valid answers exist; enforcing one path is unfair → mark as flawed.

2. **Unneeded Milestone**  
   – The GT includes intermediate tool calls designated as milestones that are **not logically necessary** to answer the query.  
   – Example: requiring `get_current_timestamp` before retrieving the first message, even though the query *“What’s my first text?”* can be answered directly with `search_messages`.  
   – Each milestone must be **indispensable** for reaching the answer. If not, mark as flawed.

────────────────────────────────────────────────────────────
RULES
• Always check: Does the **level of detail in the GT final answer** match exactly what the user asked for? If it overshoots (too detailed), it is flawed.  
• Always check: Could the task be solved **without one or more of the required milestones**? If yes, it is flawed.  
• Ignore minor wording differences or plausible missing conversation turns.  
• Stop at the **earliest undeniable flaw**.

────────────────────────────────────────────────────────────
FLAW CATEGORIES (choose one)
• Ambiguous User Query / Over-Specific Ground Truth  
• Unneeded Milestone  
• Not Flawed

────────────────────────────────────────────────────────────────────────
Output exactly the following JSON (no extra keys, no commentary):

```json
{{
  "reasoning": "<step-by-step explanation focusing on the first flaw or why the ground-truth is correct>",
  "reasoning_summary": "<one-sentence summary>",
  "error_category": "<one of the categories above>",
  "is_flawed": <true_or_false>
}}
```

## Target Sample

### User Prompt
```
{instruction}
```

### Available Function List
```json
{available_function_list}
```

### Ground-Truth Conversation
*Messages with `"role": "observation"` are tool outputs for the function call immediately before them.*
```json
{gt_conv_traj}
```

### Expected Final Assistant Message
```
{expected_output}
```
"""


SCORING_PROMPT = """
You are an expert evaluator for the **ToolSandbox** benchmark.  Assess how well
each sample measures advanced tool-use capabilities along five dimensions.  Use
the scale **1 (poor) – 5 (excellent)**.  Provide concise, critical reasoning for
each score.

Dimensions (fixed order):

1. tool necessity – Does solving the task *fundamentally require* the provided
   tools?
2. planning and context depth – Complexity of state tracking & dependency
   reasoning across turns.
3. parameter generation – Difficulty of deriving correct parameters from
   context (dates, IDs, dynamic values, etc.).
4. tool selection difficulty – How challenging is it to pick the correct tool
   amidst plausible distractors?
5. real-world applicability – How representative is the task of real user
   scenarios?

───────────
You will receive:
* **User Prompt**
* **Available Function List**
* **Ground-Truth Conversation** (including milestones)

───────────
Output a JSON **array** (no extra commentary) following this template:

```
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
"""