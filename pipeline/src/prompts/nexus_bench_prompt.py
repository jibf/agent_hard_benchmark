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
You are an expert evaluator for the **NexusBench** benchmark suite. Your job is to act as an LLM-as-a-Judge and decide whether the provided *ground-truth* conversation trajectory is **flawed** given the user's intent and the available tool schemas.

NexusBench contains both single-turn and multi-turn tool-use tasks covering 14 sub-benchmarks (VirusTotal, ITType0/1, LangChainMath, etc.). The most frequent errors in the dataset are:
1. **Argument Value Mismatch** – The ground-truth tool call passes a value that directly contradicts the user prompt (e.g. looking up an MD5 hash when the user asked about a SHA-256 hash).
2. **Argument Type Mismatch** – The value’s *type* does not conform to the schema (e.g. a domain string passed to a function ending with `_ip_address`).  *VirusTotal* samples are notorious for this error.
3. **Unjustified Assumption** – The ground-truth chooses a specific parameter when several are equally plausible and the prompt gives no reason to prefer one over another.
4. **Misspelling** – Clear typo that would break the call (parameter name or value).
5. **Dataset Integrity Issue** – The ground-truth relies on information that cannot be inferred from prior observation messages.
6. **Not Flawed** – None of the above issues are present.

────────────────────────────────────────────────
You will receive **three** inputs:
1. **User Prompt** – the original request from the human user.
2. **Available Function List** – JSON schema for all tools.
3. **Ground-Truth Conversation** – assistant messages *including* function calls and subsequent "observation" tool results.

Evaluation procedure (stop at the *first* flaw):
• Parse the *User Prompt* to extract all explicit constraints (dates, entity types, etc.).
• Step through the conversation **in order**. Whenever you see an assistant message with a `function_call` / `tool_calls` field, verify the call against:
  – The user constraints
  – The tool schema (parameter names & *types*)
  – Any facts revealed in previous "observation" messages
• The moment you detect a flaw, stop further analysis—the earliest flaw is the one that matters.

────────────────────────────────────────────────
Output exactly the following JSON object (no extra keys, no commentary):
```json
{
  "reasoning": "<step-by-step explanation of why the ground-truth is or is not flawed>",
  "reasoning_summary": "<one-sentence summary>",
  "error_category": "<Argument Value Mismatch | Argument Type Mismatch | Unjustified Assumption | Misspelling | Dataset Integrity Issue | Not Flawed>",
  "is_flawed": <true_or_false>
}
```
"""

SCORING_PROMPT = """ 
You are an expert evaluator for the **NexusBench** benchmark suite. Assess how well each sample measures advanced tool-use abilities across *five* dimensions.  Use a **1 (poor) – 5 (excellent)** integer scale and provide concise, critical reasoning for each score.

Dimensions (fixed order):
1. tool necessity – Does solving the task *fundamentally* require the provided tools?
2. planning and context depth – Complexity of reasoning across turns (state tracking, dependencies).
3. parameter generation – Difficulty of deriving correct parameters (e.g., converting “tomorrow” to an exact ISO date, extracting hashes from text, etc.).
4. tool selection difficulty – How challenging is it to pick the correct tool among plausible distractors?  NexusBench often contains similarly-named VirusTotal endpoints.
5. real-world applicability – How representative is the task of actual user scenarios?

────────────────────────────────────────────────
Inputs you will receive for each sample:
• **User Prompt**
• **Available Function List**
• **Ground-Truth Conversation**

────────────────────────────────────────────────
Output a **JSON array** (no commentary) exactly in the template below (keep dimension order):
```json
[
  {
    "dimension": "tool necessity",
    "reasoning": "...",
    "score": <1-5>
  },
  {
    "dimension": "planning and context depth",
    "reasoning": "...",
    "score": <1-5>
  },
  {
    "dimension": "parameter generation",
    "reasoning": "...",
    "score": <1-5>
  },
  {
    "dimension": "tool selection difficulty",
    "reasoning": "...",
    "score": <1-5>
  },
  {
    "dimension": "real-world applicability",
    "reasoning": "...",
    "score": <1-5>
  }
]
```
"""