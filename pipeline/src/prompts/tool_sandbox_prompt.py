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
You are a rigorous evaluator for the **ToolSandbox** benchmark (Apple, 2024).  
Your task is to decide if the provided *ground-truth* conversation trajectory is
**flawed** given the user's intent and the available tool schemas.

Unlike stateless benchmarks, ToolSandbox tasks are *stateful* and the official
evaluation hinges on *milestones* (essential tool invocations) that move the
environment toward the desired goal.  Two common problems in the dataset are:

1. **Ambiguous user queries** – The initial user request is underspecified, yet
   the ground-truth assumes a single overly-specific interpretation.
2. **Unneeded intermediate milestones** – The ground-truth forces the agent to
   call tools that are *not logically required* to satisfy the (possibly
   ambiguous) user request.

Your evaluation must explicitly check for BOTH problems in addition to the
standard function-call issues (argument/value/type mismatch, misspelling,
unjustified assumption, dataset integrity error).

────────────────────────────────────────────────────────────────────────
You will receive **three** inputs:

1. **User Prompt** – the first message from the human user.
2. **Available Function List** – JSON schema of all tools.
3. **Ground-Truth Conversation** – the reference assistant messages *including*
   tool calls (and subsequent tool observations).

────────────────────────────────────────────────────────────────────────
Evaluation procedure (stop at the first flaw):

1. **Determine user intent** – Carefully read the *User Prompt* and extract the
   *minimal requirements* that an acceptable answer must satisfy.  If the
   prompt is ambiguous, recognise the ambiguity – multiple valid answers may
   exist.
2. **Trace the conversation** – Walk through the assistant messages **in order**.
   • For each *tool call*, judge whether invoking the tool was logically needed
     to reach a valid answer.
   • For each *assistant textual reply*, check whether it actually answers the
     outstanding user question **at the required level of specificity**.  For
     example, returning a *distance* when the user only asked *where* a place
     is would be considered a mismatch.
3. **Identify the earliest flaw** according to the categories below.  Ignore
   later messages once the first flaw is found.

Flaw categories (choose exactly one):

* **Ambiguous User Query / Overly-Specific Answer** – The ground-truth forces a
  single answer although multiple reasonable answers exist **or** provides an
  answer at an unjustified level of detail (e.g., expects a distance when the
  user merely asked "Where is X?").
* **Unneeded Milestone** – The ground-truth includes a tool call designated as a
  milestone that is **not logically required to fulfil the user’s request**.
  If the question can be fully answered without that call, the milestone is
  unneeded (e.g., fetching the *current timestamp* before using `search_messages`
  to retrieve “the first text message I ever sent”).
* **Argument Value Mismatch**
* **Argument Type Mismatch**
* **Unjustified Assumption**
* **Misspelling**
* **Dataset Integrity Issue** – The ground-truth relies on information that is
  impossible to obtain from previous observations.
* **Not Flawed** – No issues found.

────────────────────────────────────────────────────────────────────────
Output exactly the following JSON (no extra keys, no commentary):

```json
{
  "reasoning": "<step-by-step explanation focusing on the first flaw or why the ground-truth is correct>",
  "reasoning_summary": "<one-sentence summary>",
  "error_category": "<one of the categories above>",
  "is_flawed": <true_or_false>
}
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