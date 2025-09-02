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



FILTERING_PROMPT = ""
SCORING_PROMPT = ""