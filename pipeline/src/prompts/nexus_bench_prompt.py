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
Your task is to decide whether the *ground-truth* answer of a benchmark task is **flawed** such that a correct agent would be unfairly penalized.

You will be provided with the following context for each sample:
* **System Prompts** – the system-level instructions supplied to the model.
* **Query** – the user’s original instruction.
* **Expected Output** – the benchmark’s reference answer.
* **Benchmark Name** – the specific NexusBench sub-benchmark the sample belongs to.
* **Tool Schemas** – complete JSON definitions for every callable tool.

A sample is flawed if at least one ground-truth function call violates one of the criteria below.

1. **Invalid Ground Truth**
   - For example, if the query is 'Cape Town, how's the time and weather? Any weather stations 'round?' and the ground truth is 'get_latitude_longitude('Cape Town'); find_nearby_stations(get_latitude_longitude('Cape Town')); get_current_time_at_location(get_latitude_longitude('Cape Town')); get_timezone(get_latitude_longitude('Cape Town'))', then the ground truth is invalid because the query is asking for the time and weather of Cape Town, but the ground truth is asking for the latitude, longitude, and timezone of Cape Town and does not make a weather query, even though the get_hourly_observation tool is available which will indicate the weather.
   - In your reasoning, you should mention the query and the ground truth, and explain exactly why the ground truth is invalid and why it does not answer all parts of the user query.
   - In these cases, ONLY flag this if the ground truth does not sufficiently answer the user query. If the ground truth does sufficiently answer the user query, then it is not invalid. If you are unsure, flag it as valid.

2. **Argument / Parameter Type Mismatch Due to Problematic Function Schemas**
   – A ground-truth function call uses a parameter value whose type clearly contradicts the schema or function name.  
   – Example: Function `vt_get_votes_on_ip_address` called with `ip="example.com"` (example.com is a domain, not an IP address).  
   – Such mismatches reflect mis-specified tools, not model reasoning, so they must be flagged.
   - NOTE: All functions are Python functions, which do not need the parameter names to be passed in the queries. If the ground truth function call passes by argument without the parameter name, then it is not a mismatch. If you are unsure, flag it as valid.

2. **Ambiguous or Poorly Written User Query**
   - Example: User query 'E. coli doubles every 20m, 120m from 5 cells' or 'what's the clock? in London, Paris and Kiev, dude?' or 'Any big weather changes in last hour in NY, NY, dude?'. It is unclear exactly what the user is asking for.
   - Such mismatches reflect ambiguous user queries, not model reasoning, so they must be flagged.
   - However, ONLY flag this if the user query cannot be reasonably inferred from the available tools and system instructions. If you are unsure, flag it as valid.

NOTE: For TypeWriter tasks (LangChainTypeWriterHard, LangChainMultitoolTypeWriterHard), the ground truth is often the final output of the agent, not the result of a single function call. So it is not a mismatch if the ground truth is the query repeated IF the query can be typed with the available tools.
For TMIHallucination, the task involves a remapping of words to other words, so for example, if the instruction specifies that "Wet" maps to "Hot" and "Hot" maps to "Wet", and the user query is "Raindrops are wet" and the ground truth is match_values(["Hot")]. This is valid because the ground truth correctly remaps the user query to the correct value.
Also, the defined functions can exist in an alternate universe and the query could be valid. For example, if the question asks to add 5 and 7 and the ground truth is -2, and defines the add() function to return the difference of two numbers, then the ground-truth is valid because the provided function call (add(5, 7)) is valid and would return -2.
-----

## Evaluation and Output Format
Think step-by-step.  Output **exactly** the JSON object below—no extra keys or commentary:

```json
{{
  "reasoning": "Provide a clear, step-by-step justification.  If flawed, specify the first flaw and why it violates the prompt, schema, or context.",
  "reasoning_summary": "One-sentence summary of the verdict.",
  "error_category": "<Invalid Ground Truth | Argument / Parameter Type Mismatch Due to Problematic Function Schemas | Ambiguous or Poorly Written User Query | Not Flawed>",
  "is_flawed": <true_or_false>
}}
```

### System Prompts & Initial Instructions Sent to the Model:
```
{system_prompts}
```

## Target Sample

### Query:
```
{instruction}
```

### Expected Output:
```
{reference}
```

### Benchmark Name:
```
{benchmark_name}
```

### Available Tools (JSON Schemas):
```
{tool_definitions}
```
"""


# ---------------------------------------------------------------------------
#                             SCORING PROMPT
# ---------------------------------------------------------------------------

SCORING_PROMPT = ""