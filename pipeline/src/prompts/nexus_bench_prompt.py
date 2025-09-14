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

1. **Argument / Parameter Type Mismatch**  
   – A ground-truth function call uses a parameter value whose type clearly contradicts the schema or function name.  
   – Example: Function `vt_get_votes_on_ip_address` called with `ip="example.com"` (a domain, not an IP).  
   – Such mismatches reflect mis-specified tools, not model reasoning, so they must be flagged.

2. Ambiguous or Invalid Ground Truth
   - Example: User query '1+2+3+4+5' and the ground-truth result is '19.8', which is correct according to the given tools, but the system prompt and initial query and system instructionsdo not provide enough information to justify the need to override the model's default (+) calculationfunctionality and instead always use the special add() function.
   – Such mismatches reflect overriden model functionality, not model reasoning, so they must be flagged.

3. Ambiguous or Poorly Written User Query
   - Example: User query 'E. coli doubles every 20m, 120m from 5 cells'. It is unclear exactly what the user is asking for.
   - Such mismatches reflect ambiguous user queries, not model reasoning, so they must be flagged.
   - However, ONLY flag this if the user query cannot be reasonably inferred from the available tools and system instructions.

-----

## Evaluation and Output Format
Think step-by-step.  Output **exactly** the JSON object below—no extra keys or commentary:

```json
{{
  "reasoning": "Provide a clear, step-by-step justification.  If flawed, specify the first flaw and why it violates the prompt, schema, or context.",
  "reasoning_summary": "One-sentence summary of the verdict.",
  "error_category": "<Argument Type Mismatch | Ambiguous or Invalid Ground Truth Value | Ambiguous or Poorly Written User Query | Not Flawed>",
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
