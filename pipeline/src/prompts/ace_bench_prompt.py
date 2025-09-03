# ACEBench-specific prompts for LLM-as-a-judge assessment
# Based on the detailed analysis of 4 main issue categories:
# 1. Parameter Value Error - Wrong, non-canonical, inconsistent values
# 2. Incorrect Parameter Value - Contradicts context or gold labels
# 3. Addition of Unnecessary Parameter - Hallucinated extra parameters
# 4. Value Error - Semantically correct but formatted incorrectly

# FILTERING: 
# {{
#   "reasoning": "Provide a clear, step-by-step explanation for your decision. If the sample is flawed, specify which issue category it belongs to and why. If it is not flawed, briefly explain why the sample is valid for evaluation.",
#   "reasoning_summary": "A shorter rationale for your decision. If the sample is not flawed, just mention that it is not flawed. If the sample is flawed, specify the issue category concisely.",
#   "error_category": "<Parameter Value Error | Incorrect Parameter Value | Addition of Unnecessary Parameter | Value Error | Not Flawed>",
#   "is_flawed": <true_or_false>,
# }}

# SCORING: 
# [
#     {{
#     "dimension": "parameter accuracy",
#     "reasoning": "Evaluate how accurately the model generates parameter values. Consider whether values are semantically correct, properly formatted, and consistent with the expected schema.",
#     "score": <1-5>
#     }},
#     {{
#     "dimension": "function call correctness",
#     "reasoning": "Assess whether the model makes the correct function calls with appropriate parameters. Consider tool selection, parameter completeness, and adherence to API specifications.",
#     "score": <1-5>
#     }},
#     {{
#     "dimension": "error handling",
#     "reasoning": "Evaluate how well the model handles imperfect instructions, missing parameters, or invalid requests. Consider robustness and appropriate error responses.",
#     "score": <1-5>
#     }},
#     {{
#     "dimension": "multi-turn reasoning",
#     "reasoning": "For multi-turn tasks, assess the model's ability to maintain context, track state changes, and make coherent sequences of function calls.",
#     "score": <1-5>
#     }},
#     {{
#     "dimension": "real-world applicability",
#     "reasoning": "Evaluate how realistic and practical the model's responses are for real-world tool usage scenarios.",
#     "score": <1-5>
#     }}
# ]

FILTERING_PROMPT = """You are an expert evaluator for ACEBench, a benchmark for evaluating LLMs' ability to perform tool usage (function calling) across scenarios of increasing complexity and realism.

Your task is to identify samples that have fundamental flaws that would make them unsuitable for evaluation. ACEBench has identified 4 main issue categories:

1. **Parameter Value Error**: Wrong, non-canonical, or inconsistent values (e.g., "$2000monthly" instead of "$2000 monthly", "24months" instead of "24 months")
2. **Incorrect Parameter Value**: Values that contradict context or gold labels (e.g., wrong dates, species names, or IDs that cannot be normalized)
3. **Addition of Unnecessary Parameter**: Hallucinated extra parameters outside the schema (e.g., adding "provide_indoor_options=true" when not requested)
4. **Value Error**: Semantically correct but formatted incorrectly (e.g., concatenated tokens, malformed ranges, inconsistent casing)

**Instructions**: Analyze the given sample and determine if it has any of these fundamental flaws. Focus on task-level issues that would make the sample unreliable for evaluation, not on individual model response quality.

**Sample Information**:
- Task: {task_name}
- Instruction: {instruction}
- Available Functions: {available_function_list}
- Ground Truth: {gt_conv_traj}
- Metadata: {meta}

**Evaluation Criteria**:
- **Not Flawed**: Sample has clear, unambiguous instructions, proper function schemas, and valid evaluation criteria
- **Parameter Value Error**: Sample contains malformed parameter values that cannot be reasonably normalized
- **Incorrect Parameter Value**: Sample has values that contradict the given context or are semantically wrong
- **Addition of Unnecessary Parameter**: Sample evaluation depends on dropping extra parameters that are semantically neutral
- **Value Error**: Sample relies on formatting rather than semantic correctness for evaluation

**Output Format**: Provide your analysis in the following JSON format:
{{
  "reasoning": "Provide a clear, step-by-step explanation for your decision. If the sample is flawed, specify which issue category it belongs to and why. If it is not flawed, briefly explain why the sample is valid for evaluation.",
  "reasoning_summary": "A shorter rationale for your decision. If the sample is not flawed, just mention that it is not flawed. If the sample is flawed, specify the issue category concisely.",
  "error_category": "<Parameter Value Error | Incorrect Parameter Value | Addition of Unnecessary Parameter | Value Error | Not Flawed>",
  "is_flawed": <true_or_false>
}}

Remember: Only flag samples with fundamental design flaws that would make evaluation unreliable. Minor formatting issues or model response variations should not be considered flaws."""

SCORING_PROMPT = """You are an expert evaluator for ACEBench, a benchmark for evaluating LLMs' ability to perform tool usage (function calling) across scenarios of increasing complexity and realism.

Your task is to score the given sample across 5 dimensions, each rated from 1-5 where:
- 1: Poor performance, fundamental errors
- 2: Below average, significant issues
- 3: Average performance, some errors but generally functional
- 4: Good performance, minor issues
- 5: Excellent performance, robust and reliable

**Sample Information**:
- Task: {task_name}
- Instruction: {instruction}
- Available Functions: {available_function_list}
- Ground Truth: {gt_conv_traj}
- Metadata: {meta}

**Scoring Dimensions**:

1. **Parameter Accuracy** (1-5): How accurately does the model generate parameter values? Consider semantic correctness, proper formatting, and consistency with expected schema.

2. **Function Call Correctness** (1-5): Does the model make the correct function calls with appropriate parameters? Consider tool selection, parameter completeness, and adherence to API specifications.

3. **Error Handling** (1-5): How well does the model handle imperfect instructions, missing parameters, or invalid requests? Consider robustness and appropriate error responses.

4. **Multi-turn Reasoning** (1-5): For multi-turn tasks, how well does the model maintain context, track state changes, and make coherent sequences of function calls?

5. **Real-world Applicability** (1-5): How realistic and practical are the model's responses for real-world tool usage scenarios?

**Output Format**: Provide your evaluation in the following JSON format:
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