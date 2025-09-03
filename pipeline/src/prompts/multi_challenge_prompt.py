# MultiChallenge-specific prompts for LLM-as-a-judge assessment
# Based on the detailed analysis of 4 main issue categories:
# 1. Memory Failure - Missing facts, broken turn structure
# 2. Instruction Violation - Vague, subjective, conflicting instructions
# 3. Self-Contradiction - Imprecise coherence, embedded conflicts
# 4. Version Confusion - Missing authoritative state, unclear edits

# FILTERING: 
# {{
#   "reasoning": "Provide a clear, step-by-step explanation for your decision. If the sample is flawed, specify which issue category it belongs to and why. If it is not flawed, briefly explain why the sample is valid for evaluation.",
#   "reasoning_summary": "A shorter rationale for your decision. If the sample is not flawed, just mention that it is not flawed. If the sample is flawed, specify the issue category concisely.",
#   "error_category": "<Memory Failure | Instruction Violation | Self-Contradiction | Version Confusion | Not Flawed>",
#   "is_flawed": <true_or_false>,
# }}

# SCORING: 
# [
#     {{
#     "dimension": "instruction retention",
#     "reasoning": "Evaluate how well the model maintains first-turn constraints (e.g., format, scope, safety) across the multi-turn dialogue.",
#     "score": <1-5>
#     }},
#     {{
#     "dimension": "inference memory",
#     "reasoning": "Assess the model's ability to recall and use user details scattered earlier when only implicitly required later.",
#     "score": <1-5>
#     }},
#     {{
#     "dimension": "reliable versioned editing",
#     "reasoning": "Evaluate how accurately the model applies iterative edits to the correct prior version without hallucinations.",
#     "score": <1-5>
#     }},
#     {{
#     "dimension": "self-coherence",
#     "reasoning": "Assess the model's consistency with its own earlier outputs, avoiding reversals or contradictions.",
#     "score": <1-5>
#     }},
#     {{
#     "dimension": "conversation quality",
#     "reasoning": "Evaluate the overall naturalness, coherence, and effectiveness of the multi-turn conversation.",
#     "score": <1-5>
#     }}
# ]

FILTERING_PROMPT = """You are an expert evaluator for MultiChallenge, a benchmark for evaluating LLMs on realistic multi-turn conversations (up to 10 turns) that test four fundamental capability clusters: Instruction Retention (IR), Inference Memory (IM), Reliable Versioned Editing (RVE), and Self-Coherence (SC).

Your task is to identify samples that have fundamental flaws that would make them unsuitable for evaluation. MultiChallenge has identified 4 main issue categories:

1. **Memory Failure**: Missing facts, broken turn structure, vague thresholds without numeric data, abrupt context shifts that make it impossible for models to maintain context
2. **Instruction Violation**: Vague, subjective, or conflicting instructions (e.g., "no bold anywhere" without clear definition, unclear scope/priorities)
3. **Self-Contradiction**: Imprecise coherence criteria, embedded conflicts, contradictory facts, competing requirements that pressure models into contradictions
4. **Version Confusion**: Missing authoritative state, unclear change specifications, competing final versions, underspecified edit intent

**Instructions**: Analyze the given sample and determine if it has any of these fundamental flaws. Focus on task-level issues that would make the sample unreliable for evaluation, not on individual model response quality.

**Sample Information**:
- Instruction: {instruction}
- Available Functions: {available_function_list}
- Ground Truth: {gt_conv_traj}
- Metadata: {meta}

**Evaluation Criteria**:
- **Not Flawed**: Sample has clear, unambiguous instructions, complete factual information, coherent turn structure, and valid evaluation criteria
- **Memory Failure**: Sample is missing essential facts, has broken turn structure, or contains vague thresholds that make context maintenance impossible
- **Instruction Violation**: Sample has vague, subjective, or conflicting instructions that cannot be objectively evaluated
- **Self-Contradiction**: Sample has embedded conflicts, contradictory facts, or competing requirements that force models into contradictions
- **Version Confusion**: Sample lacks clear authoritative state or has unclear change specifications that make evaluation ambiguous

**Output Format**: Provide your analysis in the following JSON format:
{{
  "reasoning": "Provide a clear, step-by-step explanation for your decision. If the sample is flawed, specify which issue category it belongs to and why. If it is not flawed, briefly explain why the sample is valid for evaluation.",
  "reasoning_summary": "A shorter rationale for your decision. If the sample is not flawed, just mention that it is not flawed. If the sample is flawed, specify the issue category concisely.",
  "error_category": "<Memory Failure | Instruction Violation | Self-Contradiction | Version Confusion | Not Flawed>",
  "is_flawed": <true_or_false>
}}

Remember: Only flag samples with fundamental design flaws that would make evaluation unreliable. Minor formatting issues or model response variations should not be considered flaws."""

SCORING_PROMPT = """You are an expert evaluator for MultiChallenge, a benchmark for evaluating LLMs on realistic multi-turn conversations (up to 10 turns) that test four fundamental capability clusters: Instruction Retention (IR), Inference Memory (IM), Reliable Versioned Editing (RVE), and Self-Coherence (SC).

Your task is to score the given sample across 5 dimensions, each rated from 1-5 where:
- 1: Poor performance, fundamental errors
- 2: Below average, significant issues
- 3: Average performance, some errors but generally functional
- 4: Good performance, minor issues
- 5: Excellent performance, robust and reliable

**Sample Information**:
- Instruction: {instruction}
- Available Functions: {available_function_list}
- Ground Truth: {gt_conv_traj}
- Metadata: {meta}

**Scoring Dimensions**:

1. **Instruction Retention** (1-5): How well does the model maintain first-turn constraints (e.g., format, scope, safety) across the multi-turn dialogue? Consider persistence of requirements and consistency of adherence.

2. **Inference Memory** (1-5): How effectively does the model recall and use user details scattered earlier when only implicitly required later? Consider context awareness and information retrieval.

3. **Reliable Versioned Editing** (1-5): How accurately does the model apply iterative edits to the correct prior version without hallucinations? Consider edit accuracy and version management.

4. **Self-Coherence** (1-5): How consistent is the model with its own earlier outputs? Consider avoidance of reversals, contradictions, and maintenance of stated facts.

5. **Conversation Quality** (1-5): How natural, coherent, and effective is the overall multi-turn conversation? Consider flow, relevance, and user experience.

**Output Format**: Provide your evaluation in the following JSON format:
[
  {{
    "dimension": "instruction retention",
    "reasoning": "Evaluate how well the model maintains first-turn constraints (e.g., format, scope, safety) across the multi-turn dialogue.",
    "score": <1-5>
  }},
  {{
    "dimension": "inference memory",
    "reasoning": "Assess the model's ability to recall and use user details scattered earlier when only implicitly required later.",
    "score": <1-5>
  }},
  {{
    "dimension": "reliable versioned editing",
    "reasoning": "Evaluate how accurately the model applies iterative edits to the correct prior version without hallucinations.",
    "score": <1-5>
  }},
  {{
    "dimension": "self-coherence",
    "reasoning": "Assess the model's consistency with its own earlier outputs, avoiding reversals or contradictions.",
    "score": <1-5>
  }},
  {{
    "dimension": "conversation quality",
    "reasoning": "Evaluate the overall naturalness, coherence, and effectiveness of the multi-turn conversation.",
    "score": <1-5>
  }}
]

Provide detailed reasoning for each score, considering the specific context and requirements of the task. Focus on the model's performance in maintaining context, following instructions, and producing coherent multi-turn conversations."""