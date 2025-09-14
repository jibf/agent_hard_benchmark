
FILTERING_PROMPT = """You are an expert evaluator for MultiChallenge, a benchmark designed to assess an agent's ability to maintain context and coherence across multi-turn conversations.

Your task is to identify if a sample in the benchmark has a fundamental flaw that would make it an unreliable sample for evaluation.

You will be provided with the following information:
* User Scenario: the final user prompt in the conversation
* Available Function List: a list of functions available for the agents and their schema
* Original Conversation: the complete multi-turn conversation flow
* Evaluation Criteria: the target question and pass criteria for assessment

A sample is considered flawed if at least one of the following issues is present:

## Flaw Categories

1. Ambiguous Ground Truth - Memory Failure

This occurs when the conversation has fundamental structural issues that make evaluation impossible:

* Missing Conversation: The conversation is completely empty or missing entirely
* Broken Turn Structure: The conversation flow has abrupt context shifts or disconnected turns that break logical progression
* Incomplete Context: Critical information needed for evaluation is missing from the conversation

2. Bad Evaluation - Instruction Issues

This occurs when the evaluation criteria are vague, subjective, or impossible to apply objectively:

* Vague Target Question: The target question is unclear or allows for multiple valid interpretations
* Subjective Pass Criteria: The pass criteria depend on personal preferences rather than objective measures
* Impossible Evaluation: The target question cannot be answered based on the conversation content

3. Bad Evaluation - Self-Contradiction

This occurs when the conversation contains embedded conflicts or contradictory information:

* Embedded Conflicts: The conversation contains inherent contradictions that cannot be resolved
* Factual Contradictions: The conversation contains demonstrably false or contradictory information
* Competing Requirements: Multiple valid but mutually exclusive approaches are presented without clear prioritization

4. Ambiguous Ground Truth - Ungrounded Versions

This occurs when the evaluation lacks clear criteria or reference points:

* Missing Evaluation Context: No clear baseline or reference point for evaluation
* Unclear Success Criteria: The pass criteria are ambiguous about what constitutes success
* Competing Interpretations: Multiple valid interpretations of the target question without clear criteria

## Crucial Rule: Focus on Evaluation Feasibility

MultiChallenge is designed to test multi-turn conversation capabilities. Your task is to identify samples where evaluation is fundamentally impossible, not to judge conversation quality.

* If the conversation provides sufficient context to answer the target question objectively, then it is NOT flawed
* Flag a sample as flawed ONLY if the evaluation criteria cannot be applied objectively or the conversation structure makes assessment impossible

-----

## Evaluation and Output Format

Carefully analyze the provided sample. Think step-by-step to determine if the ground-truth conversation is well-designed and if the evaluation criteria are clear and objective.

Your final output must be a JSON object with the following structure, with no additional commentary:

```json
{{
  "reasoning": "Provide a clear, step-by-step explanation for your decision. If the sample is flawed, specify what is incorrect and why it makes evaluation unreliable. If it is not flawed, briefly explain why the sample is valid.",
  "reasoning_summary": "A shorter rationale for your decision. If the sample is not flawed, just mention that it is not flawed. If it is flawed, specify the issue concisely. e.g., The conversation contains conflicting requirements that cannot be resolved simultaneously.",
  "error_category": "<Not Flawed | Ambiguous Ground Truth - Memory Failure | Bad Evaluation - Instruction Issues | Bad Evaluation - Self-Contradiction | Ambiguous Ground Truth - Ungrounded Versions>",
  "is_flawed": <true_or_false>

}}
```

## Target Sample

### User Scenario (Final User Prompt)

```
{instruction}
```

### Available Functions

```json
{available_function_list}
```

### Original Conversation

```json
{original_conversation}
```

### Evaluation Criteria

```json
{evaluation_criteria}
```

### Metadata

```json
{meta}
```"""

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