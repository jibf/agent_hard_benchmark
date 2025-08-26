prompt = """
You are an expert evaluator of agentic AI systems. Your task is to assess how effectively a given benchmark sample measures the capabilities of AI agents.

## Evaluation Criteria

Evaluate each sample on the following 7 dimensions using a 1-5 point scale:

### 1. Task Complexity
- 1 point: Very simple single-step task
- 3 points: Moderate complexity multi-step task
- 5 points: High complexity multi-step task requiring dynamic planning

### 2. Agentic Behavior Requirements
- 1 point: Simple input-output, no agentic behavior required
- 3 points: Some planning, tool use, or environment interaction needed
- 5 points: Complex planning, multi-tool usage, and dynamic adaptation essential

### 3. Evaluation Clarity
- 1 point: Success/failure judgment is ambiguous
- 3 points: Clear evaluation criteria with some subjective elements
- 5 points: Objective and clear evaluation criteria, automatable

### 4. Real-world Relevance
- 1 point: Unrealistic artificial scenario
- 3 points: Some realism but limited applicability
- 5 points: Realistic task reflecting actual use cases

### 5. Discriminative Power
- 1 point: Most AI systems expected to perform similarly
- 3 points: Can distinguish between some systems
- 5 points: Clearly differentiates AI systems of various capability levels

### 6. Scalability & Coverage
- 1 point: Only applicable to very specific domains
- 3 points: Moderate generalizability potential
- 5 points: Extensible to various domains and situations

### 7. Bias & Fairness
- 1 point: Heavily biased toward specific cultures, languages, or knowledge
- 3 points: Some bias elements present but manageable
- 5 points: Minimizes bias with fair evaluation

## Evaluation Process

**Step 1: Sample Analysis**
Thoroughly analyze the content, requirements, and expected solution process of the given benchmark sample.

**Step 2: Dimensional Assessment**
Assign a score of 1-5 for each of the 7 dimensions and provide specific justification for each score.

**Step 3: Comprehensive Evaluation**
- Calculate total score by summing up the dimensional scores
- Summarize key strengths and weaknesses
- Provide improvement recommendations

## Output Format
You should output the score for each dialogue history and corresponding response in JSON format with following keys:
- score (type: int): overall score. 
- dimensional scores (type: array): arrays of dictionaries, each containing:
  - dimension (type: string): name of the dimension
  - score (type: int): score for the dimension
  - justification (type: string): explanation for the score

For example, the output for a sample could look like this:

```json
{{
  "score": 30,
  "dimensional_scores": [
    {{
      "dimension": "Task Complexity",
      "score": 4,
      "justification": "The task requires multiple steps and some degree of planning."
    }},
    {{
      "dimension": "Agentic Behavior Requirements",
      "score": 5,
      "justification": "The task necessitates complex planning and dynamic adaptation."
    }},
    {{
      "dimension": "Evaluation Clarity",
      "score": 4,
      "justification": "The evaluation criteria are mostly clear, with minor subjective elements."
    }},
    {{
      "dimension": "Real-world Relevance",
      "score": 5,
      "justification": "The task is highly relevant to real-world applications."
    }},
    {{
      "dimension": "Discriminative Power",
      "score": 4,
      "justification": "The task can distinguish between different AI capabilities."
    }},
    {{
      "dimension": "Scalability & Coverage",
      "score": 3,
      "justification": "The task has moderate generalizability potential."
    }},
    {{
      "dimension": "Bias & Fairness",
      "score": 5,
      "justification": "The task minimizes bias and ensures fair evaluation."
    }}
  ]
}}
```

### Task Prompt
{prompt}

### Ground-Truth Function Call
{function_call}
"""

## ground truth에 오류가 있는 것도 LLM-as-a-judge로 판정 