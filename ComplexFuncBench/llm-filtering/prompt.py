prompt = """
You are an expert evaluator of agentic AI systems. 
Your task is to assess how effectively a given benchmark sample measures the capabilities of AI agents.
You will be given the task prompt (user message), list of functions that are available to the agent, and the ground-truth function call.


## Evaluation Criteria

Evaluate each sample on the following dimensions using a 1-5 point scale:

### 1. Task Complexity
- 5 points: High complexity multi-step task requiring dynamic planning
- 3 points: Moderate complexity multi-step task
- 1 point: Very simple single-step task

### 2. Agentic Behavior Requirements
- 5 points: Complex planning, multi-tool usage, and dynamic adaptation essential
- 3 points: Some planning, tool use, or environment interaction needed
- 1 point: Simple input-output, no agentic behavior required

### 3. Correctness of Ground-Truth Function Call
- 5 points: The ground-truth function call trajectory has no logical error and is consistent with the user prompt, available functions, and preceding API call results. Note that preceding API call results are provided in "role": "observation" messages in the conversation.
- 1 point: The ground-truth function call trajectory has significant logical errors or inconsistency, so that no actual agentic model will be able to follow.

### 4. Real-world Relevance
- 5 points: Realistic task reflecting actual use cases
- 3 points: Some realism but limited applicability
- 1 point: Unrealistic artificial scenario

### 5. Scalability & Coverage
- 5 points: Extensible to various domains and situations
- 3 points: Moderate generalizability potential
- 1 point: Only applicable to very specific domains

## Evaluation Process

**Step 1: Sample Analysis**
Thoroughly analyze the content, requirements, and expected solution process of the given benchmark sample.

**Step 2: Dimensional Assessment**
Assign a score of 1-5 for each dimension and provide specific justification for each score.

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
      "score": 1,
      "justification": "The task is too straightforward so that it can be solved with general knowledge and does not require agentic behavior."
    }},
    {{
      "dimension": "Correctness of Ground-Truth Function Call",
      "score": 1,
      "justification": "The user requests for the cheapest flight but the agent reserves a more expensive one."
    }},
    {{
      "dimension": "Real-world Relevance",
      "score": 5,
      "justification": "The task is highly relevant to real-world applications."
    }},
    {{
      "dimension": "Scalability & Coverage",
      "score": 4,
      "justification": "The task can distinguish between different AI capabilities."
    }},
  ]
}}
```

### Task Prompt
{prompt}

### List of Functions That are Available to the Agent
{functions}

### Ground-Truth Function Call
{function_call}

"""

