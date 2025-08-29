prompt="""
You are an expert AI assistant specializing in the meticulous evaluation of function-calling benchmarks. Your task is to assess how effectively a given benchmark sample measures the capabilities of AI agents.

This evaluation is for the {benchmark_type} benchmark.

You will be given three pieces of information:

1.  User Prompt: This could be either:
    - For ComplexFuncBench: The original request from the user.
    - For Tau-bench: The instruction given to the AI model (acting as a customer service representative). In this case, the "user prompt" actually describes the persona and scenario the AI should simulate, not a direct user request.
2.  Available Function List: The JSON schema of tools the agent can use.
3.  Ground-Truth Conversation: The sequence of assistant and tool call result (marked as "role": "observation") messages. Note that whenever an assistant makes a function call, the result will be in the subsequent "observation" message.


-----

Evaluation Criteria:

Evaluate the sample on each of the following dimensions using a 1-5 point scale. Below are example descriptions for scores 1, 3, and 5. You are veryencouraged to use scores 2 and 4 for cases that fall between these descriptions, since most real samples will likely fall somewhere between the anchor points described below. Provide a clear, critical reasoning for every score.

1. Tool Necessity
* 5 points: Every single step of the sub-task required to solve the given task is fundamentally impossible without the specific tools provided.
* 3 points: The core task requires tools to complete, but small peripheral aspects or subtasks could be handled using internal knowledge of model intensively trained on up-to-date data. e.g., identifying the airport name given the city
* 1 points: A model intensively trained on up-to-date data could potentially solve the task without any tools, making the tool calls feel optional or of limited value.

2. Planning and Context Depth 
* 5 points: Requires highly complex, non-linear planning with multiple dependencies between tool calls. The agent must track a long and detailed context to decide every next function call.
* 3 points: Requires a standard multi-step plan where the output of one step informs the next.
* 1 points: Requires only a single tool call or a static, predefined sequence of calls. Context is not important.

3. Parameter Generation
* 5 points: Generating the correct parameters for function calls requires deep semantic understanding of user intent. Some of the function calls requires a long, complex value (e.g., tokens).
* 3 points: Requires some basic reasoning or extraction from context (e.g., calculating a date from "tomorrow").
* 1 points: Parameters are simple values copied directly from the user prompt.

4. Tool Selection Difficulty
* 5 points: The toolset contains highly plausible and confusing distractors (e.g., such as similarly named tools). The task is design to actively tempt an agent into making the wrong choice, which results in the failure of the task.
* 3 points: The toolset contains a few distinct but related options, requiring the agent to discern subtle differences to make the correct choice based on the context and correct understanding of the user's intention.
* 1 points: The tool choice is obvious every step. The selection is straightforward and does not require deep reasoning or understanding of the context.

5. Real-World Applicability
* 5 points: Represents an extremely common, daily scenario that millions of users encounter with identical specificity. Every detail reflects typical user behavior patterns and natural language use.
* 3 points: Based on realistic, common scenarios that people do encounter, but with some specific requirements or constraints that are slightly artificial or less typical in practice.
* 1 points: Clearly synthetic or academic in nature - designed for evaluation rather than reflecting genuine user needs.

-----

Output Format:

Based on your evaluation, aggregate the scores of each dimension in the jsonl format as follows. 
Note that the dimensions must be arranged in the order listed above, and ensure that no dimensions are skipped.
Do not include any additional comments or explanations, and only include the JSONL output. That is, your response should start directly with [ and end with ].

Example:
[
    {{
    "dimension": "tool necessity",
    "reasoning": "The user's goal of booking a flight and a taxi involves interacting with external reservation systems. This is fundamentally impossible to achieve with only the model's internal knowledge. However, small sub-tasks such as identifying the closest airport from the user's location could be handled without external APIs.",
    "score": 3
    }},
    {{
    "dimension": "planning and context depth",
    "reasoning": "The task requires a sequence: 1. Search for a flight, 2. Use the flight's arrival airport to book a taxi. This is a **standard multi-step plan with a clear, linear dependency**. However, it does not require **complex, non-linear planning or adaptation to unexpected results**, which would be necessary for a score of 5.",
    "score": 4
    }},
    {{
    "dimension": "parameter generation",
    "reasoning": "Assuming the user prompt mentioned 'tomorrow', the agent needs to calculate the exact date. This is a **form of basic reasoning**, fitting the 3-point criteria. It does not require **deep semantic inference or the generation of a long, complex value** (like a full JSON object for filtering).",
    "score": 3
    }},
    {{
    "dimension": "tool selection difficulty",
    "reasoning": "The user's intent to 'search for a flight' and 'book a taxi' maps directly to tools like `search_flights` and `book_taxi`. There are **no plausible or confusing distractor tools** mentioned. The choice is obvious and straightforward.",
    "score": 2
    }},
    {{
    "dimension": "real-world applicability",
    "reasoning": "Booking a flight and then arranging for transportation from the airport is a very common and practical real-world scenario for travelers. However, some of the conditions that the user demands are a bit unrealistic.",
    "score": 3
    }}
]


-----

User Input:

### User Prompt

```
{user_prompt}
```

### Available Function List

```json
{available_function_list}
```

### Ground-truth conversation

```json
{conversations}
```


"""