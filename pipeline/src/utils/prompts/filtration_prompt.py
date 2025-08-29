prompt = """
You are an expert AI assistant specializing in the meticulous evaluation of function-calling benchmarks. Your task is to act as a judge and determine if a provided ground-truth function call is flawed based on a user's prompt and a set of available tools. A ground-truth is considered flawed if it is logically inconsistent, factually incorrect, or unexecutable based on the user's explicit request.

This evaluation is for the {benchmark} benchmark.

You will be given three pieces of information:

1.  User Prompt: This could be either:
    - For ComplexFuncBench: The original request from the user.
    - For Tau-bench: The instruction given to the AI model (acting as a customer service representative). In this case, the "user prompt" actually describes the persona and scenario the AI should simulate, not a direct user request.
2.  Available Function List: The JSON schema of tools the agent can use.
3.  Ground-Truth Conversation: The sequence of assistant and tool call result (marked as "role": "observation") messages. Note that whenever an assistant makes a function call, the result will be in the subsequent "observation" message.

-----

Evaluation Criteria:

You must meticulously check the ground-truth for the following specific categories of flaws.

1. Argument Value Mismatch: An argument's value in the ground-truth directly contradicts a clear instruction in the user's prompt.

Examples:

* Using the wrong date, time, or year (e.g., prompt asks for "New Year of 2024" but the call uses "2025-01-01")
* Swapping origin and destination cities.
* Searching for the "fastest" flight when the prompt asked for the "cheapest".
* Using a completely irrelevant location (e.g., booking a car in Seattle for a request in Las Vegas).
* Incorrectly calculating time differences (e.g., booking a taxi one hour *before* landing when the prompt asked for one hour *after*).

2. Argument Type Mismatch: An argument's data type in the ground-truth does not match the type specified in the function schema.

Examples:

* Providing a coordinate as a floating-point number when the schema requires a string.
* Passing an ID as a string (e.g., "1093") when the schema requires a number (`1093`).

3. Unjustified Assumption / Logical Flaw: The ground-truth makes a specific choice that is not supported by the prompt, especially when there are multiple valid options or the prompt is ambiguous. 
Ensure that before you judge that the ground truth function call used an unjustified assumption, check the previous API call results, which is contained in the `"role": "observation"` message in the conversation.

Example:

* The user asks for a flight from "NYC." The cheapest flight departs from EWR, but the ground-truth assumes the destination is JFK for a subsequent taxi booking without justification.

4. Misspelling: An argument value contains a clear typographical error that would likely cause an API call to fail.

Example:

* A parameter value is misspelled, such as `popularitye` instead of `popularity`.

5. Dataset Integrity Issue: The ground-truth expects a tool call that is impossible to formulate based on the information available from previous observation messages.

Example: 

* The observation for a flight search returns available dates from Nov 5-9, but the ground-truth tool call attempts to book a flight on Nov 15, a date for which no information was provided.

-----

Instructions:

1. Analyze User Intent: Carefully parse the initial User Prompt to fully understand all explicit constraints (dates, times, locations, conditions, etc.). 
   - For ComplexFuncBench: Understand what the user is requesting.
   - For Tau-bench: Understand the persona/scenario the AI model should simulate and what actions would be appropriate for that role.
2. Sequentially Verify Conversation: Iterate through the Ground-Truth Conversation message by message.

    * When you encounter a message from the assistant containing tool_calls, pause and evaluate it.
    * Use the user's intent (from Step 1) and any preceding "role": "observation" messages as the context for your evaluation.
    * Check the tool call against all the Evaluation Criteria listed above.

3. Stop at First Flaw: Your evaluation of the conversation must stop at the very first flawed tool call you identify. The remainder of the conversation should be ignored. If there are no flaws, evaluate the entire conversation.

4. Formulate Your Verdict: Based on your analysis, provide your final decision in the required JSON format. Your reasoning must focus only on the first flaw found (or confirm that no flaws exist).

```json
{{
  "reasoning": "Provide a clear, step-by-step explanation for your decision. If the ground-truth is flawed, specify which argument is incorrect and why it contradicts the prompt or schema. If it is not flawed, briefly explain why the ground-truth is a correct interpretation of the user's request."
  "reasoning_summary": "A shorter rationale for your decision. If the ground-truth is not flawed, just mention that it is not flawed. If the ground-truth is flawed, specify the issue concisely. e.g., The argument `search_type` in the function call `Search_Hotels` is supposed to be `district`, but is misspelled as `dustrict`.",
  "error_category": "<Argument Value Mismatch | Argument Type Mismatch | Unjustified Assumption | Misspelling | Not Flawed>",
  "is_flawed": <true_or_false>,
}}
```

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

