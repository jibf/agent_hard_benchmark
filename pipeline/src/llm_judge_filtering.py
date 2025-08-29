#!/usr/bin/env python3
"""
LLM-as-Judge filtering module.
Evaluates benchmark quality using LLM-based assessment.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from openai import OpenAI
import os
import time
from tqdm import tqdm
from utils.prompts import flawed_gt_filtering, prompt_scoring
from utils.formatters.tau_formatter import TauBenchFormatter
from benchmark_types import BenchmarkType
import multiprocessing
from multiprocessing import Pool
from datetime import datetime
import hashlib
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class Step(Enum):
    FILTER = "filter"
    SCORE = "score"

@dataclass
class LLMJudgeConfig:
    """Configuration for LLM-as-Judge filtering."""
    model: str = "openai/gpt-4.1"  # Default model
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 10
    max_samples: Optional[int] = None  # Limit for testing

class LLMJudgeFilter:
    """LLM-as-Judge filtering for benchmark quality assessment."""
    
    def __init__(self, config: LLMJudgeConfig = None):
        self.config = config or LLMJudgeConfig()
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL")
        )
        
    def filter_samples(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Apply LLM-as-Judge filtering to samples."""
        logger.info("Starting LLM-as-Judge filtering")
        
        if self.config.max_samples:
            samples = samples[:self.config.max_samples]
            logger.info(f"Limited to {len(samples)} samples for testing")
        
        # Process samples in batches
        passed_samples = []
        dropped_samples = []
        
        for i in range(0, len(samples), self.config.batch_size):
            batch = samples[i:i + self.config.batch_size]
            logger.info(f"Processing batch {i//self.config.batch_size + 1}/{(len(samples)-1)//self.config.batch_size + 1}")
            
            batch_passed, batch_dropped = self._process_batch(batch)
            passed_samples.extend(batch_passed)
            dropped_samples.extend(batch_dropped)
            
            # Add delay between batches to avoid rate limits
            if i + self.config.batch_size < len(samples):
                time.sleep(0.5)
        
        logger.info("=== LLM-as-Judge Filtering Results ===")
        logger.info(f"Total samples: {len(samples)}")
        logger.info(f"Passed: {len(passed_samples)} ({len(passed_samples)/len(samples)*100:.1f}%)")
        logger.info(f"Dropped: {len(dropped_samples)} ({len(dropped_samples)/len(samples)*100:.1f}%)")
        
        return passed_samples, dropped_samples
    
    def assess_questions(self, questions: List[Dict], step: Step = Step.FILTER, benchmark_type: BenchmarkType = BenchmarkType.COMPLEX_FUNC_BENCH, proc_num: int = 1) -> List[Dict]:
        """Assess questions independently using LLM-as-Judge."""
        logger.info(f"Starting question assessment with {len(questions)} questions")
        
        if self.config.max_samples:
            questions = questions[:self.config.max_samples]
            logger.info(f"Limited to {len(questions)} questions for testing")
        
        results = []
        
        if proc_num > 1:
            # Use multiprocessing
            with Pool(processes=proc_num) as pool:
                with tqdm(total=len(questions), desc="Processing questions", unit="question") as pbar:
                    pool_results = []
                    for question_data in questions:
                        result = pool.apply_async(self._process_question_mp, (question_data, step, benchmark_type, self.config.model))
                        pool_results.append(result)
                    
                    # Collect results as they complete
                    for result in pool_results:
                        result_data = result.get()
                        if result_data:
                            results.append(result_data)
                        pbar.update(1)
        else:
            # Sequential processing
            for question_data in tqdm(questions, desc="Processing questions"):
                result = self._process_question(question_data, step, benchmark_type)
                if result:
                    results.append(result)
        
        logger.info(f"Question assessment completed: {len(results)} results")
        return results
    
    def _process_batch(self, batch: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Process a batch of samples."""
        passed_samples = []
        dropped_samples = []
        
        for sample in batch:
            try:
                # Extract conversation data
                conversation_data = self._extract_conversation_data(sample)
                if not conversation_data:
                    dropped_samples.append(sample)
                    continue
                
                # Apply both evaluation criteria
                quality_score = self._evaluate_quality(conversation_data)
                flaw_check = self._check_for_flaws(conversation_data)
                
                # Decision logic
                if quality_score >= 3.0 and not flaw_check["is_flawed"]:
                    passed_samples.append(sample)
                else:
                    dropped_samples.append(sample)
                    
            except Exception as e:
                logger.warning(f"Error processing sample: {e}")
                dropped_samples.append(sample)
        
        return passed_samples, dropped_samples
    
    def _process_question(self, question_data: Dict, step: Step = Step.FILTER, benchmark_type: BenchmarkType = BenchmarkType.COMPLEX_FUNC_BENCH) -> Optional[Dict]:
        """Process a single question for assessment."""
        try:
            # Extract conversation data based on benchmark type
            user_prompt, conversations, available_function_list = self._extract_conversation_from_question(question_data, benchmark_type)
            
            if not user_prompt or not conversations:
                logger.warning(f"Missing prompt or conversations in question")
                logger.debug(f"Question data type: {type(question_data)}")
                logger.debug(f"Question data keys: {list(question_data.keys()) if isinstance(question_data, dict) else 'Not a dict'}")
                logger.debug(f"Has conversations attr: {hasattr(question_data, 'conversations')}")
                logger.debug(f"User prompt: '{user_prompt}'")
                logger.debug(f"Conversations: {conversations}")
                return None
            
            # Assess the question
            assessment = self._assess_question(user_prompt, conversations, available_function_list, step, benchmark_type)
            
            # Prepare result
            question_id = question_data.get('id', f'question_{hash(str(question_data))}')
            result = {
                "id": question_id,
                "original_data": question_data,
                "assessment": assessment
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"Error processing question: {e}")
            return None
    
    @staticmethod
    def _process_question_mp(question_data: Dict, step: Step, benchmark_type: BenchmarkType, model: str) -> Optional[Dict]:
        """Process a single question for multiprocessing."""
        # Create assessor instance in worker process
        config = LLMJudgeConfig(model=model)
        assessor = LLMJudgeFilter(config)
        return assessor._process_question(question_data, step, benchmark_type)
    
    def _extract_conversation_from_question(self, question_data: Dict, benchmark_type: BenchmarkType = BenchmarkType.COMPLEX_FUNC_BENCH) -> Tuple[str, List[Dict], List[Dict]]:
        """Extract conversation data from question, handling different benchmark formats."""
        
        if benchmark_type == BenchmarkType.COMPLEX_FUNC_BENCH:
            # Original ComplexFuncBench format
            conversations = question_data.get('conversations', [])
            available_function_list = question_data.get('functions', [])
            user_prompt = conversations[0].get('content', '') if conversations else ''
            return user_prompt, conversations, available_function_list
        
        elif benchmark_type == BenchmarkType.TAU_BENCH:
            # Tau-bench format - check if it's a FormattedQuestion object or dict
            if hasattr(question_data, 'conversations'):
                # It's a FormattedQuestion object
                conversations = question_data.conversations
                available_function_list = question_data.available_function_list
                # For tau-bench, create a synthetic user prompt since ground truth starts with assistant
                user_prompt = "This is a tau-bench evaluation task with ground truth conversation."
                return user_prompt, conversations, available_function_list
            elif 'conversations' in question_data:
                # It's a dict representation  
                conversations = question_data.get('conversations', [])
                available_function_list = question_data.get('available_function_list', [])
                # For tau-bench, create a synthetic user prompt since ground truth starts with assistant
                user_prompt = "This is a tau-bench evaluation task with ground truth conversation."
                return user_prompt, conversations, available_function_list
            else:
                # Use tau_formatter to extract
                tau_formatter = TauBenchFormatter()
                return tau_formatter.extract_conversation(question_data)
        
        else:
            raise ValueError(f"Unsupported benchmark type: {benchmark_type}")
    
    def _extract_conversation_data(self, sample: Dict) -> Optional[Dict]:
        """Extract conversation data for LLM evaluation."""
        try:
            messages = sample.get("messages", [])
            if not messages:
                return None
            
            # Extract user prompt (first user message)
            user_prompt = ""
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if content:
                        if isinstance(content, list):
                            text_parts = []
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    text_parts.append(part.get("text", ""))
                                elif isinstance(part, str):
                                    text_parts.append(part)
                            content = " ".join(text_parts)
                        
                        if isinstance(content, str):
                            user_prompt = content
                            break
            
            # Extract function/tool information
            available_functions = []
            for msg in messages:
                if msg.get("role") == "assistant" and "tool_calls" in msg:
                    for tool_call in msg["tool_calls"]:
                        if "function" in tool_call:
                            func_info = tool_call["function"]
                            available_functions.append({
                                "name": func_info.get("name", ""),
                                "arguments": func_info.get("arguments", "{}")
                            })
            
            # Format conversation for evaluation
            conversation = []
            for msg in messages:
                if msg.get("role") in ["user", "assistant", "observation"]:
                    conversation.append({
                        "role": msg["role"],
                        "content": msg.get("content", ""),
                        "tool_calls": msg.get("tool_calls", [])
                    })
            
            return {
                "user_prompt": user_prompt,
                "available_functions": available_functions,
                "conversation": conversation
            }
            
        except Exception as e:
            logger.warning(f"Error extracting conversation data: {e}")
            return None
    
    def _evaluate_quality(self, conversation_data: Dict) -> float:
        """Evaluate benchmark quality using the quality assessment prompt."""
        prompt = self._build_quality_prompt(conversation_data)
        
        try:
            response = self._call_llm(prompt)
            scores = self._parse_quality_response(response)
            
            if scores:
                # Calculate average score across all dimensions
                total_score = sum(score["score"] for score in scores)
                return total_score / len(scores)
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error evaluating quality: {e}")
            return 0.0
    
    def _check_for_flaws(self, conversation_data: Dict) -> Dict:
        """Check for flaws using the flaw detection prompt."""
        prompt = self._build_flaw_prompt(conversation_data)
        
        try:
            response = self._call_llm(prompt)
            return self._parse_flaw_response(response)
            
        except Exception as e:
            logger.warning(f"Error checking for flaws: {e}")
            return {"is_flawed": True, "error_category": "Error", "reasoning": str(e)}
    
    def _build_quality_prompt(self, conversation_data: Dict) -> str:
        """Build the quality evaluation prompt."""
        user_prompt = conversation_data["user_prompt"]
        available_functions = json.dumps(conversation_data["available_functions"], indent=2)
        conversations = json.dumps(conversation_data["conversation"], indent=2)
        
        return f"""You are an expert AI assistant specializing in the meticulous evaluation of function-calling benchmarks. Your task is to assess how effectively a given benchmark sample measures the capabilities of AI agents.

You will be given three pieces of information:

1.  User Prompt: The original request from the user.
2.  Available Function List: The JSON schema of tools the agent can use.
3.  Ground-Truth Conversation: The sequence of user, assistant, and the tool call result (marked as "role": "observation") messages. Note that whenever an assistant makes a function call, the result will be in the subsequent "observation" message.

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

-----

User Input:

### User Prompt

```
{user_prompt}
```

### Available Function List

```json
{available_functions}
```

### Ground-truth conversation

```json
{conversations}
```"""
    
    def _build_flaw_prompt(self, conversation_data: Dict) -> str:
        """Build the flaw detection prompt."""
        user_prompt = conversation_data["user_prompt"]
        available_functions = json.dumps(conversation_data["available_functions"], indent=2)
        conversations = json.dumps(conversation_data["conversation"], indent=2)
        
        return f"""You are an expert AI assistant specializing in the meticulous evaluation of function-calling benchmarks. Your task is to act as a judge and determine if a provided ground-truth function call is flawed based on a user's prompt and a set of available tools. A ground-truth is considered flawed if it is logically inconsistent, factually incorrect, or unexecutable based on the user's explicit request.

You will be given three pieces of information:

1.  User Prompt: The original request from the user.
2.  Available Function List: The JSON schema of tools the agent can use.
3.  Ground-Truth Conversation: The sequence of user, assistant, and the tool call result (marked as "role": "observation") messages. Note that whenever an assistant makes a function call, the result will be in the subsequent "observation" message.

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
{available_functions}
```

### Ground-truth conversation

```json
{conversations}
```"""
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=2000
                )
                return response.choices[0].message.content
                
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise e
    
    def _parse_quality_response(self, response: str) -> List[Dict]:
        """Parse the quality evaluation response."""
        try:
            # Extract JSON from response
            start_idx = response.find('[')
            end_idx = response.rfind(']')
            
            if start_idx == -1 or end_idx == -1:
                return []
            
            json_str = response[start_idx:end_idx + 1]
            scores = json.loads(json_str)
            
            return scores
            
        except Exception as e:
            logger.warning(f"Error parsing quality response: {e}")
            return []
    
    def _parse_flaw_response(self, response: str) -> Dict:
        """Parse the flaw detection response."""
        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                return {"is_flawed": True, "error_category": "Parse Error", "reasoning": "Could not parse response"}
            
            json_str = response[start_idx:end_idx + 1]
            result = json.loads(json_str)
            
            return result
            
        except Exception as e:
            logger.warning(f"Error parsing flaw response: {e}")
            return {"is_flawed": True, "error_category": "Parse Error", "reasoning": str(e)}
    
    def _assess_question(self, user_prompt: str, conversations: List[Dict], available_function_list: List[Dict], step: Step = Step.FILTER, benchmark_type: BenchmarkType = BenchmarkType.COMPLEX_FUNC_BENCH) -> Dict[str, Any]:
        """Assess a single question using LLM."""
        
        if step == Step.FILTER:
            prompt_module = flawed_gt_filtering
        elif step == Step.SCORE:
            prompt_module = prompt_scoring
        else:
            raise ValueError(f"Invalid step: {step}. Must be Step.FILTER or Step.SCORE")
        
        # Map benchmark type to readable string
        benchmark_type_str = "ComplexFuncBench" if benchmark_type == BenchmarkType.COMPLEX_FUNC_BENCH else "Tau-bench"
        
        evaluation_prompt = prompt_module.prompt.format(
            benchmark_type=benchmark_type_str,
            user_prompt=user_prompt,
            conversations=json.dumps(conversations),
            available_function_list=json.dumps(available_function_list)
        )
        
        try:
            # Only use json_object format for filter step, score step returns an array
            response_format = {"type": "json_object"} if step == Step.FILTER else None
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.0,
                response_format=response_format
            )
            
            response_content = response.choices[0].message.content
            if not response_content or response_content.strip() == "":
                return {"error": "Empty response from API"}
                
            result = json.loads(response_content)

            if step == Step.FILTER:
                try:
                    result = {
                        "is_flawed": result["is_flawed"], 
                        "reasoning_summary": result["reasoning_summary"],
                        **{k: v for k, v in result.items() if k not in ["is_flawed", "reasoning_summary"]}
                    }
                except KeyError as ke:
                    logger.warning(f"KeyError: {ke} not found in result")
                    return {"error": f"Missing key in response: {ke}"}
            return result
            
        except Exception as e:
            logger.warning(f"Error assessing question: {e}")
            return {"error": str(e)}
