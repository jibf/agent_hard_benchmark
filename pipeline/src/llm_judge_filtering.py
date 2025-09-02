#!/usr/bin/env python3
"""
LLM-as-Judge filtering module.
Evaluates benchmark quality using LLM-based assessment.
"""

import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from openai import OpenAI
from enum import Enum
import time
import os
from multiprocessing import Pool
from dotenv import load_dotenv
from tqdm import tqdm
from functools import wraps
from src.bench_loaders import (
    TauBenchLoader, Tau2BenchLoader, ComplexFuncBenchLoader, NexusBenchLoader, DrafterBenchLoader,
    AceBenchLoader, BfclV2Loader, BfclV3Loader, MultiChallengeLoader, ToolSandBoxLoader
)
from src.utils.types import Benchmark, FormattedQuestion
from src.utils.format_judge_prompt import format_judge_prompt

load_dotenv()
logger = logging.getLogger(__name__)

# Disable HTTP request logging from OpenAI and httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)


class Step(Enum):
    FILTER = "filter"
    SCORE = "score"


@dataclass
class LLMJudgeConfig:
    """Configuration for LLM-as-Judge filtering."""
    model: str = "openai/gpt-4.1"  # Default model
    max_retries: int = 3        # TODO: Implement retry logic
    retry_delay: float = 1.0
    batch_size: int = 10        # TODO: Implement batching
    num_proc: int = 32
    max_samples: Optional[int] = None  # Limit for testing


def _assess_question_worker(args):
    """Worker function for multiprocessing question assessment."""
    question, step, model, api_key, base_url, max_retries, retry_delay = args
    
    # Create OpenAI client for this process
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Construct prompt using format_judge_prompt
    eval_type = 'filtration' if step == Step.FILTER else 'scoring'
    evaluation_prompt, _ = format_judge_prompt(question, eval_type)
    
    # Retry logic
    for attempt in range(max_retries):
        try:
            # Make API call
            response_format = {"type": "json_object"} if step == Step.FILTER else None
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.0,
                response_format=response_format
            )
            
            response_content = response.choices[0].message.content
            if not response_content or response_content.strip() == "":
                raise ValueError("Empty response from API")
                
            result = json.loads(response_content)
            
            if step == Step.FILTER:
                try:
                    result = {
                        "is_flawed": result["is_flawed"],
                        "reasoning_summary": result["reasoning_summary"],
                        **{k: v for k, v in result.items() if k not in ["is_flawed", "reasoning_summary"]}
                    }
                except KeyError as ke:
                    raise ValueError(f"Missing key in response: {ke}")
            
            return {"question": question, "assessment": result}
            
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                return {"question": question, "assessment": {"error": str(e)}}
            time.sleep(retry_delay)


class LLMJudgeAssessor:
    """LLM-as-Judge filtering for benchmark quality assessment."""
    
    def __init__(self, benchmark: Benchmark, config: LLMJudgeConfig = None):
        self.config = config or LLMJudgeConfig()
        self.benchmark = benchmark

    def load_benchmark_and_get_results(self) -> List[Dict]:
        """Load benchmark questions and run both FILTER and SCORE assessments."""
        questions = self._load_benchmark_questions()

        if self.config.max_samples:
            questions = questions[:self.config.max_samples]

        logger.info(f"Running FILTER assessment on {len(questions)} questions")
        filter_results = self.assess_questions(questions, Step.FILTER)
        
        logger.info(f"Running SCORE assessment on {len(questions)} questions")  
        score_results = self.assess_questions(questions, Step.SCORE)

        # Combine results for each question
        combined_results = []
        for i, question in enumerate(questions):
            combined_result = {
                "benchmark": question.benchmark.value,
                "question_id": question.question_id,
                "user_prompt": question.instruction,
                "available_function_list": question.available_function_list,
                "conversations": question.gt_conv_traj,
                "filter_assessment": filter_results[i].get("assessment", {}),
                "score_assessment": score_results[i].get("assessment", {})
            }
            combined_results.append(combined_result)
            
        return combined_results

    def load_benchmark_and_get_step_results(self, step: Step = Step.FILTER) -> List[Dict]:
        """Run the LLM-as-Judge assessment."""
        questions = self._load_benchmark_questions()
        
        if self.config.max_samples:
            questions = questions[:self.config.max_samples]
            
        logger.info(f"Assessing {len(questions)} questions using {self.config.num_proc} processes")
        results = self.assess_questions(questions, step)
        logger.info(f"Assessment completed. {len(results)} results generated.")
        return results

    def _load_benchmark_questions(self) -> List[FormattedQuestion]:
        loader_class_map = {
            Benchmark.TAU_BENCH: TauBenchLoader,
            Benchmark.TAU2_BENCH: Tau2BenchLoader,
            Benchmark.COMPLEX_FUNC_BENCH: ComplexFuncBenchLoader,
            Benchmark.NEXUS_BENCH: NexusBenchLoader,
            Benchmark.DRAFTER_BENCH: DrafterBenchLoader,
            Benchmark.ACE_BENCH: AceBenchLoader,
            Benchmark.BFCLV2: BfclV2Loader,
            Benchmark.BFCLV3: BfclV3Loader,
            Benchmark.MULTI_CHALLENGE: MultiChallengeLoader,
            Benchmark.TOOL_SANDBOX: ToolSandBoxLoader
        }
        
        loader_class = loader_class_map.get(self.benchmark)
        loader = loader_class()
        if not loader:
            raise ValueError(f"Unsupported benchmark type: {self.benchmark}")
        
        return loader.load_questions()

    def _construct_judge_prompt(self, question: FormattedQuestion, step: Step) -> str:
        eval_type = 'filtration' if step == Step.FILTER else 'scoring'
        prompt, _ = format_judge_prompt(question, eval_type)
        return prompt

    def _assess_question(self, question: FormattedQuestion, step: Step) -> Dict:
        client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL")
        )
        response_format = {"type": "json_object"} if step == Step.FILTER else None
        evaluation_prompt = self._construct_judge_prompt(question, step)

        # Retry logic
        for attempt in range(self.config.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                       {"role": "user", "content": evaluation_prompt}
                    ],
                    temperature=0.0,
                    response_format=response_format
                )
                        
                response_content = response.choices[0].message.content
                if not response_content or response_content.strip() == "":
                    raise ValueError("Empty response from API")
                    
                result = json.loads(response_content)
                print(result)

                if step == Step.FILTER:
                    try:
                        result = {
                            "is_flawed": result["is_flawed"],
                            "reasoning_summary": result["reasoning_summary"],
                            **{k: v for k, v in result.items() if k not in ["is_flawed", "reasoning_summary"]}
                        }
                    except KeyError as ke:
                        raise ValueError(f"Missing key in response: {ke}")
                return result
            except Exception as e:
                if attempt == self.config.max_retries - 1:  # Last attempt
                    return {"error": str(e)}
                time.sleep(self.config.retry_delay)
    
    def assess_questions(self, questions: List[FormattedQuestion], step: Step) -> List[Dict]:
        """Assess questions using multiprocessing."""
        if self.config.num_proc == 1:
            # Single process mode with tqdm progress bar
            results = []
            for question in tqdm(questions, desc="Processing questions"):
                try:
                    assessment = self._assess_question(question, step)
                    results.append({
                        "benchmark": question.benchmark.value,
                        "question_id": question.question_id,
                        "assessment": assessment
                    })
                except Exception as e:
                    logger.error(f"Error assessing question: {e}")
                    results.append({
                        "benchmark": question.benchmark.value,
                        "question_id": question.question_id,
                        "assessment": {"error": str(e)}
                    })
            return results
        else:
            # Multiprocessing mode with tqdm progress bar
            logger.info(f"Using multiprocessing with {self.config.num_proc} processes")
            
            # Prepare arguments for multiprocessing
            args_list = []
            for question in questions:
                args_list.append((
                    question,
                    step,
                    self.config.model,
                    os.getenv("API_KEY"),
                    os.getenv("BASE_URL"),
                    self.config.max_retries,
                    self.config.retry_delay
                ))
            
            # Use multiprocessing with progress bar
            with Pool(processes=self.config.num_proc) as pool:
                results = []
                with tqdm(total=len(args_list), desc="Processing questions (multiprocessing)") as pbar:
                    for result in pool.imap(_assess_question_worker, args_list):
                        results.append(result)
                        pbar.update(1)
            
            return results
