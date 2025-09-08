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
import time
import os
from multiprocessing import Pool
from dotenv import load_dotenv
from tqdm import tqdm
from src.bench_loaders import get_bench_loader
from src.utils.types import Benchmark, FormattedQuestion, LLMJudgeOutput, LLMJudgeStep
from src.utils.format_judge_prompt import format_judge_prompt

load_dotenv()
logger = logging.getLogger(__name__)

# Disable HTTP request logging from OpenAI and httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)



@dataclass
class LLMJudgeConfig:
    model: str = "openai/gpt-4.1"       # Default model
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 10                # TODO: Implement batching
    num_proc: int = 32
    max_samples: Optional[int] = None   # Limit for testing
    steps: List[LLMJudgeStep] = None            # Which steps to run (default: both FILTER and SCORE)


def _assess_question_worker(args):
    """Worker function for multiprocessing question assessment."""
    question, step, model, api_key, base_url, max_retries, retry_delay = args
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    evaluation_prompt = format_judge_prompt(question, step)
    
    for attempt in range(max_retries):
        try:
            response_format = {"type": "json_object"} if step == LLMJudgeStep.FILTER else None
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
            
            if step == LLMJudgeStep.FILTER:
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


class LLMJudge:
    """LLM-as-Judge filtering for benchmark quality assessment."""
    
    def __init__(self, benchmark: Benchmark, config: LLMJudgeConfig = None):
        self.config = config or LLMJudgeConfig()
        if self.config.steps is None:
            self.config.steps = [LLMJudgeStep.FILTER, LLMJudgeStep.SCORE]  # Default to both steps
        self.benchmark = benchmark

    def get_results(self) -> List[LLMJudgeOutput]:
        """Load benchmark questions and run configured assessments."""
        questions = self._load_benchmark_questions()
        if self.config.max_samples:
            questions = questions[:self.config.max_samples]

        filter_results, score_results = [], []
        
        if LLMJudgeStep.FILTER in self.config.steps:
            logger.info(f"Running FILTER assessment on {len(questions)} questions")
            filter_results = self.assess_questions(questions, LLMJudgeStep.FILTER)
        
        if LLMJudgeStep.SCORE in self.config.steps:
            logger.info(f"Running SCORE assessment on {len(questions)} questions")  
            score_results = self.assess_questions(questions, LLMJudgeStep.SCORE)

        judgement_results: Optional[LLMJudgeOutput] = []
        for i, question in enumerate(questions):
            filter_result = filter_results[i].get("assessment", {})
            result = LLMJudgeOutput(
                benchmark=question.benchmark,
                question_id=question.question_id,
                is_flawed=filter_result['is_flawed'],
                error_category=filter_result['error_category'],
                reasoning=filter_result['reasoning'],
                reasoning_summary=filter_result['reasoning_summary']
            )
            
            if LLMJudgeStep.SCORE in self.config.steps:
                score_result = score_results[i].get("assessment", {})
                result.scores = score_result
            
            judgement_results.append(result)
            
        return judgement_results

    def load_benchmark_and_get_step_results(self, step: LLMJudgeStep = LLMJudgeStep.FILTER) -> List[Dict]:
        """Run the LLM-as-Judge assessment."""
        questions = self._load_benchmark_questions()
        
        if self.config.max_samples:
            questions = questions[:self.config.max_samples]
            
        logger.info(f"Assessing {len(questions)} questions using {self.config.num_proc} processes")
        results = self.assess_questions(questions, step)
        logger.info(f"Assessment completed. {len(results)} results generated.")
        return results

    def _load_benchmark_questions(self) -> List[FormattedQuestion]:
        loader_class = get_bench_loader(self.benchmark)
        loader = loader_class()
        return loader.load_questions()

    def _construct_judge_prompt(self, question: FormattedQuestion, step: LLMJudgeStep) -> str:
        prompt = format_judge_prompt(question, step)
        return prompt

    def _assess_question(self, question: FormattedQuestion, step: LLMJudgeStep) -> Dict:
        client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL")
        )
        evaluation_prompt = self._construct_judge_prompt(question, step)

        for attempt in range(self.config.max_retries):  # Retry logic
            try:
                response = client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                       {"role": "user", "content": evaluation_prompt}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"} if (step == LLMJudgeStep.FILTER) else None
                )

                response_content = response.choices[0].message.content
                if not response_content or response_content.strip() == "":
                    raise ValueError("Empty response from API")
                    
                result = json.loads(response_content)
                print(result)
                return result
            except Exception as e:
                if attempt == self.config.max_retries - 1:  # Last attempt
                    return {"error": str(e)}
                time.sleep(self.config.retry_delay)
    
    def assess_questions(self, questions: List[FormattedQuestion], step: LLMJudgeStep) -> List[Dict]:
        """Assess questions using multiprocessing."""
        if self.config.num_proc == 1:   # Single process
            results = []
            for question in tqdm(questions, desc="Processing questions"):
                try:
                    assessment = self._assess_question(question, step)
                except Exception as e:
                    logger.error(f"Error assessing question {question.question_id} in {question.benchmark.value} {e}")
                    assessment = {"error": str(e)}

                results.append({
                    "benchmark": question.benchmark.value,
                    "question_id": question.question_id,
                    "assessment": assessment
                })
            return results
        else:   # Multiprocessing
            logger.info(f"Using multiprocessing with {self.config.num_proc} processes")
            
            args_list = []
            for question in questions:
                args_list.append((
                    question, step, self.config.model, os.getenv("API_KEY"), os.getenv("BASE_URL"), 
                    self.config.max_retries, self.config.retry_delay
                ))
            
            with Pool(processes=self.config.num_proc) as pool:
                results = []
                with tqdm(total=len(args_list), desc="Processing questions (multiprocessing)") as pbar:
                    for result in pool.imap(_assess_question_worker, args_list):
                        results.append(result)
                        pbar.update(1)
            
            return results
