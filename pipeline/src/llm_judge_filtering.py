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
from src.utils.types import Benchmark, FormattedQuestion, LLMJudgeOutput, LLMJudgeStep, UniqueQuestionID
from src.utils.format_judge_prompt import format_judge_prompt
import re

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


class LLMJudge:
    """LLM-as-Judge filtering for benchmark quality assessment."""

    def __init__(self, config: LLMJudgeConfig = None):
        self.config = config or LLMJudgeConfig()
        if self.config.steps is None:
            self.config.steps = [LLMJudgeStep.UNIVERSAL_FILTER, LLMJudgeStep.SPECIFIC_FILTER, LLMJudgeStep.SCORE]

    @staticmethod
    def _make_api_call(client: OpenAI, model: str, evaluation_prompt: str, max_retries: int, retry_delay: float) -> Dict:
        for attempt in range(max_retries):
            try:
                response_format = {"type": "json_object"}
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
                return result

            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    return {"error": str(e)}
                time.sleep(retry_delay)
    
    def _parse_task_name_from_question_id(self, question_id: str) -> str:
        """Parse task name from question_id by removing the last number part."""
        match = re.match(r'^(.+)[-_](\d+)$', question_id)
        return match.group(1) if match else "" 

    def judge_questions(self, question_ids: List[UniqueQuestionID]) -> Dict[UniqueQuestionID, LLMJudgeOutput]:
        """Run configured assessments on the provided question IDs."""
        questions = self._load_questions_by_ids(question_ids)
        if self.config.max_samples:
            questions = questions[:self.config.max_samples]
        
        if LLMJudgeStep.UNIVERSAL_FILTER in self.config.steps:
            logger.info(f"Running universal LLM-judge filtering on {len(questions)} questions")
            filter_results = self.assess_questions(questions, LLMJudgeStep.UNIVERSAL_FILTER)
        if LLMJudgeStep.SPECIFIC_FILTER in self.config.steps:
            logger.info(f"Running benchmark-specific LLM-judge filtering on {len(questions)} questions")
            filter_results = self.assess_questions(questions, LLMJudgeStep.SPECIFIC_FILTER)
        if LLMJudgeStep.SCORE in self.config.steps:
            logger.info(f"Running SCORE assessment on {len(questions)} questions")  
            score_results = self.assess_questions(questions, LLMJudgeStep.SCORE)

        results = dict()
        for i, question in enumerate(questions):
            filter_result = filter_results[i].get("assessment", {})
            unique_question_id = UniqueQuestionID(
                benchmark=question.benchmark,
                task_name=question.task_name or self._parse_task_name_from_question_id(question.question_id),
                question_id=question.question_id
            )
            result = LLMJudgeOutput(
                is_flawed=filter_result['is_flawed'],
                error_category=filter_result['error_category'],
                reasoning=filter_result['reasoning'],
                reasoning_summary=filter_result['reasoning_summary']
            )

            if LLMJudgeStep.SCORE in self.config.steps:
                score_result = score_results[i].get("assessment", {})
                total_score = 0
                try:
                    for score in score_result['scores']:
                        total_score += score_result['scores'][score] 
                    score_result['total_score'] = total_score
                except:
                    score_result['total_score'] = 0
                result.scores = score_result

            results[unique_question_id] = result
            
        return results 

    def _load_questions_by_ids(self, question_ids: List[UniqueQuestionID]) -> List[FormattedQuestion]:
        """Load questions by their unique IDs."""
        # Group question IDs by benchmark
        questions_by_benchmark = {}
        for q_id in question_ids:
            if q_id.benchmark not in questions_by_benchmark:
                questions_by_benchmark[q_id.benchmark] = []
            questions_by_benchmark[q_id.benchmark].append(q_id)
        
        # Load questions from each benchmark
        all_questions = []
        for benchmark, ids_in_benchmark in questions_by_benchmark.items():
            loader_class = get_bench_loader(benchmark)
            loader = loader_class()
            benchmark_questions = loader.load_questions()
            # Filter to only the requested question IDs
            filtered_questions = [
                q for q in benchmark_questions 
                if q in ids_in_benchmark
            ]
            all_questions.extend(filtered_questions)
        
        return all_questions

    def _assess_question(self, question: FormattedQuestion, step: LLMJudgeStep) -> Dict:
        client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL")
        )
        evaluation_prompt = format_judge_prompt(question, step)

        return self._make_api_call(
            client, self.config.model, evaluation_prompt,
            self.config.max_retries, self.config.retry_delay
        )
    
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


def _assess_question_worker(args):
    """Worker function for multiprocessing question assessment."""
    question, step, model, api_key, base_url, max_retries, retry_delay = args
    client = OpenAI(api_key=api_key, base_url=base_url)
    evaluation_prompt = format_judge_prompt(question, step)

    result = LLMJudge._make_api_call(client, model, evaluation_prompt, max_retries, retry_delay)
    return {"question": question, "assessment": result}