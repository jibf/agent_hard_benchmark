#!/usr/bin/env python3
"""
LLM-as-Judge filtering module.
Evaluates benchmark quality using LLM-based assessment.
"""

import copy
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from src.bench_loaders import get_bench_loader
from src.utils.format_judge_prompt import format_judge_prompt
from src.utils.types import (
    Benchmark,
    FilterResult,
    FormattedQuestion,
    LLMJudgeOutput,
    LLMJudgeStep,
    RebuttalInfo,
    UniqueQuestionID,
)

load_dotenv()
logger = logging.getLogger(__name__)

# Disable HTTP request logging from OpenAI and httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)



DUMMY_ASSESSMENT = {
    "reasoning": "LLM Judge disabled for this task",
    "reasoning_summary": "LLM Judge skipped",
    "error_category": "LLM Judge disabled",
    "is_flawed": False
}

@dataclass
class LLMJudgeConfig:
    model: str = "openai/gpt-4.1"       # Default model
    max_retries: int = 3
    retry_delay: float = 1.0
    request_timeout: float = 180.0
    num_proc: int = 32
    max_samples: Optional[int] = None   # Limit for testing
    steps: List[LLMJudgeStep] = None            # Which steps to run (default: both FILTER and SCORE)
    partial_log_dir: Optional[str] = "llm_judge_partial_logs"
    enable_rebuttal: bool = False


class LLMJudge:
    """LLM-as-Judge filtering for benchmark quality assessment."""

    REBUTTAL_TRANSCRIPT_CHAR_LIMIT = 4500

    def __init__(self, config: LLMJudgeConfig = None):
        self.config = config or LLMJudgeConfig()
        if self.config.steps is None:
            self.config.steps = [LLMJudgeStep.UNIVERSAL_FILTER, LLMJudgeStep.SPECIFIC_FILTER, LLMJudgeStep.SCORE]

    @staticmethod
    def _extract_json_from_response(content: str) -> dict:
        """Extract JSON from response content that might be wrapped in code blocks."""
        # Try to parse as is first
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            pass

        # Look for JSON in code blocks
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Look for JSON without code blocks
        json_pattern = r'(\{.*?\})'
        match = re.search(json_pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not extract valid JSON from response: {content}")

    @staticmethod
    def _make_api_call(
        client: OpenAI,
        model: str,
        messages: List[Dict[str, Any]],
        max_retries: int,
        retry_delay: float,
        request_timeout: float
    ) -> Dict:
        is_gemini = "gemini" in model
        for attempt in range(max_retries):
            try:
                params = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 16384,
                }

                # For gemini models, don't use extra_body or response_format as they cause 500 errors
                if not is_gemini and "claude-4-opus-thinking-on-10k" not in model:
                    params["response_format"] = {"type": "json_object"}

                if "claude-4-opus-thinking-on-10k" in model:
                    params["temperature"] = 1.0
                response = client.chat.completions.create(timeout=request_timeout, **params)

                response_content = response.choices[0].message.content
                if not response_content or response_content.strip() == "":
                    raise ValueError("Empty response from API")

                # Use the JSON extraction method for gemini models that may wrap JSON in code blocks
                if is_gemini or "claude-4-opus-thinking-on-10k" in model:
                    result = LLMJudge._extract_json_from_response(response_content)
                else:
                    result = json.loads(response_content)

                return result

            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    error_message = str(e)
                    lowercase_error = error_message.lower()
                    is_timeout = isinstance(e, TimeoutError) or "timeout" in lowercase_error
                    error_category = "llm_judge_timeout" if is_timeout else "llm_judge_error"
                    return {
                        "is_flawed": True,
                        "error_category": error_category,
                        "reasoning": f"LLM judge request failed: {error_message}",
                        "reasoning_summary": "LLM judge request failed",
                        "error": error_message
                    }
                time.sleep(retry_delay)

    
    def _parse_task_name_from_question_id(self, question_id: str) -> str:
        """Parse task name from question_id by removing the last number part."""
        match = re.match(r'^(.+)[-_](\d+)$', question_id)
        return match.group(1) if match else "" 

    def _write_result_to_file(self, step: LLMJudgeStep, entry: Dict) -> None:
        log_dir = self.config.partial_log_dir
        if not log_dir:
            return

        try:
            base_path = Path(log_dir)
            base_path.mkdir(parents=True, exist_ok=True)
            file_path = base_path / f"{step.value}_results.jsonl"
            record = {"step": step.value, **entry}
            with file_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Failed to append partial log: %s", exc)

    def _create_client(self) -> OpenAI:
        return OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL")
        )

    def judge_questions(self, responses_by_question: Dict[UniqueQuestionID, List[Dict]]) -> Dict[UniqueQuestionID, LLMJudgeOutput]:
        """Run configured assessments on questions enriched with model responses from step1."""
        # Load questions and enrich them with model responses
        questions = self._load_questions_by_responses(responses_by_question)
        if self.config.max_samples:
            questions = questions[:self.config.max_samples]

        # Run filtering steps and collect results
        universal_results = None
        specific_results = None
        score_results = None

        if LLMJudgeStep.UNIVERSAL_FILTER in self.config.steps:
            logger.info(f"Running universal LLM-judge filtering on {len(questions)} questions")
            universal_results = self.assess_questions(questions, LLMJudgeStep.UNIVERSAL_FILTER)

        if LLMJudgeStep.SPECIFIC_FILTER in self.config.steps:
            logger.info(f"Running benchmark-specific LLM-judge filtering on {len(questions)} questions")
            specific_results = self.assess_questions(questions, LLMJudgeStep.SPECIFIC_FILTER)

        if LLMJudgeStep.SCORE in self.config.steps:
            logger.info(f"Running SCORE assessment on {len(questions)} questions")
            score_results = self.assess_questions(questions, LLMJudgeStep.SCORE)

        # Combine results
        results = dict()
        for i, question in enumerate(questions):
            unique_question_id = UniqueQuestionID(
                benchmark=question.benchmark,
                task_name=question.task_name or self._parse_task_name_from_question_id(question.question_id),
                question_id=question.question_id
            )

            result = LLMJudgeOutput()

            # Add universal filter result
            if universal_results:
                universal_assessment = universal_results[i].get("assessment", {})
                if universal_assessment:
                    try:
                        filter_kwargs = {
                            "is_flawed": universal_assessment["is_flawed"],
                            "error_category": universal_assessment.get("error_category"),
                            "reasoning": universal_assessment.get("reasoning"),
                            "reasoning_summary": universal_assessment.get("reasoning_summary"),
                        }
                        summary = universal_results[i].get("rebuttal_summary")
                        if summary:
                            filter_kwargs["rebuttal"] = RebuttalInfo(**summary)
                        result.universal_filter = FilterResult(**filter_kwargs)
                    except KeyError:
                        logger.warning(
                            "Universal assessment missing required fields for question %s",
                            question.question_id,
                        )

            # Add specific filter result
            if specific_results:
                specific_assessment = specific_results[i].get("assessment", {})
                if specific_assessment:
                    try:
                        filter_kwargs = {
                            "is_flawed": specific_assessment["is_flawed"],
                            "error_category": specific_assessment.get("error_category"),
                            "reasoning": specific_assessment.get("reasoning"),
                            "reasoning_summary": specific_assessment.get("reasoning_summary"),
                        }
                        summary = specific_results[i].get("rebuttal_summary")
                        if summary:
                            filter_kwargs["rebuttal"] = RebuttalInfo(**summary)
                        result.specific_filter = FilterResult(**filter_kwargs)
                    except:
                        result.specific_filter = None

            # Add scoring result
            if score_results:
                score_result = score_results[i].get("assessment", {})
                if score_result:
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

    def _load_questions_by_responses(self, responses_by_question: Dict[UniqueQuestionID, List[Dict]]) -> List[FormattedQuestion]:
        """Load questions by responses_by_question and enrich them with model responses."""
        # Group question IDs by benchmark
        questions_by_benchmark = {}
        for q_id in responses_by_question.keys():
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

            loader.load_responses_for_questions(filtered_questions, responses_by_question)

            all_questions.extend(filtered_questions)
        
        return all_questions

    def _assess_question(
        self,
        question: FormattedQuestion,
        step: LLMJudgeStep,
        client: Optional[OpenAI] = None
    ) -> Tuple[Dict, str]:
        evaluation_prompt = format_judge_prompt(question, step)
        if question.skip_llm_judge:
            assessment = copy.deepcopy(DUMMY_ASSESSMENT)
        else:
            local_client = client or self._create_client()
            assessment = self._make_api_call(
                local_client,
                self.config.model,
                [{"role": "user", "content": evaluation_prompt}],
                self.config.max_retries,
                self.config.retry_delay,
                self.config.request_timeout,
            )
        return assessment, evaluation_prompt

    def _maybe_run_rebuttal(
        self,
        question: FormattedQuestion,
        step: LLMJudgeStep,
        entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        summary = {"applied": False}

        if not self.config.enable_rebuttal:
            summary["reason"] = "rebuttal_disabled"
            entry["rebuttal_summary"] = summary
            return entry

        # Only applicable for filtering steps that produced a dict assessment
        assessment = entry.get("assessment")
        if not isinstance(assessment, dict):
            summary["reason"] = "invalid_assessment_format"
            entry["rebuttal_summary"] = summary
            return entry

        if step == LLMJudgeStep.SCORE:
            summary["reason"] = "not_applicable_for_score_step"
            entry["rebuttal_summary"] = summary
            return entry

        if not assessment.get("is_flawed"):
            summary["reason"] = "sample_not_marked_flawed"
            entry["rebuttal_summary"] = summary
            return entry

        model_responses = getattr(question, "model_responses", None)
        if not model_responses:
            summary["reason"] = "no_model_responses_available"
            entry["rebuttal_summary"] = summary
            return entry

        supporting_response, supporting_score = self._select_successful_response(question)
        if supporting_response is None:
            summary["reason"] = "no_successful_response_found"
            entry["rebuttal_summary"] = summary
            return entry

        initial_assessment = copy.deepcopy(assessment)
        original_prompt = entry.get("prompt")
        if not original_prompt:
            original_prompt = format_judge_prompt(question, step)
            entry["prompt"] = original_prompt

        rebuttal_prompt = self._build_rebuttal_prompt(
            question,
            supporting_response,
            supporting_score
        )
        messages = [
            {"role": "user", "content": original_prompt},
            {"role": "assistant", "content": json.dumps(initial_assessment, ensure_ascii=False)},
            {"role": "user", "content": rebuttal_prompt},
        ]

        rebuttal_assessment = self._make_api_call(
            self._create_client(),
            self.config.model,
            messages,
            self.config.max_retries,
            self.config.retry_delay,
            self.config.request_timeout,
        )

        if not isinstance(rebuttal_assessment, dict):
            summary["reason"] = "rebuttal_response_not_dict"
            entry["initial_assessment"] = initial_assessment
            entry["rebuttal_assessment"] = rebuttal_assessment
            entry["rebuttal_prompt"] = rebuttal_prompt
            entry["rebuttal_summary"] = summary
            return entry

        final_is_flawed = bool(rebuttal_assessment.get("is_flawed"))
        initial_is_flawed = bool(initial_assessment.get("is_flawed"))
        summary.update(
            {
                "applied": True,
                "initial_is_flawed": initial_is_flawed,
                "final_is_flawed": final_is_flawed,
                "overturned": initial_is_flawed and not final_is_flawed,
                "model_name": supporting_response.get("model_name")
                or supporting_response.get("model_path"),
                "response_score": supporting_score,
                "supporting_response_id": (supporting_response.get("meta") or {}).get("id"),
            }
        )

        entry["assessment"] = rebuttal_assessment
        entry["initial_assessment"] = initial_assessment
        entry["rebuttal_assessment"] = rebuttal_assessment
        entry["rebuttal_prompt"] = rebuttal_prompt
        entry["rebuttal_summary"] = summary
        return entry

    def _select_successful_response(
        self, question: FormattedQuestion
    ) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        responses = getattr(question, "model_responses", None) or []
        best_response = None
        best_score = None
        for response in responses:
            success, score_val = self._is_successful_response(response)
            if not success:
                continue
            normalized_score = score_val if score_val is not None else 0.0
            if (
                best_response is None
                or normalized_score > (best_score if best_score is not None else float("-inf"))
            ):
                best_response = response
                best_score = score_val
        return best_response, best_score

    def _is_successful_response(self, response: Dict[str, Any]) -> Tuple[bool, Optional[float]]:
        meta = response.get("meta") or {}
        eval_result = response.get("eval_result") or {}

        success_flags = [
            self._normalize_flag(meta.get("is_correct")),
            self._normalize_flag(meta.get("valid")),
            self._normalize_flag(meta.get("passed")),
            self._normalize_flag(meta.get("success")),
        ]

        success = any(flag is True for flag in success_flags)

        score_val = None
        if isinstance(eval_result, dict):
            if isinstance(eval_result.get("score"), (int, float)):
                score_val = float(eval_result["score"])
            else:
                for key in ("accuracy", "reward", "milestone_similarity", "similarity"):
                    if isinstance(eval_result.get(key), (int, float)):
                        score_val = float(eval_result[key])
                        break

            for key in ("passed", "success", "is_correct", "is_solved"):
                flag = self._normalize_flag(eval_result.get(key))
                if flag is not None:
                    success = success or flag

        if not success and score_val is not None:
            success = score_val > 0.0

        return success, score_val

    @staticmethod
    def _normalize_flag(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            if value == 0:
                return False
            if value == 1:
                return True
            return None
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "y", "pass", "passed", "correct", "success", "succeeded"}:
                return True
            if lowered in {"false", "no", "n", "fail", "failed", "incorrect"}:
                return False
        return None

    def _build_rebuttal_prompt(
        self,
        question: FormattedQuestion,
        supporting_response: Dict[str, Any],
        supporting_score: Optional[float],
    ) -> str:
        model_name = supporting_response.get("model_name") or supporting_response.get("model_path") or "the provided model"
        response_id = (supporting_response.get("meta") or {}).get("id")
        supporting_conversation = json.dumps(supporting_response.get("messages", []), ensure_ascii=False, indent=2)
        eval_result = supporting_response.get("eval_result") or {}
        eval_summary = (
            json.dumps(eval_result, indent=2, ensure_ascii=False) if eval_result else "No evaluation metadata available."
        )

        header_lines = [
            f"You previously judged benchmark sample '{question.question_id}' in benchmark '{question.benchmark.value}' to be flawed.",
            f"The model `{model_name}` completed this sample successfully according to the benchmark's reference evaluation.",
        ]
        if response_id:
            header_lines[-1] += f" (response id: {response_id})"
        if supporting_score is not None:
            header_lines.append(f"The recorded evaluation score for this run was {supporting_score:.4f}.")
        header_lines.append(
            "Review the successful agent trajectory below and reconcile it with your earlier judgement. "
        )

        scenario_guidance = [
            "While reassessing, weigh which explanation best fits the model success:",
            "1. Faulty ground truth replicated: The evaluation system would reward a model trajectory that replicates the error in the ground truth. Therefore, if you have judged a sample as flawed in your original response because of incorrect ground truth, the decision must be kept as flawed even if there is a trajectory that got reward.",
            "2. Missing context in your initial prompt — the sample is actually well-specified but you previously lacked information that the agent legitimately used (e.g., via tools). This should flip the sample to not flawed.",
            "3. Evaluation or user behaviour overly lenient — the scoring logic or user model lets incorrect behaviour pass. This keeps the sample flawed.",
            "If none apply, describe the alternative clearly. In your reasoning summary, cite which scenario (or alternative) you believe explains the success and justify how that impacts the final judgement.",
        ]

        body_lines = [
            "\n".join(header_lines),
            "",
            "\n".join(scenario_guidance),
            "",
            "Considering this evidence, reassess whether the benchmark sample is flawed. "
            "If you still conclude it is flawed, explicitly explain how the model nevertheless succeeded. "
            "Respond using the same JSON schema as your previous reply.",
            "Successful agent conversation:\n```json",
            supporting_conversation,
            "```\n",
            "Evaluation metadata:",
            eval_summary,
            "",
        ]
        return "\n".join(body_lines)

    def _format_model_transcript(
        self,
        messages: List[Dict[str, Any]],
        limit: int,
    ) -> str:
        if not messages:
            return "No agent transcript available."

        turns = []
        for turn in messages:
            role = turn.get("role", "unknown").upper()
            content = turn.get("content", "")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if "text" in item:
                            parts.append(str(item["text"]))
                        else:
                            parts.append(json.dumps(item, ensure_ascii=False))
                    else:
                        parts.append(str(item))
                text = "\n".join(parts)
            elif isinstance(content, dict):
                text = json.dumps(content, ensure_ascii=False)
            else:
                text = str(content)
            turns.append(f"{role}: {text}")

        transcript = "\n\n".join(turns)
        return self._truncate_text(transcript, limit)

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        if limit is None or limit <= 0 or len(text) <= limit:
            return text
        return text[:limit] + "\n...[truncated]"
    
    def assess_questions(self, questions: List[FormattedQuestion], step: LLMJudgeStep) -> List[Dict]:
        """Assess questions using multiprocessing."""
        if self.config.num_proc == 1:   # Single process
            results = []
            shared_client = None
            for question in tqdm(questions, desc="Processing questions"):
                if not question.skip_llm_judge and shared_client is None:
                    shared_client = self._create_client()
                assessment, evaluation_prompt = self._assess_question(
                    question, step, shared_client
                )
                entry = {
                    "benchmark": question.benchmark.value,
                    "question_id": question.question_id,
                    "assessment": assessment,
                    "prompt": evaluation_prompt,
                }
                entry = self._maybe_run_rebuttal(question, step, entry)
                results.append(entry)
                self._write_result_to_file(step, entry)
                print(question.question_id, entry["assessment"])
            return results
        else:   # Multiprocessing
            logger.info(f"Using multiprocessing with {self.config.num_proc} processes")
            
            args_list = []
            for idx, question in enumerate(questions):
                args_list.append((
                    idx, question, step, self.config.model, os.getenv("API_KEY"), os.getenv("BASE_URL"), 
                    self.config.max_retries, self.config.retry_delay, self.config.request_timeout
                ))
            
            results = [None] * len(args_list)
            with Pool(processes=self.config.num_proc) as pool:
                with tqdm(total=len(args_list), desc="Processing questions (multiprocessing)") as pbar:
                    for idx, result in pool.imap_unordered(_assess_question_worker, args_list):
                        results[idx] = result
                        pbar.update(1)
            
            for idx, entry in enumerate(results):
                if entry is None:
                    continue
                question = questions[idx]
                entry = self._maybe_run_rebuttal(question, step, entry)
                results[idx] = entry
                self._write_result_to_file(step, entry)
            
            return results


def _assess_question_worker(args):
    """Worker function for multiprocessing question assessment."""
    idx, question, step, model, api_key, base_url, max_retries, retry_delay, request_timeout = args
    client = OpenAI(api_key=api_key, base_url=base_url)
    evaluation_prompt = format_judge_prompt(question, step)

    if question.skip_llm_judge:
        assessment = copy.deepcopy(DUMMY_ASSESSMENT)
    else:
        assessment = LLMJudge._make_api_call(
            client,
            model,
            [{"role": "user", "content": evaluation_prompt}],
            max_retries,
            retry_delay,
            request_timeout,
        )
    entry = {
        "benchmark": question.benchmark.value,
        "question_id": question.question_id,
        "assessment": assessment,
        "prompt": evaluation_prompt,
    }
    return idx, entry
