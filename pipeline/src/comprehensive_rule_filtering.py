#!/usr/bin/env python3
"""
Comprehensive rule-based filtering module.
Combines sample-level and question-level filtering into a single pipeline.
"""

import json
import hashlib
import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class ComprehensiveRuleFilter:
    """Comprehensive rule-based filtering that combines sample and question-level filtering."""
    
    def __init__(self):
        self.prompt_hashes = set()  # Track seen prompt hashes for duplicate detection
        
    def filter_samples(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Apply comprehensive rule-based filtering (sample + question level)."""
        logger.info("Starting comprehensive rule-based filtering")
        
        # Step 1: Sample-level filtering
        logger.info("Step 1: Applying sample-level filtering...")
        sample_pass_samples, sample_dropped_samples = self._sample_level_filtering(samples)
        logger.info(f"Sample filtering: {len(sample_pass_samples)}/{len(samples)} samples passed")
        
        # Step 2: Question-level filtering
        logger.info("Step 2: Applying question-level filtering...")
        final_passed, question_dropped = self._question_level_filtering(sample_pass_samples)
        
        # Combine results
        final_dropped = sample_dropped_samples + question_dropped
        
        logger.info("=== Comprehensive Rule-Based Filtering Results ===")
        logger.info(f"Total samples: {len(samples)}")
        logger.info(f"Sample-level dropped: {len(sample_dropped_samples)}")
        logger.info(f"Question-level dropped: {len(question_dropped)}")
        logger.info(f"Final passed: {len(final_passed)} ({len(final_passed)/len(samples)*100:.1f}%)")
        logger.info(f"Final dropped: {len(final_dropped)} ({len(final_dropped)/len(samples)*100:.1f}%)")
        
        return final_passed, final_dropped
    
    def _sample_level_filtering(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Apply sample-level rule-based filtering."""
        passed_samples = []
        dropped_samples = []
        
        for sample in samples:
            # Apply sample-level rules
            passes, reason = self._check_sample_rules(sample)
            
            if passes:
                passed_samples.append(sample)
            else:
                dropped_samples.append(sample)
        
        return passed_samples, dropped_samples
    
    def _check_sample_rules(self, sample: Dict) -> Tuple[bool, str]:
        """Apply sample-level rule-based filtering to a single sample."""
        
        # 1. Structure sanity
        if not self._check_structure(sample):
            return False, "invalid_structure"
        
        # 2. Conversation length sanity
        if not self._check_conversation_length(sample):
            return False, "invalid_conversation_length"
        
        # 3. Scoring sanity
        if not self._check_score_sanity(sample):
            return False, "invalid_score"
        
        # 4. Obvious broken samples
        if not self._check_obvious_broken(sample):
            return False, "obviously_broken"
        
        # 5. Duplicate detection
        if self._is_duplicate(sample):
            return False, "duplicate"
        
        return True, "passed"
    
    def _check_structure(self, sample: Dict) -> bool:
        """Check basic structure sanity."""
        try:
            # Must have messages
            messages = sample.get("messages", [])
            if not isinstance(messages, list) or len(messages) == 0:
                return False
            
            # Must have eval_result with score
            eval_result = sample.get("eval_result", {})
            if not isinstance(eval_result, dict):
                return False
            
            score = eval_result.get("score")
            if score is None:
                return False
            
            # Must have at least one user message and one assistant message
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
            
            if len(user_messages) == 0 or len(assistant_messages) == 0:
                return False
            
            # Must have non-empty final model reply
            final_assistant = assistant_messages[-1]
            content = final_assistant.get("content", "")
            if not content or content.strip() == "":
                return False
            
            return True
            
        except Exception:
            return False
    
    def _check_conversation_length(self, sample: Dict) -> bool:
        """Check conversation length sanity."""
        messages = sample.get("messages", [])
        
        # Conversation length: 2-80 messages
        if len(messages) < 2 or len(messages) > 80:
            return False
        
        # Token count check (rough estimate)
        total_text = ""
        for msg in messages:
            content = msg.get("content", "")
            if content:
                # Handle both string and list content
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    content = " ".join(text_parts)
                
                if isinstance(content, str):
                    total_text += content + " "
        
        # Rough token estimation: 1 token ≈ 4 characters
        estimated_tokens = len(total_text) // 4
        if estimated_tokens > 8000:  # Safe cap at 8k tokens
            return False
        
        return True
    
    def _check_score_sanity(self, sample: Dict) -> bool:
        """Check score sanity."""
        eval_result = sample.get("eval_result", {})
        score = eval_result.get("score")
        
        if not isinstance(score, (int, float)):
            return False
        
        # Check if score is within known ranges
        if score < 0:
            return False
        
        # Allow scores up to 100 (DrafterBench uses 0-100)
        if score > 100:
            return False
        
        return True
    
    def _check_obvious_broken(self, sample: Dict) -> bool:
        """Check for obviously broken samples."""
        messages = sample.get("messages", [])
        
        # Get final assistant message
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        if not assistant_messages:
            return True  # Broken if no assistant messages
        
        final_reply = assistant_messages[-1].get("content", "")
        
        # Check for degenerate model output
        if self._is_degenerate_output(final_reply):
            return False
        
        # Check for tool requirements (if applicable)
        if not self._check_tool_requirements(sample):
            return False
        
        return True
    
    def _is_degenerate_output(self, text: str) -> bool:
        """Check for degenerate model output."""
        if not text or len(text.strip()) == 0:
            return True
        
        # Check for single repeated token >70% of reply
        words = text.split()
        if len(words) > 10:  # Only check if reply is substantial
            word_counts = defaultdict(int)
            for word in words:
                word_counts[word] += 1
            
            most_common_count = max(word_counts.values())
            if most_common_count / len(words) > 0.7:
                return True
        
        # Check for empty code blocks only
        if re.match(r'^```\s*\n\s*```\s*$', text.strip()):
            return True
        
        # Check for boilerplate refusals
        refusal_patterns = [
            r"I cannot help with that",
            r"I'm sorry, I cannot",
            r"I'm unable to",
            r"I don't have access to",
            r"I cannot provide"
        ]
        
        text_lower = text.lower()
        for pattern in refusal_patterns:
            if re.search(pattern.lower(), text_lower):
                return True
        
        return False
    
    def _check_tool_requirements(self, sample: Dict) -> bool:
        """Check if tool requirements are met (if applicable)."""
        # This is a simplified check - we'll implement more sophisticated tool detection later
        messages = sample.get("messages", [])
        
        # Check if any assistant message has tool calls
        has_tool_calls = False
        for msg in messages:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                has_tool_calls = True
                break
        
        # For now, we'll be lenient and not drop samples based on tool requirements
        # This can be enhanced later with more sophisticated detection
        return True
    
    def _is_duplicate(self, sample: Dict) -> bool:
        """Check for duplicates based on prompt hash + model reply."""
        messages = sample.get("messages", [])
        
        # Extract user prompt (first user message)
        user_prompt = ""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content:
                    # Handle both string and list content
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
        
        # Get final assistant reply
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        final_reply = assistant_messages[-1].get("content", "") if assistant_messages else ""
        
        # Create hash of prompt + reply
        combined_text = user_prompt + "|||" + final_reply
        combined_hash = hashlib.md5(combined_text.encode()).hexdigest()
        
        if combined_hash in self.prompt_hashes:
            return True
        
        self.prompt_hashes.add(combined_hash)
        return False
    
    def _question_level_filtering(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Apply question-level filtering based on multi-model performance."""
        # Group samples by question
        question_groups = self._group_samples_by_question(samples)
        logger.info(f"Found {len(question_groups)} unique questions")
        
        # Analyze each question and decide to keep or drop
        kept_questions = []
        dropped_questions = []
        
        for question_id, question_samples in question_groups.items():
            keep_question, reason = self._analyze_question(question_id, question_samples)
            
            if keep_question:
                kept_questions.extend(question_samples)
            else:
                dropped_questions.extend(question_samples)
        
        return kept_questions, dropped_questions
    
    def _group_samples_by_question(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """Group samples by question ID."""
        question_groups = defaultdict(list)
        
        for sample in samples:
            question_id = self._compute_question_id(sample)
            question_groups[question_id].append(sample)
        
        return question_groups
    
    def _compute_question_id(self, sample: Dict) -> str:
        """Compute unique question ID based on prompt content and task."""
        messages = sample.get("messages", [])
        
        # Extract user prompt (first user message)
        user_prompt = ""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content:
                    # Handle both string and list content
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
        
        # Create question ID from prompt + task + benchmark
        task_name = sample.get("task_name", "unknown")
        benchmark_name = sample.get("benchmark_name", "unknown")
        question_text = f"{benchmark_name}|||{task_name}|||{user_prompt}"
        question_id = hashlib.md5(question_text.encode()).hexdigest()
        
        return question_id
    
    def _analyze_question(self, question_id: str, question_samples: List[Dict]) -> Tuple[bool, str]:
        """Analyze a question and decide whether to keep it based on multi-model performance."""
        if len(question_samples) < 2:
            return False, "insufficient_models"
        
        # Extract scores for all models on this question
        scores = []
        for sample in question_samples:
            eval_result = sample.get("eval_result", {})
            score = eval_result.get("score")
            if score is not None and isinstance(score, (int, float)):
                scores.append(score)
        
        if len(scores) < 2:
            return False, "insufficient_scores"
        
        # Analyze score characteristics
        unique_scores = set(scores)
        num_unique = len(unique_scores)
        
        if num_unique == 1:
            # No variation at all - drop immediately
            return False, "no_variation"
        
        elif num_unique == 2:
            # Binary scores - check if there's meaningful variation
            score_list = list(unique_scores)
            if 0 in score_list and 1 in score_list:
                # Binary pass/fail - keep if there's variation
                return True, "binary_with_variation"
            else:
                # Binary but not 0/1 - keep
                return True, "binary_other"
        
        else:
            # For continuous scores, apply variance-based filtering
            # Normalize DrafterBench scores to 0-1 range
            max_score = max(scores)
            if max_score > 10:  # DrafterBench uses 0-100 scale
                normalized_scores = [s / 100.0 for s in scores]
            else:
                normalized_scores = scores
            
            variance = np.var(normalized_scores)
            mean_score = np.mean(normalized_scores)
            
            # Apply variance-based filtering rules
            if variance < 0.005:  # Too easy - all models get similar scores
                return False, "too_easy_low_variance"
            elif variance > 0.8:  # Too hard - high variance indicates inconsistent performance
                return False, "too_hard_high_variance"
            elif mean_score > 0.98:  # Too easy - almost perfect scores
                return False, "too_easy_high_mean"
            elif mean_score < 0.02:  # Too hard - almost all failures
                return False, "too_hard_low_mean"
            else:
                return True, "good_variance_and_mean"
