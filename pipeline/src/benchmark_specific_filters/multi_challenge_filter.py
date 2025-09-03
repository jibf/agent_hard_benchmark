"""
MultiChallenge-specific rule-based filtering.
Implements custom filtering logic for MultiChallenge evaluation data.
Based on the detailed analysis of 4 main issue categories.
"""

from typing import Dict, List, Tuple
from .base_filter import BaseBenchmarkFilter
import logging
import re
import numpy as np

logger = logging.getLogger(__name__)

class MultiChallengeFilter(BaseBenchmarkFilter):
    """MultiChallenge-specific filtering rules."""
    
    def __init__(self):
        super().__init__("MultiChallenge")
    
    def get_filter_name(self) -> str:
        return "MultiChallenge-Specific Filter"
    
    def is_applicable(self, sample: Dict) -> bool:
        """Check if sample is from MultiChallenge."""
        # MultiChallenge samples typically have multi-turn conversations
        return (
            'conversation' in sample or 
            'messages' in sample or
            'turns' in sample
        )
    
    def filter_samples(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Apply MultiChallenge-specific filtering at the QUESTION/TASK level, not response level."""
        logger.info(f"Applying MultiChallenge-specific filtering to {len(samples)} samples")
        
        # Group samples by question/task
        question_groups = self._group_samples_by_question(samples)
        logger.info(f"Found {len(question_groups)} unique questions/tasks")
        
        # Filter questions (not individual responses) - VERY LENIENT
        good_questions = []
        bad_questions = []
        
        for question_id, question_samples in question_groups.items():
            if self._is_question_good_lenient(question_samples):
                good_questions.append(question_id)
            else:
                bad_questions.append(question_id)
        
        logger.info(f"Good questions: {len(good_questions)}")
        logger.info(f"Bad questions: {len(bad_questions)}")
        
        # Collect all samples for good questions (preserve all responses)
        passed_samples = []
        dropped_samples = []
        
        for question_id, question_samples in question_groups.items():
            if question_id in good_questions:
                passed_samples.extend(question_samples)
            else:
                dropped_samples.extend(question_samples)
        
        logger.info(f"Final results: {len(passed_samples)} samples passed, {len(dropped_samples)} dropped")
        
        self.log_filtering_stats(len(samples), len(passed_samples), len(dropped_samples))
        
        return passed_samples, dropped_samples
    
    def _group_samples_by_question(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """Group samples by question/task identifier."""
        question_groups = {}
        
        for sample in samples:
            question_id = self._extract_question_id(sample)
            if question_id not in question_groups:
                question_groups[question_id] = []
            question_groups[question_id].append(sample)
        return question_groups
    
    def _is_question_good_lenient(self, question_samples: List[Dict]) -> bool:
        """Evaluate if a question/task is good (VERY LENIENT approach)."""
        if not question_samples:
            return False
        
        # Use the first sample to evaluate the question structure
        sample = question_samples[0]
        
        # Check 1: Basic structure validation (very lenient)
        if not self._has_basic_structure(sample):
            return False
        
        # Check 2: Discriminativeness (keep this - it's from comprehensive filtering)
        if not self._is_question_discriminative(question_samples):
            return False
        
        # Check 3: Only drop if completely broken (very lenient)
        if self._is_completely_broken(sample):
            return False
        
        return True
    
    def _has_basic_structure(self, sample: Dict) -> bool:
        """Very basic structure check - only drop completely broken samples."""
        try:
            # Just check if we have the essential fields
            if not sample.get('messages') and not sample.get('conversation'):
                return False
            if not sample.get('eval_result', {}).get('score'):
                return False
            return True
        except:
            return False
    
    def _is_completely_broken(self, sample: Dict) -> bool:
        """Only detect completely broken questions (very lenient)."""
        try:
            messages = sample.get('messages', []) or sample.get('conversation', [])
            if not messages:
                return False
            
            # Only flag if the question is completely incomprehensible
            # (very lenient - only obvious cases)
            return False  # For now, be very lenient
        except:
            return False
    
    def _filter_by_structure(self, samples: List[Dict]) -> List[Dict]:
        """Filter by MultiChallenge-specific structure requirements."""
        valid_samples = []
        
        for sample in samples:
            if not self.is_applicable(sample):
                continue
            
            # Must have conversation or messages
            has_conversation = (
                'conversation' in sample and len(sample['conversation']) > 0
            ) or (
                'messages' in sample and len(sample['messages']) > 0
            )
            
            if not has_conversation:
                continue
            
            # Must have evaluation result
            if 'eval_result' not in sample:
                continue
            
            valid_samples.append(sample)
        
        return valid_samples
    
    def _filter_by_memory_failure(self, samples: List[Dict]) -> List[Dict]:
        """
        Filter out samples with memory failure issues.
        
        Detects:
        - Missing facts (distances, times, coordinates)
        - Broken turn structure
        - Vague thresholds without numeric data
        - Abrupt context shifts
        """
        valid_samples = []
        
        for sample in samples:
            conversation = self._extract_conversation(sample)
            if not conversation:
                continue
            
            # Check for missing factual information
            if self._has_missing_facts(conversation):
                continue
            
            # Check for broken turn structure
            if self._has_broken_structure(conversation):
                continue
            
            # Check for vague thresholds
            if self._has_vague_thresholds(conversation):
                continue
            
            valid_samples.append(sample)
        
        return valid_samples
    
    def _filter_by_instruction_violation(self, samples: List[Dict]) -> List[Dict]:
        """
        Filter out samples with instruction violation issues.
        
        Detects:
        - Vague formatting rules
        - Unclear scope/priorities
        - Conflicting instructions
        - Subjective constraints
        """
        valid_samples = []
        
        for sample in samples:
            conversation = self._extract_conversation(sample)
            if not conversation:
                continue
            
            # Check for vague formatting rules
            if self._has_vague_formatting(conversation):
                continue
            
            # Check for unclear scope
            if self._has_unclear_scope(conversation):
                continue
            
            # Check for conflicting instructions
            if self._has_conflicting_instructions(conversation):
                continue
            
            valid_samples.append(sample)
        
        return valid_samples
    
    def _filter_by_self_contradiction(self, samples: List[Dict]) -> List[Dict]:
        """
        Filter out samples with self-contradiction issues.
        
        Detects:
        - Imprecise coherence criteria
        - Embedded conflicts
        - Contradictory facts
        - Competing requirements
        """
        valid_samples = []
        
        for sample in samples:
            conversation = self._extract_conversation(sample)
            if not conversation:
                continue
            
            # Check for embedded conflicts
            if self._has_embedded_conflicts(conversation):
                continue
            
            # Check for contradictory facts
            if self._has_contradictory_facts(conversation):
                continue
            
            # Check for competing requirements
            if self._has_competing_requirements(conversation):
                continue
            
            valid_samples.append(sample)
        
        return valid_samples
    
    def _filter_by_version_confusion(self, samples: List[Dict]) -> List[Dict]:
        """
        Filter out samples with version confusion issues.
        
        Detects:
        - Missing authoritative state
        - Unclear change specifications
        - Competing final versions
        - Underspecified edit intent
        """
        valid_samples = []
        
        for sample in samples:
            conversation = self._extract_conversation(sample)
            if not conversation:
                continue
            
            # Check for missing authoritative state
            if self._has_missing_authoritative_state(conversation):
                continue
            
            # Check for unclear change specs
            if self._has_unclear_changes(conversation):
                continue
            
            # Check for competing versions
            if self._has_competing_versions(conversation):
                continue
            
            valid_samples.append(sample)
        
        return valid_samples
    
    def _filter_by_discriminativeness(self, samples: List[Dict]) -> List[Dict]:
        """
        Filter out non-discriminative questions based on score variance.
        This ensures questions help distinguish between different LLM capabilities.
        """
        # Group samples by question/task
        question_groups = {}
        for sample in samples:
            # Extract question identifier (could be conversation_id, task_id, etc.)
            question_id = self._extract_question_id(sample)
            if question_id not in question_groups:
                question_groups[question_id] = []
            question_groups[question_id].append(sample)
        
        # Filter discriminative questions
        discriminative_samples = []
        for question_id, question_samples in question_groups.items():
            if self._is_question_discriminative(question_samples):
                discriminative_samples.extend(question_samples)
        
        return discriminative_samples
    
    def _extract_question_id(self, sample: Dict) -> str:
        """Extract question identifier from sample."""
        # Try different possible question ID fields
        if 'conversation_id' in sample:
            return sample['conversation_id']
        elif 'task_id' in sample:
            return sample['task_id']
        elif 'id' in sample:
            return sample['id']
        elif 'question_id' in sample:
            return sample['question_id']
        else:
            # Fallback: use conversation content hash as identifier
            conversation = self._extract_conversation(sample)
            if conversation:
                content = str(conversation[:2])  # First two turns
                return str(hash(content))
            return str(hash(str(sample)))
    
    def _is_question_discriminative(self, question_samples: List[Dict]) -> bool:
        """
        Determine if a question is discriminative based on model performance variation.
        Returns True if the question helps distinguish between different LLM capabilities.
        """
        if len(question_samples) < 2:
            return False  # Need at least 2 model responses to compare
        
        # Extract scores for this question
        scores = []
        for sample in question_samples:
            # Try different possible score locations
            if 'eval_result' in sample and 'score' in sample['eval_result']:
                scores.append(sample['eval_result']['score'])
            elif 'eval_result' in sample and 'scores' in sample['eval_result']:
                scores.extend(sample['eval_result']['scores'])
            elif 'score' in sample:
                scores.append(sample['score'])
            elif 'scores' in sample:
                scores.extend(sample['scores'])
        
        if not scores:
            return False
        
        # Convert to numeric scores
        numeric_scores = []
        for score in scores:
            if isinstance(score, (int, float)):
                numeric_scores.append(float(score))
            elif isinstance(score, dict) and 'score' in score:
                try:
                    numeric_scores.append(float(score['score']))
                except (ValueError, TypeError):
                    continue
        
        if len(numeric_scores) < 2:
            return False
        
        # Calculate variance to measure discriminativeness
        variance = np.var(numeric_scores)
        
        # Question is discriminative if there's sufficient variance in scores
        # This means different models perform differently on this question
        return variance > 0.01  # Threshold for meaningful variation
    
    def _extract_conversation(self, sample: Dict) -> List[Dict]:
        """Extract conversation from sample."""
        if 'conversation' in sample:
            return sample['conversation']
        elif 'messages' in sample:
            return sample['messages']
        return []
    
    def _has_missing_facts(self, conversation: List[Dict]) -> bool:
        """Check if conversation is missing essential facts."""
        text = ' '.join([str(turn.get('content', '')) for turn in conversation])
        
        # Check for vague distance/time constraints without specific values
        distance_patterns = [
            r'within\s+\d+\s*minutes?',
            r'within\s+\d+\s*miles?',
            r'within\s+\d+\s*km',
            r'within\s+\d+\s*meters?'
        ]
        
        time_patterns = [
            r'within\s+\d+\s*hours?',
            r'within\s+\d+\s*days?',
            r'within\s+\d+\s*weeks?'
        ]
        
        # If we see constraints but no specific values, flag as missing facts
        has_constraints = any(re.search(pattern, text, re.IGNORECASE) for pattern in distance_patterns + time_patterns)
        
        if has_constraints:
            # Check if specific values are provided
            has_specific_values = re.search(r'\d+', text)
            return not has_specific_values
        
        return False
    
    def _has_broken_structure(self, conversation: List[Dict]) -> bool:
        """Check if conversation has broken turn structure."""
        if len(conversation) < 2:
            return True
        
        # Check for abrupt context shifts
        for i in range(1, len(conversation)):
            prev_turn = conversation[i-1]
            curr_turn = conversation[i]
            
            # Skip if turns are None or don't have content
            if not prev_turn or not curr_turn:
                continue
                
            prev_content = prev_turn.get('content', '')
            curr_content = curr_turn.get('content', '')
            
            # Check for extreme topic shifts
            if len(prev_content) > 50 and len(curr_content) > 50:
                # Simple heuristic: if consecutive turns have very different content lengths
                length_diff = abs(len(prev_content) - len(curr_content))
                if length_diff > len(prev_content) * 0.8:  # 80% difference
                    return True
        
        return False
    
    def _has_vague_thresholds(self, conversation: List[Dict]) -> bool:
        """Check if conversation has vague thresholds."""
        text = ' '.join([str(turn.get('content', '')) for turn in conversation])
        
        vague_patterns = [
            r'within\s+a\s+reasonable\s+distance',
            r'within\s+a\s+reasonable\s+time',
            r'not\s+too\s+far',
            r'not\s+too\s+long',
            r'close\s+by',
            r'nearby'
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in vague_patterns)
    
    def _has_vague_formatting(self, conversation: List[Dict]) -> bool:
        """Check if conversation has vague formatting rules."""
        text = ' '.join([str(turn.get('content', '')) for turn in conversation])
        
        vague_formatting = [
            r'no\s+bold\s+anywhere',
            r'no\s+formatting',
            r'keep\s+it\s+simple',
            r'use\s+plain\s+text',
            r'no\s+special\s+characters'
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in vague_formatting)
    
    def _has_unclear_scope(self, conversation: List[Dict]) -> bool:
        """Check if conversation has unclear scope."""
        text = ' '.join([str(turn.get('content', '')) for turn in conversation])
        
        unclear_scope = [
            r'and\s+so\s+on',
            r'etc\.',
            r'and\s+similar',
            r'related\s+topics',
            r'other\s+relevant'
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in unclear_scope)
    
    def _has_conflicting_instructions(self, conversation: List[Dict]) -> bool:
        """Check if conversation has conflicting instructions."""
        text = ' '.join([str(turn.get('content', '')) for turn in conversation])
        
        # Check for contradictory pairs
        contradictions = [
            (r'be\s+detailed', r'keep\s+it\s+brief'),
            (r'be\s+formal', r'be\s+casual'),
            (r'include\s+examples', r'no\s+examples'),
            (r'use\s+technical\s+language', r'use\s+simple\s+language')
        ]
        
        for pattern1, pattern2 in contradictions:
            if re.search(pattern1, text, re.IGNORECASE) and re.search(pattern2, text, re.IGNORECASE):
                return True
        
        return False
    
    def _has_embedded_conflicts(self, conversation: List[Dict]) -> bool:
        """Check if conversation has embedded conflicts."""
        text = ' '.join([str(turn.get('content', '')) for turn in conversation])
        
        conflict_patterns = [
            r'both\s+yes\s+and\s+no',
            r'contradicts\s+earlier',
            r'opposite\s+of',
            r'inconsistent\s+with'
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in conflict_patterns)
    
    def _has_contradictory_facts(self, conversation: List[Dict]) -> bool:
        """Check if conversation has contradictory facts."""
        text = ' '.join([str(turn.get('content', '')) for turn in conversation])
        
        # Check for date/range contradictions
        dates = re.findall(r'\d{4}-\d{2}-\d{2}', text)
        if len(dates) > 1:
            # Check if dates are in chronological order
            sorted_dates = sorted(dates)
            if dates != sorted_dates:
                return True
        
        return False
    
    def _has_competing_requirements(self, conversation: List[Dict]) -> bool:
        """Check if conversation has competing requirements."""
        text = ' '.join([str(turn.get('content', '')) for turn in conversation])
        
        competing_patterns = [
            r'on\s+one\s+hand.*on\s+the\s+other\s+hand',
            r'however.*but',
            r'although.*nevertheless',
            r'despite.*still'
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in competing_patterns)
    
    def _has_missing_authoritative_state(self, conversation: List[Dict]) -> bool:
        """Check if conversation is missing authoritative state."""
        text = ' '.join([str(turn.get('content', '')) for turn in conversation])
        
        # Check for editing tasks without clear final state
        editing_keywords = ['edit', 'modify', 'change', 'update', 'revise']
        has_editing = any(keyword in text.lower() for keyword in editing_keywords)
        
        if has_editing:
            # Check for clear final state indicators
            final_state_indicators = [
                'final version',
                'latest version',
                'updated version',
                'revised version',
                'final result'
            ]
            
            has_final_state = any(indicator in text.lower() for indicator in final_state_indicators)
            return not has_final_state
        
        return False
    
    def _has_unclear_changes(self, conversation: List[Dict]) -> bool:
        """Check if conversation has unclear change specifications."""
        text = ' '.join([str(turn.get('content', '')) for turn in conversation])
        
        unclear_changes = [
            r'make\s+it\s+better',
            r'improve\s+it',
            r'fix\s+it',
            r'change\s+as\s+needed',
            r'update\s+appropriately'
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in unclear_changes)
    
    def _has_competing_versions(self, conversation: List[Dict]) -> bool:
        """Check if conversation has competing versions."""
        text = ' '.join([str(turn.get('content', '')) for turn in conversation])
        
        # Check for multiple "final" or "latest" versions
        final_indicators = re.findall(r'(?:final|latest|updated|revised)\s+version', text, re.IGNORECASE)
        
        if len(final_indicators) > 1:
            return True
        
        return False
