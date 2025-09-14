"""
TAU Bench-specific rule-based filtering.
Implements custom filtering logic for TAU Bench evaluation data.
"""

from typing import Dict, List, Tuple
from .base_filter import BaseBenchmarkFilter
import logging
import numpy as np
from collections import defaultdict
from src.bench_loaders.tau2_bench_loader import Tau2BenchLoader

logger = logging.getLogger(__name__)

class TAU2BenchFilter(BaseBenchmarkFilter):
    """TAU2 Bench-specific filtering rules."""
    
    def __init__(self):
        super().__init__("TAU2 Bench")
    
    def get_filter_name(self) -> str:
        return "TAU2 Bench-Specific Filter"
    
    def is_applicable(self, sample: Dict) -> bool:
        """Check if sample is from TAU2 Bench."""
        # TODO: Implement proper detection logic
        return True
    
    def filter_samples(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Apply TAU2 Bench-specific filtering rules.
        Note: Comprehensive filtering has already been applied before this method is called.
        """
        logger.info(f"Applying TAU2 Bench-specific filtering to {len(samples)} samples")
        
        # STEP 1: Filter questions solvable by a trivial agent, but keep those with success rate <= 0.5
        DO_NOTHING_SUCCESS_RATE_THRESHOLD = 0.5
        qids_to_filter = self._get_qids_solvable_by_do_nothing()
        question_groups = self._group_samples_by_question(samples)
        
        passed_samples = []
        dropped_samples = []
        
        for sample in samples:
            qid = f"{sample['task_name']}-{sample['meta']['id']}"
            mean_score = self._calculate_mean_score(question_groups[qid])
            if qid not in qids_to_filter or mean_score <= DO_NOTHING_SUCCESS_RATE_THRESHOLD:
                passed_samples.append(sample)
            else:
                dropped_samples.append(sample)
        
        logger.info(f"TAU2 Bench filtering completed: {len(passed_samples)} passed, {len(dropped_samples)} dropped")
        return passed_samples, dropped_samples


    def _get_qids_solvable_by_do_nothing(self) -> List[str]:
        function_names_modifying_database = {
            'retail': ['cancel_pending_order', "exchange_delivered_order_items", "modify_pending_order_address", "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address", "return_delivered_order_items"],
            'airline': ["book_reservation", "cancel_reservation", "send_certificate", "update_reservation_baggages", "update_reservation_flights", "update_reservation_passengers"],
            'telecom': ["make_payment", "resume_line", "refuel_data", "send_payment_request"]
        }
        
        result = []
        questions = Tau2BenchLoader().load_questions() # only the questions, not responses
        for question in questions:
            qid = question.question_id
            
            domain, _ = get_domain_and_id(qid)

            is_modifying_database = False
            if domain in function_names_modifying_database:
                for message in question.gt_conv_traj:
                    if message['role'] == 'assistant' and 'function_call' in message.keys():
                        for function_call in message['function_call']:
                            if function_call['name'] in function_names_modifying_database[domain]:
                                is_modifying_database = True

            if len(question.evaluation_criteria.get('actions', [])) == 0 and not is_modifying_database:
                result.append(qid)

        return result

    def _group_samples_by_question(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """Group samples by their question identifier."""
        question_groups = defaultdict(list)
        
        for sample in samples:
            qid = f"{sample['task_name']}-{sample['meta']['id']}"
            question_groups[qid].append(sample)
        
        return dict(question_groups)

    def _calculate_mean_score(self, question_samples: List[Dict]) -> float:
        """Calculate mean score for a question based on all model responses."""
        if not question_samples:
            return None
        
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
            return None
        
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
        
        if not numeric_scores:
            return None
        
        # Calculate mean score (success rate)
        return np.mean(numeric_scores)


def get_domain_and_id(question_id: str) -> Tuple[str, int]:
    domain, id_str = question_id.split("-")
    return domain, int(id_str)