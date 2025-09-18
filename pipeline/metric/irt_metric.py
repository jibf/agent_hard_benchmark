#!/usr/bin/env python3
"""
IRT (Item Response Theory) metric for benchmark evaluation.
Uses 2-Parameter Logistic model to estimate item discrimination parameters.
"""

import torch
import numpy as np
import logging
from typing import Dict
from py_irt.models import two_param_logistic
from pyro.infer import MCMC, NUTS

logger = logging.getLogger(__name__)


class IRTMetric:
    """IRT metric for evaluating item discrimination in benchmark data."""

    def __init__(self, device: str = "cuda", mcmc_samples: int = 1000, warmup_steps: int = 100, threshold: float=0.5):
        """
        Args:
            device: Device to run computation on ("cuda" or "cpu")
            mcmc_samples: Number of MCMC samples for parameter estimation
            warmup_steps: Number of warmup steps for MCMC
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.mcmc_samples = mcmc_samples
        self.warmup_steps = warmup_steps
        self.threshold = threshold


    def compute_irt_discrimination(self, responses_by_question: Dict) -> Dict:
        """
        Compute IRT discrimination parameters and return alpha values for each question.

        Args:
            responses_by_question: Dict with question_id as key and list of model responses as value

        Returns:
            Dict with question_id as key and alpha (discrimination) value as value
        """
        logger.info("Computing IRT discrimination parameters...")

        # Prepare data
        question_ids = list(responses_by_question.keys())
        num_items = len(question_ids)

        if num_items == 0:
            logger.warning("No questions found in responses_by_question")
            return {}

        # Create mapping from question_id to item index
        question_to_item = {q_id: idx for idx, q_id in enumerate(question_ids)}

        # Extract model names and create subject mapping
        all_model_names = set()
        for responses in responses_by_question.values():
            for response in responses:
                all_model_names.add(response["model_name"])
            break

        model_names = sorted(list(all_model_names))
        num_subjects = len(model_names)
        model_to_subject = {model: idx for idx, model in enumerate(model_names)}

        if num_subjects == 0:
            logger.warning("No models found in responses")
            return {}

        logger.info(f"IRT data: {num_items} items, {num_subjects} subjects")

        # Convert responses to binary format and collect data
        responses_data = []
        subject_indices = []
        item_indices = []

        for question_id, responses in responses_by_question.items():
            item_idx = question_to_item[question_id]

            for response in responses:
                model_name = response["model_name"]
                subject_idx = model_to_subject[model_name]

                # Extract score and binarize
                score = response["eval_result"]["score"]
                binary_response = 1 if score > self.threshold else 0

                responses_data.append(binary_response)
                subject_indices.append(subject_idx)
                item_indices.append(item_idx)

        if len(responses_data) == 0:
            logger.warning("No valid responses found for IRT analysis")
            return {}

        # Convert to tensors
        subjects_tensor = torch.tensor(subject_indices, dtype=torch.long, device=self.device)
        items_tensor = torch.tensor(item_indices, dtype=torch.long, device=self.device)
        responses_tensor = torch.tensor(responses_data, dtype=torch.float, device=self.device)

        # Fit IRT model
        try:
            model = two_param_logistic.TwoParamLog(
                priors="vague",
                num_items=num_items,
                num_subjects=num_subjects,
                device=self.device
            )

            nuts_kernel = NUTS(model.model_vague, adapt_step_size=True)
            mcmc = MCMC(nuts_kernel, num_samples=self.mcmc_samples, warmup_steps=self.warmup_steps)
            mcmc.run(subjects_tensor, items_tensor, responses_tensor)

            # Get samples and extract discrimination parameters
            samples = mcmc.get_samples()

            if 'a' not in samples:
                logger.warning("Discrimination parameters (a) not found in MCMC samples")
                return {}

            # Calculate alpha values for each item across MCMC samples
            a_samples = samples['a'].cpu().numpy()
            alpha_means = np.mean(a_samples, axis=0)  # Average across MCMC samples for each item

            # Create mapping from item index back to question_id
            item_to_question = {idx: q_id for q_id, idx in question_to_item.items()}
            
            # Return dict with question_id as key and alpha value as value
            alpha_dict = {}
            for item_idx, alpha_value in enumerate(alpha_means):
                question_id = item_to_question[item_idx]
                alpha_dict[question_id] = float(alpha_value)

            return alpha_dict

        except Exception as e:
            logger.error(f"Error during IRT model fitting: {e}")
            return {}


def compute_irt_metric(responses_by_question: Dict, device: str = "cuda", threshold: float=0.5) -> Dict:
    """
    Convenience function to compute IRT discrimination metric.

    Args:
        responses_by_question: Dict with question_id as key and list of model responses as value
        device: Device to run computation on
        threshold: Threshold for binarizing scores

    Returns:
        Dict with question_id as key and alpha (discrimination) value as value
    """
    irt_metric = IRTMetric(device=device, threshold=threshold)
    return irt_metric.compute_irt_discrimination(responses_by_question)