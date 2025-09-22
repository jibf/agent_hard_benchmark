"""ACEBench duplicate question filtering."""

from typing import Dict, List, Tuple
from .base_filter import BaseBenchmarkFilter
import logging
import json

logger = logging.getLogger(__name__)


class ACEBenchFilter(BaseBenchmarkFilter):
    """Remove duplicate ACEBench questions while leaving other samples untouched."""

    def __init__(self):
        super().__init__("ACEBench")

    def get_filter_name(self) -> str:
        return "ACEBench Duplicate Question Filter"

    def filter_samples(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        logger.info(
            f"Applying ACEBench duplicate question filtering to {len(samples)} samples"
        )

        seen_models_by_question: Dict[str, set] = {}
        passed_samples: List[Dict] = []
        dropped_samples: List[Dict] = []

        for sample in samples:
            question_key = self._build_question_key(sample)
            model_key = self._extract_model_key(sample)

            models_for_question = seen_models_by_question.setdefault(question_key, set())
            if model_key in models_for_question:
                dropped_samples.append(sample)
                continue

            models_for_question.add(model_key)
            passed_samples.append(sample)

        self.log_filtering_stats(
            len(samples), len(passed_samples), len(dropped_samples)
        )
        return passed_samples, dropped_samples

    def _build_question_key(self, sample: Dict) -> str:
        prompt_key = self._extract_prompt_key(sample)
        if prompt_key:
            task_name = sample.get("task_name")
            if task_name:
                normalised_task = self._normalise_text(str(task_name))
                if normalised_task:
                    return f"{normalised_task}||{prompt_key}"
            return prompt_key

        meta = sample.get("meta")
        if isinstance(meta, dict):
            question_id = meta.get("id") or meta.get("question_id")
            if question_id:
                return str(question_id)

        instruction = sample.get("instruction")
        if instruction:
            return self._normalise_text(str(instruction))

        return json.dumps(sample, sort_keys=True, ensure_ascii=False)

    def _extract_prompt_key(self, sample: Dict) -> str:
        fragments: List[str] = []

        question_text = sample.get("question")
        if question_text:
            fragments.append(self._normalise_text(str(question_text)))

        messages = sample.get("messages")
        if isinstance(messages, list):
            fragments.extend(self._extract_messages(messages))

        conversation = sample.get("conversation")
        if isinstance(conversation, list):
            fragments.extend(self._extract_messages(conversation))

        if not fragments:
            return ""

        joined = " ".join(fragment for fragment in fragments if fragment)
        return self._normalise_text(joined)

    def _extract_messages(self, messages: List[Dict]) -> List[str]:
        fragments: List[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role not in {"system", "user"}:
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                fragments.append(self._normalise_text(content))
        return fragments

    def _extract_model_key(self, sample: Dict) -> str:
        model_name = sample.get("model_name")
        if model_name:
            return str(model_name)

        response = sample.get("response")
        if isinstance(response, dict):
            model = response.get("model") or response.get("model_name")
            if model:
                return str(model)

        return "__unknown_model__"

    def _normalise_text(self, text: str) -> str:
        return " ".join(text.split())
