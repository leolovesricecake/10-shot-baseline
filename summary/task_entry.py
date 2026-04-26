"""Summarization task registry and processor."""

from __future__ import annotations

import difflib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from baseline_task_base import BaseTaskProcessor, LabelSchema, compute_rouge_fallback, mean


class SummarizationTaskProcessor(BaseTaskProcessor):
    task_name = "summarization"
    metric_columns = ("acc", "rouge1", "rouge2", "rougeL")

    def build_zero_shot_prompt(
        self, input_text: str, dataset_cfg: Dict[str, Any], label_schema: Optional[LabelSchema]
    ) -> str:
        return (
            "You are a summarization assistant.\n"
            "Write a concise and faithful summary for the input text.\n"
            "Think step by step internally.\n"
            "Output only the final summary.\n\n"
            f"Input:\n{input_text}\n\n"
            "Summary:"
        )

    def build_few_shot_prompt(
        self,
        input_text: str,
        in_context: str,
        dataset_cfg: Dict[str, Any],
        label_schema: Optional[LabelSchema],
    ) -> str:
        return (
            "You are a summarization assistant.\n"
            "Learn from the examples and summarize the target document.\n"
            "Output only the final summary.\n\n"
            f"{in_context}\n\n"
            f"Target document:\n{input_text}\n\n"
            "Summary:"
        )

    def build_verification_prompt(
        self,
        input_text: str,
        initial_raw: str,
        dataset_cfg: Dict[str, Any],
        label_schema: Optional[LabelSchema],
    ) -> str:
        return (
            "You are verifying a summary.\n"
            "Check whether the previous summary is faithful and concise.\n"
            "Output only the improved final summary.\n\n"
            f"Document:\n{input_text}\n\n"
            f"Previous summary:\n{initial_raw}\n\n"
            "Final summary:"
        )

    def parse_prediction(
        self, raw_text: Any, dataset_cfg: Dict[str, Any], label_schema: Optional[LabelSchema]
    ) -> Any:
        return str(raw_text).strip() if raw_text is not None else ""

    def evaluate(self, truths: Sequence[Any], predictions: Sequence[Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        references = [str(value) for value in truths]
        hypotheses = [str(value) for value in predictions]
        backend = "fallback"
        try:
            from rouge_score import rouge_scorer  # type: ignore

            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
            rouge1_values: List[float] = []
            rouge2_values: List[float] = []
            rouge_l_values: List[float] = []
            for reference, hypothesis in zip(references, hypotheses):
                score = scorer.score(reference, hypothesis)
                rouge1_values.append(score["rouge1"].fmeasure)
                rouge2_values.append(score["rouge2"].fmeasure)
                rouge_l_values.append(score["rougeL"].fmeasure)
            rouge1 = mean(rouge1_values)
            rouge2 = mean(rouge2_values)
            rouge_l = mean(rouge_l_values)
            backend = "rouge_score"
        except Exception:
            rouge_metrics = compute_rouge_fallback(hypotheses, references)
            rouge1 = rouge_metrics["rouge1"]
            rouge2 = rouge_metrics["rouge2"]
            rouge_l = rouge_metrics["rougeL"]
        return {"acc": rouge1, "rouge1": rouge1, "rouge2": rouge2, "rougeL": rouge_l}, {"metric_backend": backend}

    def format_demonstration(
        self,
        train_text: str,
        train_label: Any,
        dataset_cfg: Dict[str, Any],
        label_schema: Optional[LabelSchema],
    ) -> str:
        return f"document: {train_text}\nsummary: {train_label}"

    def aggregate_sc_predictions(
        self,
        parsed_predictions: Sequence[Any],
        raw_predictions: Sequence[str],
        dataset_cfg: Dict[str, Any],
        label_schema: Optional[LabelSchema],
    ) -> Tuple[Any, str]:
        valid_pairs = [
            (str(parsed).strip(), str(raw))
            for parsed, raw in zip(parsed_predictions, raw_predictions)
            if parsed is not None and str(parsed).strip()
        ]
        if len(valid_pairs) == 0:
            raw_value = raw_predictions[0] if len(raw_predictions) > 0 else ""
            return self.parse_prediction(raw_value, dataset_cfg, label_schema), raw_value

        texts = [item[0] for item in valid_pairs]
        best_index = 0
        best_score = -1.0
        for idx, item in enumerate(texts):
            scores = []
            for jdx, other in enumerate(texts):
                if idx == jdx:
                    continue
                scores.append(difflib.SequenceMatcher(None, item, other).ratio())
            score = mean(scores) if len(scores) > 0 else 0.0
            if score > best_score:
                best_score = score
                best_index = idx
        return valid_pairs[best_index][0], valid_pairs[best_index][1]


TASK_PROCESSOR = SummarizationTaskProcessor()


TASK_DATASETS: Dict[str, Dict[str, Any]] = {
    "gigaword": {
        "task": "summarization",
        "task_group": "summary",
        "train_file": "gigaword_train.csv",
        "test_file": "gigaword_test.csv",
    },
}
