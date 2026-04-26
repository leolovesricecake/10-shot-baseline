"""Similarity regression task registry and processor."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from baseline_task_base import (
    BaseTaskProcessor,
    LabelSchema,
    extract_first_float,
    mean,
    pearson_correlation,
    spearman_correlation,
    try_float,
)


class RegressionTaskProcessor(BaseTaskProcessor):
    task_name = "regression"
    metric_columns = ("mse", "rmse", "pearson_coeff", "spearman_coeff")

    def _score_min(self, dataset_cfg: Dict[str, Any]) -> float:
        return float(dataset_cfg.get("score_min", 0.0))

    def _score_max(self, dataset_cfg: Dict[str, Any]) -> float:
        return float(dataset_cfg.get("score_max", 5.0))

    def _score_fallback(self, dataset_cfg: Dict[str, Any]) -> float:
        return float(dataset_cfg.get("score_fallback", 2.5))

    def _clamp(self, value: float, dataset_cfg: Dict[str, Any]) -> float:
        return min(max(value, self._score_min(dataset_cfg)), self._score_max(dataset_cfg))

    def build_zero_shot_prompt(
        self, input_text: str, dataset_cfg: Dict[str, Any], label_schema: Optional[LabelSchema]
    ) -> str:
        score_min = self._score_min(dataset_cfg)
        score_max = self._score_max(dataset_cfg)
        return (
            "You are asked to score semantic similarity for an input text pair.\n"
            f"Output one numeric score between {score_min} and {score_max}.\n"
            "Higher score means stronger semantic similarity.\n"
            "Think step by step internally.\n"
            "Output only the final numeric score.\n\n"
            f"Text pair:\n{input_text}\n\n"
            "Score:"
        )

    def build_few_shot_prompt(
        self,
        input_text: str,
        in_context: str,
        dataset_cfg: Dict[str, Any],
        label_schema: Optional[LabelSchema],
    ) -> str:
        score_min = self._score_min(dataset_cfg)
        score_max = self._score_max(dataset_cfg)
        return (
            "You are asked to score semantic similarity for an input text pair.\n"
            f"Output one numeric score between {score_min} and {score_max}.\n"
            "Learn from the examples and then score the target pair.\n"
            "Think step by step internally.\n"
            "Output only the final numeric score.\n\n"
            f"{in_context}\n\n"
            f"Target text pair:\n{input_text}\n\n"
            "Score:"
        )

    def build_verification_prompt(
        self,
        input_text: str,
        initial_raw: str,
        dataset_cfg: Dict[str, Any],
        label_schema: Optional[LabelSchema],
    ) -> str:
        score_min = self._score_min(dataset_cfg)
        score_max = self._score_max(dataset_cfg)
        return (
            "You are verifying a semantic similarity score.\n"
            f"Valid range: {score_min} to {score_max}.\n"
            "Check whether the previous score is reasonable and output one final numeric score.\n"
            "Do not output explanations.\n\n"
            f"Text pair:\n{input_text}\n\n"
            f"Previous score:\n{initial_raw}\n\n"
            "Final score:"
        )

    def parse_prediction(
        self, raw_text: Any, dataset_cfg: Dict[str, Any], label_schema: Optional[LabelSchema]
    ) -> Any:
        score = extract_first_float(raw_text)
        if score is None:
            score = self._score_fallback(dataset_cfg)
        return self._clamp(float(score), dataset_cfg)

    def evaluate(self, truths: Sequence[Any], predictions: Sequence[Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        truth_values: List[float] = []
        pred_values: List[float] = []
        for truth, prediction in zip(truths, predictions):
            truth_number = try_float(truth)
            pred_number = try_float(prediction)
            if truth_number is None or pred_number is None:
                continue
            truth_values.append(truth_number)
            pred_values.append(pred_number)

        if len(truth_values) == 0:
            output = {name: float("nan") for name in self.metric_columns}
            return output, {}

        mse = mean([(truth - pred) ** 2 for truth, pred in zip(truth_values, pred_values)])
        rmse = math.sqrt(mse)
        pearson_value = pearson_correlation(truth_values, pred_values)
        spearman_value = spearman_correlation(truth_values, pred_values)
        return {
            "mse": mse,
            "rmse": rmse,
            "pearson_coeff": pearson_value,
            "spearman_coeff": spearman_value,
        }, {}

    def format_demonstration(
        self,
        train_text: str,
        train_label: Any,
        dataset_cfg: Dict[str, Any],
        label_schema: Optional[LabelSchema],
    ) -> str:
        return f"text pair: {train_text}\nscore: {train_label}"

    def aggregate_sc_predictions(
        self,
        parsed_predictions: Sequence[Any],
        raw_predictions: Sequence[str],
        dataset_cfg: Dict[str, Any],
        label_schema: Optional[LabelSchema],
    ) -> Tuple[Any, str]:
        numbers: List[float] = []
        for item in parsed_predictions:
            value = try_float(item)
            if value is not None:
                numbers.append(value)
        if len(numbers) == 0:
            raw_value = raw_predictions[0] if len(raw_predictions) > 0 else ""
            parsed = self.parse_prediction(raw_value, dataset_cfg, label_schema)
            return parsed, raw_value

        avg_value = self._clamp(float(mean(numbers)), dataset_cfg)
        closest_idx = 0
        closest_dist = float("inf")
        for idx, item in enumerate(parsed_predictions):
            value = try_float(item)
            if value is None:
                continue
            dist = abs(value - avg_value)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = idx
        return avg_value, raw_predictions[closest_idx]


TASK_PROCESSOR = RegressionTaskProcessor()


TASK_DATASETS: Dict[str, Dict[str, Any]] = {
    "sts12": {
        "task": "regression",
        "task_group": "similarity",
        "train_file": "sts12_train.csv",
        "test_file": "sts12_test.csv",
        "score_min": 0.0,
        "score_max": 5.0,
        "score_fallback": 2.5,
    },
    "stsb": {
        "task": "regression",
        "task_group": "similarity",
        "train_file": "stsb_train.csv",
        "test_file": "stsb_test.csv",
        "score_min": 0.0,
        "score_max": 5.0,
        "score_fallback": 2.5,
    },
}
