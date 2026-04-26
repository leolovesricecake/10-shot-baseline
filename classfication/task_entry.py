"""Classification task registry and processor."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from baseline_task_base import BaseTaskProcessor, LabelSchema, mean, normalize_text


class ClassificationTaskProcessor(BaseTaskProcessor):
    task_name = "classification"
    metric_columns = ("acc", "accuracy", "macro_f1")

    def needs_label_schema(self) -> bool:
        return True

    def canonicalize_truth(
        self, truth: Any, dataset_cfg: Dict[str, Any], label_schema: Optional[LabelSchema]
    ) -> Any:
        if label_schema is None:
            return normalize_text(truth)
        return label_schema.canonicalize_truth(truth)

    def canonicalize_prediction(
        self, prediction: Any, dataset_cfg: Dict[str, Any], label_schema: Optional[LabelSchema]
    ) -> Any:
        if prediction is None:
            return None
        if label_schema is None:
            return normalize_text(prediction)
        return label_schema.canonicalize_truth(prediction)

    def display_truth(self, truth: Any, dataset_cfg: Dict[str, Any], label_schema: Optional[LabelSchema]) -> Any:
        if label_schema is None:
            return str(truth)
        return label_schema.display(truth)

    def display_prediction(
        self, prediction: Any, dataset_cfg: Dict[str, Any], label_schema: Optional[LabelSchema]
    ) -> Any:
        if label_schema is None:
            return "" if prediction is None else str(prediction)
        return label_schema.display(prediction)

    def build_zero_shot_prompt(
        self, input_text: str, dataset_cfg: Dict[str, Any], label_schema: Optional[LabelSchema]
    ) -> str:
        labels_text = ", ".join((label_schema.prompt_labels if label_schema else []))
        return (
            "You are a text classification assistant.\n"
            "Classify the input text into one label from the candidate labels.\n"
            f"Candidate labels: {labels_text}\n"
            "Think step by step internally.\n"
            "Output only one final label.\n\n"
            f"Input text:\n{input_text}\n\n"
            "Label:"
        )

    def build_few_shot_prompt(
        self,
        input_text: str,
        in_context: str,
        dataset_cfg: Dict[str, Any],
        label_schema: Optional[LabelSchema],
    ) -> str:
        labels_text = ", ".join((label_schema.prompt_labels if label_schema else []))
        return (
            "You are a text classification assistant.\n"
            "Learn from the examples and classify the target text.\n"
            f"Candidate labels: {labels_text}\n"
            "Think step by step internally.\n"
            "Output only one final label.\n\n"
            f"{in_context}\n\n"
            f"Target input text:\n{input_text}\n\n"
            "Label:"
        )

    def build_verification_prompt(
        self,
        input_text: str,
        initial_raw: str,
        dataset_cfg: Dict[str, Any],
        label_schema: Optional[LabelSchema],
    ) -> str:
        labels_text = ", ".join((label_schema.prompt_labels if label_schema else []))
        return (
            "You are verifying a text classification answer.\n"
            f"Candidate labels: {labels_text}\n"
            "Check the previous answer and output the corrected final label if needed.\n"
            "Output only one label.\n\n"
            f"Input text:\n{input_text}\n\n"
            f"Previous answer:\n{initial_raw}\n\n"
            "Final label:"
        )

    def parse_prediction(
        self, raw_text: Any, dataset_cfg: Dict[str, Any], label_schema: Optional[LabelSchema]
    ) -> Any:
        if label_schema is None:
            return normalize_text(raw_text)
        return label_schema.parse_prediction(raw_text)

    def evaluate(self, truths: Sequence[Any], predictions: Sequence[Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        if len(truths) == 0:
            output = {name: float("nan") for name in self.metric_columns}
            return output, {}

        normalized_truths = [normalize_text(value) for value in truths]
        normalized_predictions = [
            normalize_text(value) if value is not None and str(value).strip() else "__none__"
            for value in predictions
        ]
        matched = sum(
            1
            for truth, prediction in zip(normalized_truths, normalized_predictions)
            if truth == prediction
        )
        accuracy = matched / len(normalized_truths)

        labels = sorted(set(normalized_truths) | set(normalized_predictions))
        f1_values: List[float] = []
        for label in labels:
            tp = sum(
                1
                for truth, prediction in zip(normalized_truths, normalized_predictions)
                if truth == label and prediction == label
            )
            fp = sum(
                1
                for truth, prediction in zip(normalized_truths, normalized_predictions)
                if truth != label and prediction == label
            )
            fn = sum(
                1
                for truth, prediction in zip(normalized_truths, normalized_predictions)
                if truth == label and prediction != label
            )
            denominator = 2 * tp + fp + fn
            if denominator == 0:
                f1_values.append(0.0)
            else:
                f1_values.append((2 * tp) / denominator)

        macro_f1 = mean(f1_values)
        return {"acc": accuracy, "accuracy": accuracy, "macro_f1": macro_f1}, {}

    def format_demonstration(
        self,
        train_text: str,
        train_label: Any,
        dataset_cfg: Dict[str, Any],
        label_schema: Optional[LabelSchema],
    ) -> str:
        return f"input text: {train_text}\nlabel: {train_label}"

    def aggregate_sc_predictions(
        self,
        parsed_predictions: Sequence[Any],
        raw_predictions: Sequence[str],
        dataset_cfg: Dict[str, Any],
        label_schema: Optional[LabelSchema],
    ) -> Tuple[Any, str]:
        valid_pairs = [
            (parsed, raw)
            for parsed, raw in zip(parsed_predictions, raw_predictions)
            if parsed is not None and str(parsed).strip() != ""
        ]
        if len(valid_pairs) == 0:
            raw_value = raw_predictions[0] if len(raw_predictions) > 0 else ""
            return self.parse_prediction(raw_value, dataset_cfg, label_schema), raw_value

        counts: Dict[str, int] = {}
        for parsed, _ in valid_pairs:
            key = normalize_text(parsed)
            counts[key] = counts.get(key, 0) + 1
        ordered_keys = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        winner = ordered_keys[0][0]

        for parsed, raw in valid_pairs:
            if normalize_text(parsed) == winner:
                return parsed, raw
        return valid_pairs[0][0], valid_pairs[0][1]


TASK_PROCESSOR = ClassificationTaskProcessor()


TASK_DATASETS: Dict[str, Dict[str, Any]] = {
    "emotion": {
        "task": "classification",
        "task_group": "classification",
        "train_file": "emotion_train.csv",
        "test_file": "emotion_test.csv",
        "label_names": {
            0: "sadness",
            1: "joy",
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise",
        },
    },
    "sst5": {
        "task": "classification",
        "task_group": "classification",
        "train_file": "sst5_train.csv",
        "test_file": "sst5_test.csv",
        "label_names": {
            0: "very negative",
            1: "negative",
            2: "neutral",
            3: "positive",
            4: "very positive",
        },
    },
}
