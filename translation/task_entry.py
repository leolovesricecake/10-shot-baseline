"""Translation task registry and processor."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from baseline_task_base import (
    BaseTaskProcessor,
    LabelSchema,
    compute_bleu_fallback,
    compute_chrf_fallback,
    mean,
)


class TranslationTaskProcessor(BaseTaskProcessor):
    task_name = "translation"
    metric_columns = ("acc", "bleu", "chrf")

    def _source_lang(self, dataset_cfg: Dict[str, Any]) -> str:
        return str(dataset_cfg.get("source_language", "source language"))

    def _target_lang(self, dataset_cfg: Dict[str, Any]) -> str:
        return str(dataset_cfg.get("target_language", "target language"))

    def build_zero_shot_prompt(
        self, input_text: str, dataset_cfg: Dict[str, Any], label_schema: Optional[LabelSchema]
    ) -> str:
        source_language = self._source_lang(dataset_cfg)
        target_language = self._target_lang(dataset_cfg)
        return (
            "You are a translation assistant.\n"
            f"Translate the input text from {source_language} to {target_language}.\n"
            "Think step by step internally.\n"
            "Output only the final translation.\n\n"
            f"Input ({source_language}):\n{input_text}\n\n"
            f"Translation ({target_language}):"
        )

    def build_few_shot_prompt(
        self,
        input_text: str,
        in_context: str,
        dataset_cfg: Dict[str, Any],
        label_schema: Optional[LabelSchema],
    ) -> str:
        source_language = self._source_lang(dataset_cfg)
        target_language = self._target_lang(dataset_cfg)
        return (
            "You are a translation assistant.\n"
            f"Translate text from {source_language} to {target_language}.\n"
            "Learn from the examples and output only the final translation.\n\n"
            f"{in_context}\n\n"
            f"Input ({source_language}):\n{input_text}\n\n"
            f"Translation ({target_language}):"
        )

    def build_verification_prompt(
        self,
        input_text: str,
        initial_raw: str,
        dataset_cfg: Dict[str, Any],
        label_schema: Optional[LabelSchema],
    ) -> str:
        source_language = self._source_lang(dataset_cfg)
        target_language = self._target_lang(dataset_cfg)
        return (
            "You are verifying a translation result.\n"
            "Check and refine the previous translation if needed.\n"
            "Output only the final translation.\n\n"
            f"Input ({source_language}):\n{input_text}\n\n"
            f"Previous translation ({target_language}):\n{initial_raw}\n\n"
            f"Final translation ({target_language}):"
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
            from sacrebleu.metrics import BLEU, CHRF  # type: ignore

            bleu_metric = BLEU(effective_order=True)
            chrf_metric = CHRF()
            bleu = float(bleu_metric.corpus_score(hypotheses, [references]).score) / 100.0
            chrf = float(chrf_metric.corpus_score(hypotheses, [references]).score) / 100.0
            backend = "sacrebleu"
        except Exception:
            bleu = compute_bleu_fallback(hypotheses, references)
            chrf = compute_chrf_fallback(hypotheses, references)
        return {"acc": bleu, "bleu": bleu, "chrf": chrf}, {"metric_backend": backend}

    def format_demonstration(
        self,
        train_text: str,
        train_label: Any,
        dataset_cfg: Dict[str, Any],
        label_schema: Optional[LabelSchema],
    ) -> str:
        source_language = self._source_lang(dataset_cfg)
        target_language = self._target_lang(dataset_cfg)
        return (
            f"{source_language}: {train_text}\n"
            f"{target_language}: {train_label}"
        )

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

        # Choose the candidate closest to other candidates in edit similarity.
        texts = [item[0] for item in valid_pairs]
        best_index = 0
        best_score = -1.0
        import difflib

        for idx, item in enumerate(texts):
            scores: List[float] = []
            for jdx, other in enumerate(texts):
                if idx == jdx:
                    continue
                scores.append(difflib.SequenceMatcher(None, item, other).ratio())
            score = mean(scores) if len(scores) > 0 else 0.0
            if score > best_score:
                best_score = score
                best_index = idx
        return valid_pairs[best_index][0], valid_pairs[best_index][1]


TASK_PROCESSOR = TranslationTaskProcessor()


TASK_DATASETS: Dict[str, Dict[str, Any]] = {
    "wmt19_En-Zh": {
        "task": "translation",
        "task_group": "translation",
        "train_file": "wmt19_En-Zh_train.csv",
        "test_file": "wmt19_En-Zh_test.csv",
        "source_language": "English",
        "target_language": "Chinese",
    },
    "wmt19_Zh-En": {
        "task": "translation",
        "task_group": "translation",
        "train_file": "wmt19_Zh-En_train.csv",
        "test_file": "wmt19_Zh-En_test.csv",
        "source_language": "Chinese",
        "target_language": "English",
    },
}
