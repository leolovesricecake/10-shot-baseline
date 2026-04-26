"""Task processor base classes and shared metric/parsing utilities."""

from __future__ import annotations

import difflib
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


def normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def try_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_first_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    matches = re.findall(r"[-+]?\d*\.?\d+", str(value))
    for item in matches:
        number = try_float(item)
        if number is not None:
            return number
    return None


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def mean(values: Sequence[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return sum(values) / len(values)


def build_ngrams(tokens: Sequence[str], n: int) -> Counter:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def tokenize_text(text: Any) -> List[str]:
    raw = str(text).strip()
    if not raw:
        return []
    if any(char.isspace() for char in raw):
        return [token for token in raw.split() if token]
    return [char for char in raw if not char.isspace()]


def tokenize_chars(text: Any) -> List[str]:
    return [char for char in str(text) if not char.isspace()]


def rank_with_ties(values: Sequence[float]) -> List[float]:
    if len(values) == 0:
        return []
    indexed = sorted((value, index) for index, value in enumerate(values))
    output = [0.0 for _ in values]
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][0] == indexed[start][0]:
            end += 1
        rank_value = (start + 1 + end) / 2.0
        for cursor in range(start, end):
            output[indexed[cursor][1]] = rank_value
        start = end
    return output


def pearson_correlation(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return float("nan")
    x_mean = mean(x_values)
    y_mean = mean(y_values)
    x_centered = [x - x_mean for x in x_values]
    y_centered = [y - y_mean for y in y_values]
    numerator = sum(x * y for x, y in zip(x_centered, y_centered))
    denominator = math.sqrt(sum(x * x for x in x_centered)) * math.sqrt(sum(y * y for y in y_centered))
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def spearman_correlation(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return float("nan")
    return pearson_correlation(rank_with_ties(x_values), rank_with_ties(y_values))


def compute_bleu_fallback(predictions: Sequence[str], references: Sequence[str]) -> float:
    max_order = 4
    match_counts = [0 for _ in range(max_order)]
    total_counts = [0 for _ in range(max_order)]
    pred_length = 0
    ref_length = 0

    for prediction, reference in zip(predictions, references):
        pred_tokens = tokenize_text(prediction)
        ref_tokens = tokenize_text(reference)
        pred_length += len(pred_tokens)
        ref_length += len(ref_tokens)
        for order in range(1, max_order + 1):
            pred_ngrams = build_ngrams(pred_tokens, order)
            ref_ngrams = build_ngrams(ref_tokens, order)
            total_counts[order - 1] += sum(pred_ngrams.values())
            match_counts[order - 1] += sum(
                min(count, ref_ngrams.get(ngram, 0)) for ngram, count in pred_ngrams.items()
            )

    if pred_length == 0:
        return 0.0

    precisions: List[float] = []
    for matches, totals in zip(match_counts, total_counts):
        precisions.append((matches + 1.0) / (totals + 1.0))

    brevity_penalty = 1.0
    if pred_length < ref_length:
        brevity_penalty = math.exp(1.0 - safe_divide(ref_length, pred_length))
    return brevity_penalty * math.exp(sum(math.log(item) for item in precisions) / max_order)


def compute_chrf_fallback(
    predictions: Sequence[str], references: Sequence[str], max_order: int = 6, beta: float = 2.0
) -> float:
    beta_sq = beta * beta
    scores: List[float] = []
    for order in range(1, max_order + 1):
        matches = 0
        pred_total = 0
        ref_total = 0
        for prediction, reference in zip(predictions, references):
            pred_chars = tokenize_chars(prediction)
            ref_chars = tokenize_chars(reference)
            pred_ngrams = build_ngrams(pred_chars, order)
            ref_ngrams = build_ngrams(ref_chars, order)
            pred_total += sum(pred_ngrams.values())
            ref_total += sum(ref_ngrams.values())
            matches += sum(min(count, ref_ngrams.get(ngram, 0)) for ngram, count in pred_ngrams.items())

        precision = safe_divide(matches, pred_total)
        recall = safe_divide(matches, ref_total)
        denominator = recall + beta_sq * precision
        if denominator == 0:
            scores.append(0.0)
        else:
            scores.append((1.0 + beta_sq) * precision * recall / denominator)
    return mean(scores)


def compute_ngram_f1(pred_tokens: Sequence[str], ref_tokens: Sequence[str], order: int) -> float:
    pred_ngrams = build_ngrams(pred_tokens, order)
    ref_ngrams = build_ngrams(ref_tokens, order)
    overlap = sum(min(count, ref_ngrams.get(ngram, 0)) for ngram, count in pred_ngrams.items())
    precision = safe_divide(overlap, sum(pred_ngrams.values()))
    recall = safe_divide(overlap, sum(ref_ngrams.values()))
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0
    if len(right) > len(left):
        left, right = right, left
    row = [0 for _ in range(len(right) + 1)]
    for token_left in left:
        previous = 0
        for index in range(1, len(right) + 1):
            current = row[index]
            if token_left == right[index - 1]:
                row[index] = previous + 1
            else:
                row[index] = max(row[index], row[index - 1])
            previous = current
    return row[-1]


def compute_rouge_fallback(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    rouge1_values: List[float] = []
    rouge2_values: List[float] = []
    rouge_l_values: List[float] = []
    for prediction, reference in zip(predictions, references):
        pred_tokens = tokenize_text(prediction)
        ref_tokens = tokenize_text(reference)
        rouge1_values.append(compute_ngram_f1(pred_tokens, ref_tokens, order=1))
        rouge2_values.append(compute_ngram_f1(pred_tokens, ref_tokens, order=2))
        lcs = lcs_length(pred_tokens, ref_tokens)
        precision = safe_divide(lcs, len(pred_tokens))
        recall = safe_divide(lcs, len(ref_tokens))
        if precision + recall == 0:
            rouge_l_values.append(0.0)
        else:
            rouge_l_values.append(2.0 * precision * recall / (precision + recall))
    return {
        "rouge1": mean(rouge1_values),
        "rouge2": mean(rouge2_values),
        "rougeL": mean(rouge_l_values),
    }


def centroid_text_index(candidates: Sequence[str]) -> int:
    if len(candidates) == 0:
        return 0
    if len(candidates) == 1:
        return 0
    best_index = 0
    best_score = -1.0
    for index, item in enumerate(candidates):
        scores = []
        for other_index, other_item in enumerate(candidates):
            if index == other_index:
                continue
            scores.append(difflib.SequenceMatcher(None, item, other_item).ratio())
        avg_score = mean(scores) if len(scores) > 0 else 0.0
        if avg_score > best_score:
            best_score = avg_score
            best_index = index
    return best_index


@dataclass
class LabelSchema:
    prompt_labels: List[str]
    alias_to_canonical: Dict[str, str]
    canonical_to_display: Dict[str, str]

    def canonicalize_truth(self, value: Any) -> str:
        normalized = normalize_text(value)
        return self.alias_to_canonical.get(normalized, normalized)

    def parse_prediction(self, raw_text: Any) -> Optional[str]:
        if raw_text is None:
            return None
        text = str(raw_text).strip()
        if not text:
            return None

        candidates: List[str] = [text]
        candidates.extend([line for line in text.splitlines() if line.strip()])
        normalized_colon_text = text.replace("\uFF1A", ":")
        candidates.extend([segment for segment in normalized_colon_text.split(":") if segment.strip()])
        for candidate in reversed(candidates):
            normalized_candidate = normalize_text(candidate)
            if normalized_candidate in self.alias_to_canonical:
                return self.alias_to_canonical[normalized_candidate]

        normalized_text = normalize_text(text)
        ordered_aliases = sorted(self.alias_to_canonical.items(), key=lambda item: len(item[0]), reverse=True)
        for alias, canonical in ordered_aliases:
            if alias and re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized_text):
                return canonical

        number = extract_first_float(text)
        if number is not None:
            candidate_number = normalize_text(str(int(round(number))))
            if candidate_number in self.alias_to_canonical:
                return self.alias_to_canonical[candidate_number]
        return None

    def display(self, canonical_value: Any) -> str:
        if canonical_value is None:
            return ""
        canonical = normalize_text(canonical_value)
        return self.canonical_to_display.get(canonical, str(canonical_value))


def build_label_schema(train_labels: Sequence[Any], dataset_cfg: Dict[str, Any]) -> LabelSchema:
    alias_to_canonical: Dict[str, str] = {}
    canonical_to_display: Dict[str, str] = {}
    prompt_labels: List[str] = []

    label_names_cfg = dataset_cfg.get("label_names")
    if isinstance(label_names_cfg, dict) and len(label_names_cfg) > 0:
        def _label_key(item: Tuple[Any, Any]) -> Tuple[int, Any]:
            number = try_float(item[0])
            if number is None:
                return (1, str(item[0]))
            return (0, number)

        for raw_key, display_name in sorted(label_names_cfg.items(), key=_label_key):
            canonical = normalize_text(raw_key)
            display = str(display_name).strip()
            canonical_to_display[canonical] = display
            alias_to_canonical[canonical] = canonical
            alias_to_canonical[normalize_text(display)] = canonical
            prompt_labels.append(display)
        return LabelSchema(
            prompt_labels=prompt_labels,
            alias_to_canonical=alias_to_canonical,
            canonical_to_display=canonical_to_display,
        )

    unique_labels = []
    seen = set()
    for label in train_labels:
        canonical = normalize_text(label)
        if canonical in seen:
            continue
        seen.add(canonical)
        unique_labels.append(label)

    for label in unique_labels:
        canonical = normalize_text(label)
        display = str(label).strip()
        canonical_to_display[canonical] = display
        alias_to_canonical[canonical] = canonical
        prompt_labels.append(display)

    prompt_labels = sorted(prompt_labels, key=normalize_text)
    return LabelSchema(
        prompt_labels=prompt_labels,
        alias_to_canonical=alias_to_canonical,
        canonical_to_display=canonical_to_display,
    )


class BaseTaskProcessor(ABC):
    task_name: str
    metric_columns: Tuple[str, ...]

    def needs_label_schema(self) -> bool:
        return False

    def create_label_schema(self, train_labels: Sequence[Any], dataset_cfg: Dict[str, Any]) -> Optional[LabelSchema]:
        if not self.needs_label_schema():
            return None
        return build_label_schema(train_labels, dataset_cfg)

    def canonicalize_truth(
        self, truth: Any, dataset_cfg: Dict[str, Any], label_schema: Optional[LabelSchema]
    ) -> Any:
        return truth

    def canonicalize_prediction(
        self, prediction: Any, dataset_cfg: Dict[str, Any], label_schema: Optional[LabelSchema]
    ) -> Any:
        return prediction

    def display_truth(self, truth: Any, dataset_cfg: Dict[str, Any], label_schema: Optional[LabelSchema]) -> Any:
        return truth

    def display_prediction(
        self, prediction: Any, dataset_cfg: Dict[str, Any], label_schema: Optional[LabelSchema]
    ) -> Any:
        return prediction

    def prepare_input_text(self, input_text: str, dataset_cfg: Dict[str, Any]) -> str:
        text = str(input_text)
        limit = dataset_cfg.get("prompt_truncation")
        if isinstance(limit, int) and limit > 0 and len(text) > limit:
            return text[:limit]
        return text

    @abstractmethod
    def build_zero_shot_prompt(
        self, input_text: str, dataset_cfg: Dict[str, Any], label_schema: Optional[LabelSchema]
    ) -> str:
        raise NotImplementedError

    def build_few_shot_prompt(
        self,
        input_text: str,
        in_context: str,
        dataset_cfg: Dict[str, Any],
        label_schema: Optional[LabelSchema],
    ) -> str:
        prompt = self.build_zero_shot_prompt(input_text, dataset_cfg, label_schema)
        if not in_context.strip():
            return prompt
        return f"{in_context}\n\nNow solve the target example:\n{prompt}"

    def build_verification_prompt(
        self,
        input_text: str,
        initial_raw: str,
        dataset_cfg: Dict[str, Any],
        label_schema: Optional[LabelSchema],
    ) -> str:
        return (
            "Review your previous answer and provide a corrected final answer if needed.\n"
            "Only output the final answer.\n\n"
            f"Input:\n{input_text}\n\n"
            f"Previous answer:\n{initial_raw}\n\n"
            "Final answer:"
        )

    @abstractmethod
    def parse_prediction(
        self, raw_text: Any, dataset_cfg: Dict[str, Any], label_schema: Optional[LabelSchema]
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, truths: Sequence[Any], predictions: Sequence[Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        raise NotImplementedError

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

        parsed_texts = [str(item[0]) for item in valid_pairs]
        center_idx = centroid_text_index(parsed_texts)
        return valid_pairs[center_idx][0], valid_pairs[center_idx][1]
