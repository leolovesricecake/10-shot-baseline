"""Pluggable baseline strategy implementations."""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from baseline_retrieval import BaseRetriever, build_retriever
from baseline_task_base import BaseTaskProcessor, LabelSchema


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


@dataclass
class StrategySampleOutput:
    raw_prediction: str
    parsed_prediction: Any
    input_tokens: int
    output_tokens: int
    runtime: float
    extra_columns: Dict[str, Any] = field(default_factory=dict)


class InferenceEngineProtocol(ABC):
    @abstractmethod
    def generate_prompts(
        self,
        prompts: Sequence[str],
        *,
        do_sample: bool,
        temperature: float,
        top_p: float,
        fallback_texts: Optional[Sequence[str]] = None,
    ) -> Tuple[List[str], List[int], List[int], List[float]]:
        raise NotImplementedError


@dataclass
class StrategyRuntimeContext:
    processor: BaseTaskProcessor
    dataset_cfg: Dict[str, Any]
    label_schema: Optional[LabelSchema]
    train_texts: Sequence[str]
    train_labels_for_context: Sequence[Any]
    shot_k: int
    random_seed: int
    sc_num_samples: int
    sc_temperature: float
    sc_top_p: float
    retrieval_backend: str
    retrieval_model_name: str
    default_do_sample: bool
    default_temperature: float
    default_top_p: float
    state: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    name: str
    extra_columns: Tuple[str, ...] = ()

    def file_suffix(self, context: StrategyRuntimeContext) -> str:
        return ""

    def prepare_dataset(self, context: StrategyRuntimeContext) -> None:
        return None

    def footer_items(self, context: StrategyRuntimeContext) -> List[Tuple[str, Any]]:
        return []

    @abstractmethod
    def run_batch(
        self,
        *,
        batch_texts: Sequence[str],
        batch_truths: Sequence[Any],
        batch_indices: Sequence[int],
        context: StrategyRuntimeContext,
        engine: InferenceEngineProtocol,
    ) -> List[StrategySampleOutput]:
        raise NotImplementedError


class CotStrategy(BaseStrategy):
    name = "cot"

    def run_batch(
        self,
        *,
        batch_texts: Sequence[str],
        batch_truths: Sequence[Any],
        batch_indices: Sequence[int],
        context: StrategyRuntimeContext,
        engine: InferenceEngineProtocol,
    ) -> List[StrategySampleOutput]:
        prompts = [
            context.processor.build_zero_shot_prompt(
                context.processor.prepare_input_text(text, context.dataset_cfg),
                context.dataset_cfg,
                context.label_schema,
            )
            for text in batch_texts
        ]
        fallback_texts = [str(item) for item in batch_truths]
        raws, in_tokens, out_tokens, runtimes = engine.generate_prompts(
            prompts,
            do_sample=context.default_do_sample,
            temperature=context.default_temperature,
            top_p=context.default_top_p,
            fallback_texts=fallback_texts,
        )

        outputs: List[StrategySampleOutput] = []
        for raw, in_tok, out_tok, runtime in zip(raws, in_tokens, out_tokens, runtimes):
            parsed = context.processor.parse_prediction(raw, context.dataset_cfg, context.label_schema)
            outputs.append(
                StrategySampleOutput(
                    raw_prediction=raw,
                    parsed_prediction=parsed,
                    input_tokens=int(in_tok),
                    output_tokens=int(out_tok),
                    runtime=float(runtime),
                )
            )
        return outputs


class _ContextStrategyBase(BaseStrategy):
    extra_columns = ("in_context", "context_indices", "retrieval_scores")

    def file_suffix(self, context: StrategyRuntimeContext) -> str:
        return f"_k{context.shot_k}"

    def _build_in_context(self, context: StrategyRuntimeContext, indices: Sequence[int]) -> str:
        if len(indices) == 0:
            return ""
        chunks = ["For example,"]
        for index in indices:
            chunks.append(
                context.processor.format_demonstration(
                    str(context.train_texts[index]),
                    context.train_labels_for_context[index],
                    context.dataset_cfg,
                    context.label_schema,
                )
            )
        return "\n".join(chunks)

    def _finalize_outputs(
        self,
        *,
        prompts: Sequence[str],
        batch_truths: Sequence[Any],
        in_contexts: Sequence[str],
        context_indices: Sequence[List[int]],
        retrieval_scores: Sequence[List[float]],
        context: StrategyRuntimeContext,
        engine: InferenceEngineProtocol,
    ) -> List[StrategySampleOutput]:
        fallback_texts = [str(item) for item in batch_truths]
        raws, in_tokens, out_tokens, runtimes = engine.generate_prompts(
            prompts,
            do_sample=context.default_do_sample,
            temperature=context.default_temperature,
            top_p=context.default_top_p,
            fallback_texts=fallback_texts,
        )
        outputs: List[StrategySampleOutput] = []
        for raw, in_tok, out_tok, runtime, in_ctx, idxs, scores in zip(
            raws, in_tokens, out_tokens, runtimes, in_contexts, context_indices, retrieval_scores
        ):
            parsed = context.processor.parse_prediction(raw, context.dataset_cfg, context.label_schema)
            outputs.append(
                StrategySampleOutput(
                    raw_prediction=raw,
                    parsed_prediction=parsed,
                    input_tokens=int(in_tok),
                    output_tokens=int(out_tok),
                    runtime=float(runtime),
                    extra_columns={
                        "in_context": in_ctx,
                        "context_indices": _json_dump(idxs),
                        "retrieval_scores": _json_dump(scores),
                    },
                )
            )
        return outputs


class FixedStrategy(_ContextStrategyBase):
    name = "fixed"

    def prepare_dataset(self, context: StrategyRuntimeContext) -> None:
        shot_k = min(context.shot_k, len(context.train_texts))
        context.state["fixed_indices"] = list(range(shot_k))

    def footer_items(self, context: StrategyRuntimeContext) -> List[Tuple[str, Any]]:
        return [("shot_k", context.shot_k)]

    def run_batch(
        self,
        *,
        batch_texts: Sequence[str],
        batch_truths: Sequence[Any],
        batch_indices: Sequence[int],
        context: StrategyRuntimeContext,
        engine: InferenceEngineProtocol,
    ) -> List[StrategySampleOutput]:
        fixed_indices = list(context.state.get("fixed_indices", []))
        in_context = self._build_in_context(context, fixed_indices)
        prompts = [
            context.processor.build_few_shot_prompt(
                context.processor.prepare_input_text(text, context.dataset_cfg),
                in_context,
                context.dataset_cfg,
                context.label_schema,
            )
            for text in batch_texts
        ]
        return self._finalize_outputs(
            prompts=prompts,
            batch_truths=batch_truths,
            in_contexts=[in_context for _ in batch_texts],
            context_indices=[fixed_indices for _ in batch_texts],
            retrieval_scores=[[] for _ in batch_texts],
            context=context,
            engine=engine,
        )


class RandomStrategy(_ContextStrategyBase):
    name = "random"

    def prepare_dataset(self, context: StrategyRuntimeContext) -> None:
        shot_k = min(context.shot_k, len(context.train_texts))
        rng = random.Random(context.random_seed)
        candidates = list(range(len(context.train_texts)))
        if shot_k == len(candidates):
            sampled = candidates
        else:
            sampled = rng.sample(candidates, shot_k)
        context.state["random_indices"] = sampled

    def footer_items(self, context: StrategyRuntimeContext) -> List[Tuple[str, Any]]:
        return [("shot_k", context.shot_k), ("random_seed", context.random_seed)]

    def run_batch(
        self,
        *,
        batch_texts: Sequence[str],
        batch_truths: Sequence[Any],
        batch_indices: Sequence[int],
        context: StrategyRuntimeContext,
        engine: InferenceEngineProtocol,
    ) -> List[StrategySampleOutput]:
        random_indices = list(context.state.get("random_indices", []))
        in_context = self._build_in_context(context, random_indices)
        prompts = [
            context.processor.build_few_shot_prompt(
                context.processor.prepare_input_text(text, context.dataset_cfg),
                in_context,
                context.dataset_cfg,
                context.label_schema,
            )
            for text in batch_texts
        ]
        return self._finalize_outputs(
            prompts=prompts,
            batch_truths=batch_truths,
            in_contexts=[in_context for _ in batch_texts],
            context_indices=[random_indices for _ in batch_texts],
            retrieval_scores=[[] for _ in batch_texts],
            context=context,
            engine=engine,
        )


class FewShotCotStrategy(_ContextStrategyBase):
    name = "few-shot-cot"

    def prepare_dataset(self, context: StrategyRuntimeContext) -> None:
        retriever, backend_used = build_retriever(
            context.train_texts,
            backend=context.retrieval_backend,
            semantic_model_name=context.retrieval_model_name,
        )
        context.state["retriever"] = retriever
        context.state["retrieval_backend_used"] = backend_used

    def footer_items(self, context: StrategyRuntimeContext) -> List[Tuple[str, Any]]:
        return [
            ("shot_k", context.shot_k),
            ("retrieval_backend", context.state.get("retrieval_backend_used", context.retrieval_backend)),
        ]

    def run_batch(
        self,
        *,
        batch_texts: Sequence[str],
        batch_truths: Sequence[Any],
        batch_indices: Sequence[int],
        context: StrategyRuntimeContext,
        engine: InferenceEngineProtocol,
    ) -> List[StrategySampleOutput]:
        retriever: BaseRetriever = context.state["retriever"]
        shot_k = min(context.shot_k, len(context.train_texts))
        prompts: List[str] = []
        in_contexts: List[str] = []
        context_indices: List[List[int]] = []
        retrieval_scores: List[List[float]] = []

        for text in batch_texts:
            idxs, scores = retriever.query(text, shot_k)
            if len(idxs) == 0:
                idxs = list(range(shot_k))
                scores = [0.0 for _ in idxs]
            in_context = self._build_in_context(context, idxs)
            prompts.append(
                context.processor.build_few_shot_prompt(
                    context.processor.prepare_input_text(text, context.dataset_cfg),
                    in_context,
                    context.dataset_cfg,
                    context.label_schema,
                )
            )
            in_contexts.append(in_context)
            context_indices.append(idxs)
            retrieval_scores.append(scores)

        return self._finalize_outputs(
            prompts=prompts,
            batch_truths=batch_truths,
            in_contexts=in_contexts,
            context_indices=context_indices,
            retrieval_scores=retrieval_scores,
            context=context,
            engine=engine,
        )


class ScCotStrategy(BaseStrategy):
    name = "sc-cot"
    extra_columns = ("sc_raw_samples", "sc_parsed_samples")

    def footer_items(self, context: StrategyRuntimeContext) -> List[Tuple[str, Any]]:
        return [
            ("sc_num_samples", context.sc_num_samples),
            ("sc_temperature", context.sc_temperature),
            ("sc_top_p", context.sc_top_p),
        ]

    def run_batch(
        self,
        *,
        batch_texts: Sequence[str],
        batch_truths: Sequence[Any],
        batch_indices: Sequence[int],
        context: StrategyRuntimeContext,
        engine: InferenceEngineProtocol,
    ) -> List[StrategySampleOutput]:
        sample_count = max(1, int(context.sc_num_samples))
        prompts: List[str] = []
        prompt_owner_indices: List[int] = []
        fallback_texts: List[str] = []
        for owner_index, (text, truth) in enumerate(zip(batch_texts, batch_truths)):
            prepared_text = context.processor.prepare_input_text(text, context.dataset_cfg)
            for _ in range(sample_count):
                prompts.append(
                    context.processor.build_zero_shot_prompt(
                        prepared_text,
                        context.dataset_cfg,
                        context.label_schema,
                    )
                )
                prompt_owner_indices.append(owner_index)
                fallback_texts.append(str(truth))

        raws, in_tokens, out_tokens, runtimes = engine.generate_prompts(
            prompts,
            do_sample=True,
            temperature=context.sc_temperature,
            top_p=context.sc_top_p,
            fallback_texts=fallback_texts,
        )

        grouped_raws: List[List[str]] = [[] for _ in batch_texts]
        grouped_in_tokens: List[int] = [0 for _ in batch_texts]
        grouped_out_tokens: List[int] = [0 for _ in batch_texts]
        grouped_runtime: List[float] = [0.0 for _ in batch_texts]
        for owner_index, raw, in_tok, out_tok, runtime in zip(
            prompt_owner_indices, raws, in_tokens, out_tokens, runtimes
        ):
            grouped_raws[owner_index].append(raw)
            grouped_in_tokens[owner_index] += int(in_tok)
            grouped_out_tokens[owner_index] += int(out_tok)
            grouped_runtime[owner_index] += float(runtime)

        outputs: List[StrategySampleOutput] = []
        for owner_index in range(len(batch_texts)):
            raw_samples = grouped_raws[owner_index]
            parsed_samples = [
                context.processor.parse_prediction(item, context.dataset_cfg, context.label_schema)
                for item in raw_samples
            ]
            final_prediction, final_raw = context.processor.aggregate_sc_predictions(
                parsed_samples,
                raw_samples,
                context.dataset_cfg,
                context.label_schema,
            )
            outputs.append(
                StrategySampleOutput(
                    raw_prediction=str(final_raw),
                    parsed_prediction=final_prediction,
                    input_tokens=grouped_in_tokens[owner_index],
                    output_tokens=grouped_out_tokens[owner_index],
                    runtime=grouped_runtime[owner_index],
                    extra_columns={
                        "sc_raw_samples": _json_dump(raw_samples),
                        "sc_parsed_samples": _json_dump(parsed_samples),
                    },
                )
            )
        return outputs


class SvCotStrategy(BaseStrategy):
    name = "sv-cot"
    extra_columns = ("sv_initial_raw", "sv_verified_raw")

    def run_batch(
        self,
        *,
        batch_texts: Sequence[str],
        batch_truths: Sequence[Any],
        batch_indices: Sequence[int],
        context: StrategyRuntimeContext,
        engine: InferenceEngineProtocol,
    ) -> List[StrategySampleOutput]:
        initial_prompts = [
            context.processor.build_zero_shot_prompt(
                context.processor.prepare_input_text(text, context.dataset_cfg),
                context.dataset_cfg,
                context.label_schema,
            )
            for text in batch_texts
        ]
        fallback_texts = [str(item) for item in batch_truths]
        initial_raws, in_1, out_1, rt_1 = engine.generate_prompts(
            initial_prompts,
            do_sample=context.default_do_sample,
            temperature=context.default_temperature,
            top_p=context.default_top_p,
            fallback_texts=fallback_texts,
        )

        verify_prompts = [
            context.processor.build_verification_prompt(
                input_text=context.processor.prepare_input_text(text, context.dataset_cfg),
                initial_raw=initial_raw,
                dataset_cfg=context.dataset_cfg,
                label_schema=context.label_schema,
            )
            for text, initial_raw in zip(batch_texts, initial_raws)
        ]
        verified_raws, in_2, out_2, rt_2 = engine.generate_prompts(
            verify_prompts,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            fallback_texts=fallback_texts,
        )

        outputs: List[StrategySampleOutput] = []
        for initial_raw, verified_raw, in_first, out_first, rt_first, in_second, out_second, rt_second in zip(
            initial_raws, verified_raws, in_1, out_1, rt_1, in_2, out_2, rt_2
        ):
            parsed = context.processor.parse_prediction(
                verified_raw, context.dataset_cfg, context.label_schema
            )
            outputs.append(
                StrategySampleOutput(
                    raw_prediction=str(verified_raw),
                    parsed_prediction=parsed,
                    input_tokens=int(in_first) + int(in_second),
                    output_tokens=int(out_first) + int(out_second),
                    runtime=float(rt_first) + float(rt_second),
                    extra_columns={
                        "sv_initial_raw": str(initial_raw),
                        "sv_verified_raw": str(verified_raw),
                    },
                )
            )
        return outputs


STRATEGY_REGISTRY: Dict[str, BaseStrategy] = {
    "cot": CotStrategy(),
    "few-shot-cot": FewShotCotStrategy(),
    "sc-cot": ScCotStrategy(),
    "sv-cot": SvCotStrategy(),
    "random": RandomStrategy(),
    "fixed": FixedStrategy(),
}
