"""Unified runner for multi-baseline text experiments."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import time
import traceback
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from baseline_runner_config import (
    DEFAULT_BASELINES,
    DEFAULT_MODELS,
    DEFAULT_RETRIEVAL_MODEL_NAME,
    MODEL_REGISTRY,
    PUBLIC_RESULT_COLUMNS,
    TASK_ENTRY_PATHS,
)
from baseline_logging import RunLogger, build_run_id
from baseline_retrieval import RetrievalInitializationError
from baseline_strategies import (
    InferenceEngineProtocol,
    STRATEGY_REGISTRY,
    BaseStrategy,
    StrategyRuntimeContext,
    StrategySampleOutput,
)
from baseline_task_base import BaseTaskProcessor, safe_divide


@dataclass
class DatasetSpec:
    name: str
    config: Dict[str, Any]
    processor: BaseTaskProcessor


@dataclass
class RunnerOptions:
    baselines: List[str]
    datasets_root: Path
    output_root: Path
    log_dir: Path
    models: List[str]
    datasets: List[str]
    task_groups: Optional[List[str]]
    skip_missing_datasets: bool
    batch_size: int
    max_new_tokens: int
    max_samples: Optional[int]
    device_map: str
    torch_dtype: Optional[str]
    do_sample: bool
    temperature: float
    top_p: float
    dry_run: bool
    overwrite: bool
    cuda_device: Optional[str]
    shot_k: int
    random_seed: int
    retrieval_backend: str
    retrieval_model_name: str
    sc_num_samples: int
    sc_temperature: float
    sc_top_p: float


class InferenceEngine(InferenceEngineProtocol):
    def __init__(self, tokenizer: Any, model: Any, options: RunnerOptions):
        self.tokenizer = tokenizer
        self.model = model
        self.options = options

    def generate_prompts(
        self,
        prompts: Sequence[str],
        *,
        do_sample: bool,
        temperature: float,
        top_p: float,
        fallback_texts: Optional[Sequence[str]] = None,
    ) -> Tuple[List[str], List[int], List[int], List[float]]:
        if len(prompts) == 0:
            return [], [], [], []

        if self.options.dry_run:
            if fallback_texts is None:
                outputs = ["" for _ in prompts]
            else:
                outputs = [str(item) for item in fallback_texts]
            zeros = [0 for _ in prompts]
            runtimes = [0.0 for _ in prompts]
            return outputs, zeros, zeros, runtimes

        messages_batch = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ]
        return generate_batch(
            tokenizer=self.tokenizer,
            model=self.model,
            messages_batch=messages_batch,
            max_new_tokens=self.options.max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )


def parse_args(default_baselines: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-baseline experiments on CSV datasets.")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=list(default_baselines or DEFAULT_BASELINES),
        help="Baseline names to run.",
    )
    parser.add_argument("--datasets-root", default="datasets", help="Root directory containing dataset CSVs.")
    parser.add_argument("--output-root", default="results", help="Output root directory.")
    parser.add_argument("--log-dir", default=None, help="Directory for run logs. Default: <output_root>/_logs")
    parser.add_argument("--models", nargs="+", default=None, help="Model aliases to run.")
    parser.add_argument("--datasets", nargs="+", default=None, help="Dataset names to run.")
    parser.add_argument("--task-groups", nargs="+", default=None, help="Task groups to run.")
    parser.add_argument(
        "--skip-missing-datasets",
        action="store_true",
        help="Skip missing dataset CSV files instead of raising errors.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max generation tokens.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit test samples.")
    parser.add_argument("--device-map", default="auto", help="transformers device_map.")
    parser.add_argument(
        "--torch-dtype",
        default=None,
        choices=["float16", "bfloat16", "float32"],
        help="Override torch dtype.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling for single-pass baselines (cot/few-shot/random/fixed).",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Sampling top-p.")
    parser.add_argument("--dry-run", action="store_true", help="Skip model inference.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing result files.")
    parser.add_argument(
        "--cuda-device",
        default=None,
        help="Single GPU index to use, e.g. 0. Multi-GPU values are not allowed.",
    )
    parser.add_argument("--shot-k", type=int, default=10, help="Number of in-context demonstrations.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for random baseline.")
    parser.add_argument(
        "--retrieval-backend",
        choices=["semantic", "bm25"],
        default="semantic",
        help="Retriever backend for few-shot-cot.",
    )
    parser.add_argument(
        "--retrieval-model-name",
        default=DEFAULT_RETRIEVAL_MODEL_NAME,
        help="Sentence embedding model for semantic retrieval.",
    )
    parser.add_argument("--sc-num-samples", type=int, default=5, help="Number of samples for sc-cot.")
    parser.add_argument("--sc-temperature", type=float, default=0.7, help="Temperature for sc-cot sampling.")
    parser.add_argument("--sc-top-p", type=float, default=0.9, help="Top-p for sc-cot sampling.")
    return parser.parse_args()


def build_runner_options(args: argparse.Namespace, registry: Dict[str, DatasetSpec]) -> RunnerOptions:
    output_root = Path(args.output_root).resolve()
    log_dir = Path(args.log_dir).resolve() if args.log_dir else (output_root / "_logs").resolve()
    selected_task_groups = list(args.task_groups) if args.task_groups else None
    selected_datasets = (
        list(args.datasets)
        if args.datasets
        else default_datasets_from_registry(
            registry=registry,
            task_groups=selected_task_groups,
        )
    )
    return RunnerOptions(
        baselines=[item.strip() for item in list(args.baselines)],
        datasets_root=Path(args.datasets_root).resolve(),
        output_root=output_root,
        log_dir=log_dir,
        models=list(args.models or DEFAULT_MODELS),
        datasets=selected_datasets,
        task_groups=selected_task_groups,
        skip_missing_datasets=bool(args.skip_missing_datasets),
        batch_size=max(1, int(args.batch_size)),
        max_new_tokens=max(1, int(args.max_new_tokens)),
        max_samples=args.max_samples,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        dry_run=bool(args.dry_run),
        overwrite=bool(args.overwrite),
        cuda_device=str(args.cuda_device) if args.cuda_device is not None else None,
        shot_k=max(1, int(args.shot_k)),
        random_seed=int(args.random_seed),
        retrieval_backend=str(args.retrieval_backend),
        retrieval_model_name=str(args.retrieval_model_name),
        sc_num_samples=max(1, int(args.sc_num_samples)),
        sc_temperature=float(args.sc_temperature),
        sc_top_p=float(args.sc_top_p),
    )


def _normalize_single_cuda_device(value: str) -> str:
    device = str(value).strip()
    if not device:
        raise ValueError("CUDA device is empty.")
    if "," in device:
        raise ValueError(f"Only single GPU is allowed, got CUDA device list: '{value}'.")
    if any(ch.isspace() for ch in device):
        raise ValueError(f"CUDA device must be a single token, got: '{value}'.")
    if not device.lstrip("-").isdigit():
        raise ValueError(f"CUDA device must be an integer index, got: '{value}'.")
    return device


def configure_single_gpu(options: RunnerOptions) -> str:
    if options.cuda_device is not None:
        device = _normalize_single_cuda_device(options.cuda_device)
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        return device

    env_value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env_value:
        device = _normalize_single_cuda_device(env_value)
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        return device

    # Default to single card 0 when not explicitly provided.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    return "0"


def options_to_dict(options: RunnerOptions) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in vars(options).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def load_task_registry() -> Dict[str, DatasetSpec]:
    merged: Dict[str, DatasetSpec] = {}
    for path in TASK_ENTRY_PATHS:
        if not path.exists():
            continue
        module_name = f"task_entry_{path.parent.name}"
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        task_datasets = getattr(module, "TASK_DATASETS", None)
        task_processor = getattr(module, "TASK_PROCESSOR", None)
        if not isinstance(task_datasets, dict):
            raise ValueError(f"{path} must define TASK_DATASETS as dict.")
        if not isinstance(task_processor, BaseTaskProcessor):
            raise ValueError(f"{path} must define TASK_PROCESSOR as BaseTaskProcessor instance.")
        for dataset_name, config in task_datasets.items():
            if dataset_name in merged:
                raise ValueError(f"Duplicated dataset config name: {dataset_name}")
            cfg = dict(config)
            cfg.setdefault("task", task_processor.task_name)
            cfg.setdefault("task_group", path.parent.name)
            merged[dataset_name] = DatasetSpec(name=dataset_name, config=cfg, processor=task_processor)
    return merged


def default_datasets_from_registry(
    registry: Dict[str, DatasetSpec],
    task_groups: Optional[Sequence[str]] = None,
) -> List[str]:
    output: List[str] = []
    for dataset_name, spec in registry.items():
        if task_groups is not None and len(task_groups) > 0:
            if spec.config.get("task_group") not in task_groups:
                continue
        output.append(dataset_name)
    return sorted(output)


def resolve_dataset_paths(
    datasets_root: Path,
    dataset_name: str,
    dataset_cfg: Dict[str, Any],
) -> Tuple[Path, Path]:
    train_file = dataset_cfg["train_file"]
    test_file = dataset_cfg["test_file"]
    candidates = [
        (datasets_root / dataset_name / train_file, datasets_root / dataset_name / test_file),
        (datasets_root / train_file, datasets_root / test_file),
    ]
    for train_path, test_path in candidates:
        if train_path.exists() and test_path.exists():
            return train_path, test_path
    attempted = [f"{train_path} | {test_path}" for train_path, test_path in candidates]
    raise FileNotFoundError(f"Cannot find train/test csv for dataset '{dataset_name}'. Tried: {attempted}")


def maybe_disable_thinking(tokenizer: Any, messages_batch: Sequence[List[Dict[str, str]]]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "return_tensors": "pt",
        "return_dict": True,
        "add_generation_prompt": True,
        "padding": True,
        "truncation": True,
    }
    try:
        return tokenizer.apply_chat_template(list(messages_batch), enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(list(messages_batch), **kwargs)


def get_model_device(model: Any) -> Any:
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except Exception:
        return "cpu"


def generate_batch(
    tokenizer: Any,
    model: Any,
    messages_batch: Sequence[List[Dict[str, str]]],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> Tuple[List[str], List[int], List[int], List[float]]:
    import torch

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    tokenized = maybe_disable_thinking(tokenizer, messages_batch)
    model_device = get_model_device(model)
    input_ids = tokenized["input_ids"].to(model_device)
    attention_mask = tokenized["attention_mask"].to(model_device)

    generation_kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    input_lengths = attention_mask.sum(dim=1).tolist()
    decoded_texts: List[str] = []
    input_tokens: List[int] = []
    output_tokens: List[int] = []
    runtimes: List[float] = []
    avg_runtime = elapsed / max(len(messages_batch), 1)
    for index in range(len(messages_batch)):
        input_len = int(input_lengths[index])
        generated_ids = output_ids[index][input_len:]
        decoded_texts.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
        input_tokens.append(input_len)
        output_tokens.append(int(generated_ids.shape[-1]))
        runtimes.append(avg_runtime)
    return decoded_texts, input_tokens, output_tokens, runtimes


def load_model(
    model_alias: str,
    model_cfg: Dict[str, Any],
    options: RunnerOptions,
    logger: Optional[RunLogger] = None,
) -> Tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = model_cfg["path"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
    dtype_name = options.torch_dtype or model_cfg.get("torch_dtype", "bfloat16")
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    torch_dtype = getattr(torch, dtype_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=options.device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    print(f"[INFO] Loaded model '{model_alias}' from: {model_path}")
    if logger is not None:
        logger.info(
            "model_loaded",
            model_alias=model_alias,
            model_path=model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype_name,
        )
    return tokenizer, model


def release_model(model: Any) -> None:
    try:
        import torch

        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        gc.collect()


def write_footer(path: Path, footer_items: Iterable[Tuple[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write("\nmetric,value\n")
        for key, value in footer_items:
            file.write(f"{key},{value}\n")


def validate_dataset_frame(dataset_name: str, split_name: str, frame: pd.DataFrame) -> None:
    required = {"text", "label"}
    missing = required.difference(set(frame.columns))
    if missing:
        raise ValueError(f"Dataset '{dataset_name}' split '{split_name}' missing columns: {sorted(missing)}")


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def run_single_dataset(
    *,
    model_alias: str,
    strategy: BaseStrategy,
    dataset_spec: DatasetSpec,
    options: RunnerOptions,
    tokenizer: Any,
    model: Any,
    logger: Optional[RunLogger] = None,
) -> Optional[Path]:
    dataset_name = dataset_spec.name
    dataset_cfg = dataset_spec.config
    processor = dataset_spec.processor

    strategy_output_dir = options.output_root / strategy.name / model_alias / "result"
    strategy_output_dir.mkdir(parents=True, exist_ok=True)

    temp_context_for_suffix = StrategyRuntimeContext(
        processor=processor,
        dataset_cfg=dataset_cfg,
        label_schema=None,
        train_texts=[],
        train_labels_for_context=[],
        shot_k=options.shot_k,
        random_seed=options.random_seed,
        sc_num_samples=options.sc_num_samples,
        sc_temperature=options.sc_temperature,
        sc_top_p=options.sc_top_p,
        retrieval_backend=options.retrieval_backend,
        retrieval_model_name=options.retrieval_model_name,
        default_do_sample=options.do_sample,
        default_temperature=options.temperature,
        default_top_p=options.top_p,
    )
    file_suffix = strategy.file_suffix(temp_context_for_suffix)
    output_path = strategy_output_dir / f"{dataset_name}_{strategy.name}{file_suffix}_data_related.csv"
    if output_path.exists() and not options.overwrite:
        print(f"[SKIP] Existing file: {output_path}")
        if logger is not None:
            logger.info(
                "dataset_skipped_existing",
                model=model_alias,
                baseline=strategy.name,
                dataset=dataset_name,
                output_path=str(output_path),
            )
        return output_path

    try:
        train_path, test_path = resolve_dataset_paths(options.datasets_root, dataset_name, dataset_cfg)
    except FileNotFoundError as exc:
        if options.skip_missing_datasets:
            print(f"[SKIP] {exc}")
            if logger is not None:
                logger.warning(
                    "dataset_skipped_missing",
                    model=model_alias,
                    baseline=strategy.name,
                    dataset=dataset_name,
                    message=str(exc),
                )
            return None
        raise

    if logger is not None:
        logger.info(
            "dataset_start",
            model=model_alias,
            baseline=strategy.name,
            dataset=dataset_name,
            train_path=str(train_path),
            test_path=str(test_path),
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    validate_dataset_frame(dataset_name, "train", train_df)
    validate_dataset_frame(dataset_name, "test", test_df)

    if options.max_samples is not None:
        test_df = test_df.iloc[: options.max_samples].copy()

    label_schema = processor.create_label_schema(train_df["label"].tolist(), dataset_cfg)

    train_context_labels: List[Any] = []
    for truth in train_df["label"].tolist():
        canonical = processor.canonicalize_truth(truth, dataset_cfg, label_schema)
        display = processor.display_truth(canonical, dataset_cfg, label_schema)
        train_context_labels.append(display)

    runtime_context = StrategyRuntimeContext(
        processor=processor,
        dataset_cfg=dataset_cfg,
        label_schema=label_schema,
        train_texts=train_df["text"].astype(str).tolist(),
        train_labels_for_context=train_context_labels,
        shot_k=options.shot_k,
        random_seed=options.random_seed,
        sc_num_samples=options.sc_num_samples,
        sc_temperature=options.sc_temperature,
        sc_top_p=options.sc_top_p,
        retrieval_backend=options.retrieval_backend,
        retrieval_model_name=options.retrieval_model_name,
        default_do_sample=options.do_sample,
        default_temperature=options.temperature,
        default_top_p=options.top_p,
    )

    try:
        strategy.prepare_dataset(runtime_context)
    except Exception as exc:
        if logger is not None:
            logger.error(
                "strategy_prepare_failed",
                model=model_alias,
                baseline=strategy.name,
                dataset=dataset_name,
                message=str(exc),
                traceback=traceback.format_exc(),
            )
        raise
    if strategy.name == "few-shot-cot":
        retriever_debug_info = runtime_context.state.get("retriever_debug_info", {})
        selected_backend = ""
        fallback_reason = ""
        if isinstance(retriever_debug_info, dict):
            selected_backend = str(retriever_debug_info.get("selected_backend", ""))
            fallback_reason = str(retriever_debug_info.get("fallback_reason", ""))
        print(
            f"[INFO] Retriever dataset={dataset_name} backend={selected_backend or 'unknown'} "
            f"fallback_reason={fallback_reason or '-'}"
        )
    if logger is not None and strategy.name == "few-shot-cot":
        retriever_debug_info = runtime_context.state.get("retriever_debug_info", {})
        payload = {
            "model": model_alias,
            "baseline": strategy.name,
            "dataset": dataset_name,
            **(retriever_debug_info if isinstance(retriever_debug_info, dict) else {}),
        }
        fallback_reason = str(payload.get("fallback_reason", ""))
        requested_backend = str(payload.get("requested_backend", ""))
        selected_backend = str(payload.get("selected_backend", ""))
        if requested_backend == "semantic" and selected_backend != "semantic":
            logger.warning("retriever_init", message="semantic retriever unavailable, fallback to bm25", **payload)
        elif fallback_reason.startswith("semantic_load_failed"):
            logger.warning("retriever_init", message="semantic retriever load failed, fallback to bm25", **payload)
        else:
            logger.info("retriever_init", message="retriever initialized", **payload)
    engine = InferenceEngine(tokenizer=tokenizer, model=model, options=options)

    texts = test_df["text"].astype(str).tolist()
    raw_truths = test_df["label"].tolist()

    internal_truths: List[Any] = []
    internal_predictions: List[Any] = []
    display_truths: List[Any] = []
    display_predictions: List[Any] = []
    pred_raw_list: List[str] = []
    input_tokens_list: List[int] = []
    output_tokens_list: List[int] = []
    runtime_list: List[float] = []
    strategy_extra_columns: Dict[str, List[Any]] = {name: [] for name in strategy.extra_columns}

    progress_desc = f"{model_alias}::{strategy.name}::{dataset_name}"
    for start in tqdm(range(0, len(texts), options.batch_size), desc=progress_desc, unit="batch"):
        end = min(start + options.batch_size, len(texts))
        batch_texts = texts[start:end]
        batch_truths = raw_truths[start:end]
        batch_indices = list(range(start, end))

        try:
            strategy_outputs = strategy.run_batch(
                batch_texts=batch_texts,
                batch_truths=batch_truths,
                batch_indices=batch_indices,
                context=runtime_context,
                engine=engine,
            )
        except Exception as exc:
            if logger is not None:
                logger.error(
                    "dataset_batch_error",
                    model=model_alias,
                    baseline=strategy.name,
                    dataset=dataset_name,
                    batch_start=start,
                    batch_end=end,
                    message=str(exc),
                    traceback=traceback.format_exc(),
                )
            strategy_outputs = [
                StrategySampleOutput(
                    raw_prediction=f"ERROR: {exc}",
                    parsed_prediction=None,
                    input_tokens=0,
                    output_tokens=0,
                    runtime=0.0,
                    extra_columns={name: "" for name in strategy.extra_columns},
                )
                for _ in batch_texts
            ]

        if len(strategy_outputs) != len(batch_texts):
            raise ValueError(
                f"Strategy '{strategy.name}' returned {len(strategy_outputs)} rows, expected {len(batch_texts)}."
            )

        for truth, output in zip(batch_truths, strategy_outputs):
            canonical_truth = processor.canonicalize_truth(truth, dataset_cfg, label_schema)
            canonical_prediction = processor.canonicalize_prediction(
                output.parsed_prediction, dataset_cfg, label_schema
            )
            display_truth = processor.display_truth(canonical_truth, dataset_cfg, label_schema)
            display_prediction = processor.display_prediction(canonical_prediction, dataset_cfg, label_schema)

            internal_truths.append(canonical_truth)
            internal_predictions.append(canonical_prediction)
            display_truths.append(display_truth)
            display_predictions.append(display_prediction)
            pred_raw_list.append(_stringify_value(output.raw_prediction))
            input_tokens_list.append(int(output.input_tokens))
            output_tokens_list.append(int(output.output_tokens))
            runtime_list.append(float(output.runtime))
            for extra_name in strategy.extra_columns:
                strategy_extra_columns[extra_name].append(output.extra_columns.get(extra_name, ""))

    if len(display_truths) == 0:
        print(f"[WARN] Empty test split for dataset '{dataset_name}'.")
        if logger is not None:
            logger.warning(
                "dataset_empty_test_split",
                model=model_alias,
                baseline=strategy.name,
                dataset=dataset_name,
            )
        return None

    metrics, extra_footer = processor.evaluate(internal_truths, internal_predictions)
    total_tokens_list = [a + b for a, b in zip(input_tokens_list, output_tokens_list)]

    result_rows: List[Dict[str, Any]] = []
    for index in range(len(display_truths)):
        row = {
            "Text": texts[index],
            "truth": display_truths[index],
            "pre": display_predictions[index],
            "pred_raw": pred_raw_list[index],
            "input_tokens": input_tokens_list[index],
            "output_tokens": output_tokens_list[index],
            "total_tokens": total_tokens_list[index],
            "runtime": runtime_list[index],
        }
        for metric_name in processor.metric_columns:
            row[metric_name] = metrics.get(metric_name, float("nan"))
        for extra_name in strategy.extra_columns:
            row[extra_name] = strategy_extra_columns[extra_name][index]
        result_rows.append(row)

    output_columns = list(PUBLIC_RESULT_COLUMNS) + list(processor.metric_columns) + list(strategy.extra_columns)
    result_df = pd.DataFrame(result_rows)[output_columns]
    result_df.to_csv(output_path, index=False)

    total_input_tokens = sum(input_tokens_list)
    total_output_tokens = sum(output_tokens_list)
    total_tokens = total_input_tokens + total_output_tokens
    total_runtime = sum(runtime_list)
    sample_count = len(result_df)

    footer_items: List[Tuple[str, Any]] = [
        ("task", processor.task_name),
        ("strategy", strategy.name),
        ("model", model_alias),
        ("dataset", dataset_name),
        ("num_samples", sample_count),
    ]
    footer_items.extend(strategy.footer_items(runtime_context))
    footer_items.extend([(name, metrics.get(name, float("nan"))) for name in processor.metric_columns])
    for key in sorted(extra_footer.keys()):
        footer_items.append((key, extra_footer[key]))
    footer_items.extend(
        [
            ("total_input_tokens", total_input_tokens),
            ("total_output_tokens", total_output_tokens),
            ("total_tokens", total_tokens),
            ("total_runtime_sec", total_runtime),
            ("avg_input_tokens", safe_divide(total_input_tokens, sample_count)),
            ("avg_output_tokens", safe_divide(total_output_tokens, sample_count)),
            ("avg_total_tokens", safe_divide(total_tokens, sample_count)),
            ("avg_runtime_sec", safe_divide(total_runtime, sample_count)),
        ]
    )
    write_footer(output_path, footer_items)
    if logger is not None:
        logger.info(
            "dataset_finished",
            model=model_alias,
            baseline=strategy.name,
            dataset=dataset_name,
            output_path=str(output_path),
            num_samples=sample_count,
            metrics=metrics,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_tokens=total_tokens,
            total_runtime_sec=total_runtime,
        )
    return output_path


def run_cli(default_baselines: Optional[Sequence[str]] = None) -> None:
    registry = load_task_registry()
    args = parse_args(default_baselines=default_baselines)
    options = build_runner_options(args, registry)
    run_id = build_run_id()
    logger = RunLogger(options.log_dir, run_id=run_id)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    run_results: List[Dict[str, Any]] = []
    run_failures: List[Dict[str, Any]] = []
    configured_cuda_device = ""

    logger.info(
        "run_started",
        message="Experiment run started",
        options=options_to_dict(options),
        cli_args=vars(args),
        available_baselines=sorted(STRATEGY_REGISTRY.keys()),
        available_models=sorted(MODEL_REGISTRY.keys()),
        available_datasets=sorted(registry.keys()),
    )

    try:
        unknown_baselines = [name for name in options.baselines if name not in STRATEGY_REGISTRY]
        if unknown_baselines:
            raise ValueError(
                f"Unknown baselines: {unknown_baselines}. Supported: {sorted(STRATEGY_REGISTRY.keys())}"
            )
        unknown_models = [name for name in options.models if name not in MODEL_REGISTRY]
        if unknown_models:
            raise ValueError(f"Unknown model aliases: {unknown_models}. Supported: {sorted(MODEL_REGISTRY.keys())}")
        unknown_datasets = [name for name in options.datasets if name not in registry]
        if unknown_datasets:
            raise ValueError(f"Unknown datasets: {unknown_datasets}. Supported: {sorted(registry.keys())}")

        configured_cuda_device = configure_single_gpu(options)
        logger.info(
            "gpu_configured",
            message="Single-GPU runtime configured",
            cuda_visible_devices=configured_cuda_device,
        )

        print(f"[INFO] Baselines: {options.baselines}")
        print(f"[INFO] Models: {options.models}")
        print(f"[INFO] Datasets root: {options.datasets_root}")
        print(f"[INFO] Output root: {options.output_root}")
        print(f"[INFO] Log root: {options.log_dir}")
        print(f"[INFO] Datasets: {options.datasets}")
        print(f"[INFO] Dry-run: {options.dry_run}")
        print(f"[INFO] Overwrite: {options.overwrite}")
        print(f"[INFO] Skip missing datasets: {options.skip_missing_datasets}")
        print(f"[INFO] CUDA_VISIBLE_DEVICES: {configured_cuda_device}")
        logger.info(
            "run_plan",
            message="Resolved run matrix",
            baselines=options.baselines,
            models=options.models,
            datasets=options.datasets,
        )

        for model_alias in options.models:
            tokenizer = None
            model = None
            logger.info("model_started", message="Model loop started", model=model_alias)
            try:
                if not options.dry_run:
                    try:
                        tokenizer, model = load_model(
                            model_alias,
                            MODEL_REGISTRY[model_alias],
                            options,
                            logger=logger,
                        )
                    except Exception as exc:
                        logger.error(
                            "model_load_failed",
                            message=str(exc),
                            model=model_alias,
                            traceback=traceback.format_exc(),
                        )
                        run_failures.append(
                            {
                                "stage": "model_load",
                                "model": model_alias,
                                "message": str(exc),
                            }
                        )
                        continue

                for baseline_name in options.baselines:
                    strategy = STRATEGY_REGISTRY[baseline_name]
                    logger.info(
                        "baseline_started",
                        message="Baseline loop started",
                        model=model_alias,
                        baseline=baseline_name,
                    )
                    for dataset_name in options.datasets:
                        dataset_spec = registry[dataset_name]
                        dataset_started = time.perf_counter()
                        try:
                            output_path = run_single_dataset(
                                model_alias=model_alias,
                                strategy=strategy,
                                dataset_spec=dataset_spec,
                                options=options,
                                tokenizer=tokenizer,
                                model=model,
                                logger=logger,
                            )
                            elapsed = time.perf_counter() - dataset_started
                            run_results.append(
                                {
                                    "model": model_alias,
                                    "baseline": baseline_name,
                                    "dataset": dataset_name,
                                    "status": "saved" if output_path is not None else "skipped",
                                    "elapsed_sec": elapsed,
                                    "output_path": str(output_path) if output_path else None,
                                }
                            )
                            if output_path is not None:
                                print(f"[OK] Saved: {output_path}")
                        except Exception as exc:
                            elapsed = time.perf_counter() - dataset_started
                            print(
                                f"[ERROR] model={model_alias} baseline={baseline_name} dataset={dataset_name}: {exc}"
                            )
                            logger.error(
                                "dataset_failed",
                                message=str(exc),
                                model=model_alias,
                                baseline=baseline_name,
                                dataset=dataset_name,
                                elapsed_sec=elapsed,
                                traceback=traceback.format_exc(),
                            )
                            run_failures.append(
                                {
                                    "stage": "dataset_run",
                                    "model": model_alias,
                                    "baseline": baseline_name,
                                    "dataset": dataset_name,
                                    "elapsed_sec": elapsed,
                                    "message": str(exc),
                                }
                            )
                            if isinstance(exc, RetrievalInitializationError):
                                raise
                    logger.info(
                        "baseline_finished",
                        message="Baseline loop finished",
                        model=model_alias,
                        baseline=baseline_name,
                    )
            finally:
                release_model(model)
                logger.info("model_finished", message="Model loop finished", model=model_alias)
    except Exception as exc:
        logger.error("run_fatal", message=str(exc), traceback=traceback.format_exc())
        run_failures.append({"stage": "run", "message": str(exc)})
        raise
    finally:
        finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
        summary = {
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "options": options_to_dict(options),
            "configured_cuda_device": configured_cuda_device,
            "num_success_or_skip": len(run_results),
            "num_failures": len(run_failures),
            "results": run_results,
            "failures": run_failures,
            "events_path": str(logger.events_path),
            "text_log_path": str(logger.text_log_path),
        }
        logger.write_summary(summary)
        logger.info(
            "run_finished",
            message="Experiment run finished",
            num_success_or_skip=len(run_results),
            num_failures=len(run_failures),
            summary_path=str(logger.summary_path),
        )
        print(f"[INFO] Logs: {logger.log_dir}")


if __name__ == "__main__":
    run_cli(default_baselines=DEFAULT_BASELINES)
