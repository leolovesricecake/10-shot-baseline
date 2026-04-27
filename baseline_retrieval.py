"""Retrieval backends for few-shot strategies."""

from __future__ import annotations

import math
import re
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from baseline_runner_config import DEFAULT_RETRIEVAL_MODEL_NAME


def retrieval_tokenize(text: str) -> List[str]:
    raw = str(text).strip().lower()
    if not raw:
        return []
    if any(char.isspace() for char in raw):
        tokens = re.findall(r"[a-z0-9_]+", raw)
        if len(tokens) > 0:
            return tokens
        return [token for token in raw.split() if token]
    return [char for char in raw if not char.isspace()]


class BaseRetriever:
    backend_name: str = "base"

    def query(self, text: str, top_k: int) -> Tuple[List[int], List[float]]:
        raise NotImplementedError


class RetrievalInitializationError(RuntimeError):
    """Raised when retriever initialization fails and should be treated as fatal."""


@dataclass
class BM25Retriever(BaseRetriever):
    tokenized_docs: List[List[str]]
    doc_freq: dict
    avg_doc_len: float
    k1: float = 1.5
    b: float = 0.75
    backend_name: str = "bm25"

    @classmethod
    def build(cls, train_texts: Sequence[str]) -> "BM25Retriever":
        tokenized_docs = [retrieval_tokenize(text) for text in train_texts]
        doc_freq = defaultdict(int)
        for tokens in tokenized_docs:
            for token in set(tokens):
                doc_freq[token] += 1
        avg_len = float(sum(len(tokens) for tokens in tokenized_docs)) / max(len(tokenized_docs), 1)
        return cls(tokenized_docs=tokenized_docs, doc_freq=dict(doc_freq), avg_doc_len=avg_len)

    def query(self, text: str, top_k: int) -> Tuple[List[int], List[float]]:
        query_tokens = retrieval_tokenize(text)
        if len(query_tokens) == 0:
            return [], []
        total_docs = len(self.tokenized_docs)
        if total_docs == 0:
            return [], []

        scores = []
        query_counter = Counter(query_tokens)
        for doc_idx, doc_tokens in enumerate(self.tokenized_docs):
            doc_len = len(doc_tokens)
            doc_tf = Counter(doc_tokens)
            score = 0.0
            for token, q_count in query_counter.items():
                if token not in doc_tf:
                    continue
                tf = doc_tf[token]
                df = self.doc_freq.get(token, 0)
                idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(self.avg_doc_len, 1e-8)))
                score += idf * tf * (self.k1 + 1.0) / max(denom, 1e-8) * q_count
            scores.append((doc_idx, float(score)))

        scores.sort(key=lambda item: item[1], reverse=True)
        top_items = scores[: max(0, top_k)]
        return [item[0] for item in top_items], [item[1] for item in top_items]


@dataclass
class SemanticRetriever(BaseRetriever):
    model: any
    embeddings: np.ndarray
    backend_name: str = "semantic"

    @classmethod
    def build(
        cls,
        train_texts: Sequence[str],
        model_name: str = DEFAULT_RETRIEVAL_MODEL_NAME,
        batch_size: int = 64,
    ) -> "SemanticRetriever":
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            list(train_texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        return cls(model=model, embeddings=embeddings)

    def query(self, text: str, top_k: int) -> Tuple[List[int], List[float]]:
        if self.embeddings.shape[0] == 0:
            return [], []
        query_vec = self.model.encode(  # type: ignore[attr-defined]
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        scores = np.matmul(self.embeddings, np.asarray(query_vec[0], dtype=np.float32))
        top_k = min(max(0, top_k), scores.shape[0])
        if top_k <= 0:
            return [], []
        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        sorted_indices = top_indices[np.argsort(-scores[top_indices])]
        return sorted_indices.tolist(), scores[sorted_indices].astype(float).tolist()


def build_retriever(
    train_texts: Sequence[str],
    backend: str = "semantic",
    semantic_model_name: str = DEFAULT_RETRIEVAL_MODEL_NAME,
) -> Tuple[BaseRetriever, str]:
    backend = backend.lower().strip()
    if backend not in {"semantic", "bm25"}:
        raise ValueError(f"Unsupported retrieval backend: {backend}")

    model_path = Path(str(semantic_model_name)).expanduser()
    debug_info = {
        "requested_backend": backend,
        "selected_backend": None,
        "semantic_model_name": str(semantic_model_name),
        "semantic_model_path_exists": bool(model_path.exists()),
        "semantic_load_success": False,
        "fallback_reason": None,
    }

    if backend == "semantic":
        if not model_path.exists():
            debug_info["selected_backend"] = "semantic"
            debug_info["fallback_reason"] = "semantic_model_path_not_found"
            raise RetrievalInitializationError(
                f"Semantic retrieval model path not found: {model_path}"
            )
        try:
            retriever = SemanticRetriever.build(train_texts, model_name=str(model_path))
        except Exception as exc:
            debug_info["selected_backend"] = "semantic"
            debug_info["fallback_reason"] = f"semantic_load_failed: {exc}"
            raise RetrievalInitializationError(
                f"Failed to load semantic retriever model from '{model_path}': {exc}"
            ) from exc
        debug_info["selected_backend"] = "semantic"
        debug_info["semantic_load_success"] = True
        setattr(retriever, "debug_info", debug_info)
        return retriever, "semantic"

    retriever = BM25Retriever.build(train_texts)
    debug_info["selected_backend"] = "bm25"
    setattr(retriever, "debug_info", debug_info)
    return retriever, "bm25"
