"""Global configuration for multi-baseline text runner."""

from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parent


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "glm4_9b": {
        "path": "/mnt/huawei/ymb/model/glm-4-9b/model",
        "torch_dtype": "bfloat16",
        "trust_remote_code": True,
    },
    "llama3_1_8b": {
        "path": "/mnt/huawei/ymb/model/Llama-3.1-8B/model",
        "torch_dtype": "bfloat16",
        "trust_remote_code": True,
    },
    "qwen3_5_4b": {
        "path": "/mnt/huawei/ymb/model/qwen3.5-4b/model",
        "torch_dtype": "bfloat16",
        "trust_remote_code": True,
    },
    "qwen3_5_9b": {
        "path": "/mnt/huawei/ymb/model/qwen3.5-9b/model",
        "torch_dtype": "bfloat16",
        "trust_remote_code": True,
    },
}


DEFAULT_MODELS: List[str] = ["glm4_9b", "llama3_1_8b", "qwen3_5_4b", "qwen3_5_9b"]


DEFAULT_BASELINES: List[str] = ["cot", "sc-cot", "sv-cot", "few-shot-cot", "random", "fixed"]


DEFAULT_RETRIEVAL_MODEL_NAME: str = (
    "/mnt/huawei/ymb/model/models--sentence-transformers--all-MiniLM-L6-v2/"
    "snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
)


PUBLIC_RESULT_COLUMNS: List[str] = [
    "Text",
    "truth",
    "pre",
    "pred_raw",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "runtime",
]


TASK_ENTRY_PATHS: List[Path] = [
    REPO_ROOT / "classfication" / "task_entry.py",
    REPO_ROOT / "similarity" / "task_entry.py",
    REPO_ROOT / "summary" / "task_entry.py",
    REPO_ROOT / "translation" / "task_entry.py",
    REPO_ROOT / "kuoxie" / "task_entry.py",
]
