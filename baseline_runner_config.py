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
}


DEFAULT_MODELS: List[str] = ["glm4_9b", "llama3_1_8b"]


DEFAULT_BASELINES: List[str] = ["cot", "sc-cot", "sv-cot", "few-shot-cot", "random", "fixed"]


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
