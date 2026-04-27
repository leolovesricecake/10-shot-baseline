"""Lightweight structured logging utilities for experiment runs."""

from __future__ import annotations

import json
import socket
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def build_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"{timestamp}_{suffix}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


class RunLogger:
    def __init__(self, log_root: Path, run_id: str):
        self.run_id = run_id
        self.host = socket.gethostname()
        self.log_dir = (log_root / run_id).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.log_dir / "events.jsonl"
        self.summary_path = self.log_dir / "run_summary.json"
        self.text_log_path = self.log_dir / "run.log"

    def _now(self) -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def _write_text_line(self, level: str, event: str, payload: Dict[str, Any]) -> None:
        message = payload.get("message")
        if message is None:
            message = ""
        line = f"[{self._now()}] [{level}] [{event}] {message}\n"
        with self.text_log_path.open("a", encoding="utf-8") as file:
            file.write(line)

    def log(self, level: str, event: str, **fields: Any) -> None:
        record = {
            "ts": self._now(),
            "level": str(level).upper(),
            "event": event,
            "run_id": self.run_id,
            "host": self.host,
        }
        safe_fields = _json_safe(fields)
        if isinstance(safe_fields, dict):
            record.update(safe_fields)
        with self.events_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._write_text_line(record["level"], event, record)

    def info(self, event: str, **fields: Any) -> None:
        self.log("INFO", event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self.log("WARNING", event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self.log("ERROR", event, **fields)

    def write_summary(self, payload: Dict[str, Any]) -> None:
        with self.summary_path.open("w", encoding="utf-8") as file:
            json.dump(_json_safe(payload), file, ensure_ascii=False, indent=2)
