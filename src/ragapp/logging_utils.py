"""Structured per-query logging: one JSON line per query with latency
breakdown, chunk count, and token usage - as required by the brief."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class QueryLogger:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        record = {"timestamp": time.time(), **record}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def read_all(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        out = []
        with self.log_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
