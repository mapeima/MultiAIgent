from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from multiagent.utils import ensure_directory, stable_json_dumps, utc_now


class EventLogger:
    def __init__(self, path: Path, run_id: str) -> None:
        ensure_directory(path.parent)
        self._path = path
        self._run_id = run_id
        self._lock = Lock()

    def log(self, **payload: Any) -> None:
        record = {
            "timestamp": utc_now().isoformat(),
            "run_id": self._run_id,
            **payload,
        }
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True, default=str))
                handle.write("\n")


@dataclass(slots=True)
class MetricsTracker:
    counters: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    histograms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def increment(self, name: str, amount: float = 1.0) -> None:
        self.counters[name] += amount

    def observe(self, name: str, value: float) -> None:
        self.histograms[name].append(value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "counters": dict(self.counters),
            "histograms": {key: value[:] for key, value in self.histograms.items()},
        }

    def write(self, path: Path) -> None:
        ensure_directory(path.parent)
        path.write_text(stable_json_dumps(self.as_dict()), encoding="utf-8")
