from __future__ import annotations

from pathlib import Path
from typing import Any

from multiagent.adapters.filesystem import FileSystemAdapter
from multiagent.config import Settings
from multiagent.utils import ensure_directory


class ArtifactStore:
    def __init__(self, settings: Settings, run_id: str, fs: FileSystemAdapter) -> None:
        self.settings = settings
        self.run_id = run_id
        self.fs = fs
        self.root = ensure_directory(settings.artifact_dir / run_id)
        for name in [
            "benchmark",
            "agent_outputs",
            "evaluations",
            "logs",
            "patches",
            "summaries",
        ]:
            ensure_directory(self.root / name)

    def write_json(self, relative_path: str, payload: Any) -> Path:
        path = self.root / relative_path
        self.fs.write_json(path, payload)
        return path

    def write_markdown(self, relative_path: str, payload: str) -> Path:
        path = self.root / relative_path
        self.fs.write_text(path, payload)
        return path

    def write_text(self, relative_path: str, payload: str) -> Path:
        path = self.root / relative_path
        self.fs.write_text(path, payload)
        return path

    def write_agent_output(self, subtask_id: str, candidate_id: str, payload: Any) -> Path:
        return self.write_json(f"agent_outputs/{subtask_id}__{candidate_id}.json", payload)

    def write_evaluation(self, subtask_id: str, payload: Any) -> Path:
        return self.write_json(f"evaluations/{subtask_id}.json", payload)

    def write_patch(self, subtask_id: str, filename: str, patch_text: str) -> Path:
        safe_name = filename.replace("/", "__").replace("\\", "__")
        return self.write_text(f"patches/{subtask_id}__{safe_name}.patch", patch_text)

    def snapshot_settings(self, settings: Settings) -> Path:
        return self.write_json("config_snapshot.json", settings.model_dump(mode="json"))

    def log_path(self) -> Path:
        return self.root / "events.jsonl"

    def metrics_path(self) -> Path:
        return self.root / "metrics.json"

    def cost_report_path(self) -> Path:
        return self.root / "cost_report.json"

    def batch_root(self) -> Path:
        return ensure_directory(self.settings.batch_artifact_dir)
