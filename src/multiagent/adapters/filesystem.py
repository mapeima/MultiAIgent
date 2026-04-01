from __future__ import annotations

import difflib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from multiagent.utils import compact_text, ensure_directory, sha256_file, stable_json_dumps


class FileSystemAdapter:
    def list_files(self, root: Path, pattern: str = "*") -> list[Path]:
        return sorted(path for path in root.rglob(pattern) if path.is_file())

    def read_file(self, path: Path, max_chars: int = 20_000) -> str:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text[:max_chars]

    def search_text(self, root: Path, query: str, limit: int = 50) -> list[dict[str, Any]]:
        ripgrep = shutil.which("rg")
        if ripgrep:
            completed = subprocess.run(
                [ripgrep, "-n", "--hidden", "--glob", "!.git", query, str(root)],
                capture_output=True,
                text=True,
                check=False,
            )
            matches: list[dict[str, Any]] = []
            for line in completed.stdout.splitlines()[:limit]:
                try:
                    path_str, line_no, excerpt = line.split(":", 2)
                except ValueError:
                    continue
                matches.append(
                    {
                        "path": path_str,
                        "line": int(line_no),
                        "excerpt": compact_text(excerpt, limit=240),
                    }
                )
            return matches

        matches = []
        for path in self.list_files(root):
            try:
                for index, line in enumerate(
                    path.read_text(encoding="utf-8", errors="ignore").splitlines(),
                    start=1,
                ):
                    if query.lower() in line.lower():
                        matches.append(
                            {
                                "path": str(path),
                                "line": index,
                                "excerpt": compact_text(line, limit=240),
                            }
                        )
                        if len(matches) >= limit:
                            return matches
            except OSError:
                continue
        return matches

    def write_json(self, path: Path, payload: Any) -> None:
        ensure_directory(path.parent)
        path.write_text(stable_json_dumps(payload), encoding="utf-8")

    def read_json(self, path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    def write_text(self, path: Path, text: str) -> None:
        ensure_directory(path.parent)
        path.write_text(text, encoding="utf-8")

    def create_patch(
        self,
        *,
        path: Path,
        original: str,
        updated: str,
        from_label: str = "before",
        to_label: str = "after",
    ) -> str:
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=f"{path}:{from_label}",
            tofile=f"{path}:{to_label}",
        )
        return "".join(diff)

    def fingerprint_file(self, path: Path) -> str:
        return sha256_file(path)
