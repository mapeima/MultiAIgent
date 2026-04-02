from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from multiagent.adapters.filesystem import FileSystemAdapter
from multiagent.adapters.subprocess_tools import CommandResult, SubprocessTools
from multiagent.domain.models import ChangeAction, RepoContext, Subtask, WorkerResult
from multiagent.errors import RepoMutationError
from multiagent.services.artifact_store import ArtifactStore


@dataclass(slots=True)
class RepoMutationOutcome:
    applied: bool
    reverted: bool
    validation_results: list[CommandResult]
    message: str


class RepoMutationService:
    _PYTHON_SUFFIXES = {".py"}
    _TEST_FILE_PATTERNS = ("test_*.py", "*_test.py")
    _MYPY_CONFIG_FILES = ("mypy.ini", ".mypy.ini")

    def __init__(
        self,
        fs: FileSystemAdapter,
        subprocess_tools: SubprocessTools,
        artifact_store: ArtifactStore,
    ) -> None:
        self._fs = fs
        self._subprocess = subprocess_tools
        self._artifact_store = artifact_store

    async def apply_selected_result(
        self,
        *,
        repo_path: Path,
        subtask: Subtask,
        repo_context: RepoContext | None,
        result: WorkerResult,
    ) -> RepoMutationOutcome:
        if not result.code_changes:
            return RepoMutationOutcome(True, False, [], "No code changes to apply.")

        repo_path.mkdir(parents=True, exist_ok=True)
        original_contents: dict[Path, str | None] = {}
        expected_hashes = {item.path: item.sha256 for item in (repo_context.selected_files if repo_context else [])}
        changed_paths: list[Path] = []
        conflict_details: list[dict[str, str | None]] = []

        for change in result.code_changes:
            absolute_path = repo_path / change.path
            changed_paths.append(absolute_path)
            if absolute_path.exists():
                original_contents[absolute_path] = absolute_path.read_text(encoding="utf-8", errors="ignore")
            else:
                original_contents[absolute_path] = None
            if change.path in expected_hashes and absolute_path.exists():
                current_hash = self._fs.fingerprint_file(absolute_path)
                if current_hash != expected_hashes[change.path]:
                    conflict_details.append(
                        {
                            "path": change.path,
                            "expected_sha256": expected_hashes[change.path],
                            "current_sha256": current_hash,
                        }
                    )
        if conflict_details:
            self._write_conflict_artifacts(
                subtask=subtask,
                result=result,
                original_contents=original_contents,
                conflict_details=conflict_details,
                repo_path=repo_path,
            )
            return RepoMutationOutcome(
                applied=False,
                reverted=False,
                validation_results=[],
                message="Conflict detected; skipped applying changes because the repo changed since context selection.",
            )

        try:
            for change in result.code_changes:
                absolute_path = repo_path / change.path
                if change.action is ChangeAction.DELETE:
                    if absolute_path.exists():
                        patch = self._fs.create_patch(
                            path=absolute_path,
                            original=original_contents[absolute_path] or "",
                            updated="",
                        )
                        self._artifact_store.write_patch(subtask.id, change.path, patch)
                        absolute_path.unlink()
                    continue
                if change.content is None:
                    raise RepoMutationError(f"Code change for {change.path} is missing content")
                absolute_path.parent.mkdir(parents=True, exist_ok=True)
                original = original_contents[absolute_path] or ""
                patch = self._fs.create_patch(path=absolute_path, original=original, updated=change.content)
                self._artifact_store.write_patch(subtask.id, change.path, patch)
                absolute_path.write_text(change.content, encoding="utf-8")

            validations = await self._run_validations(repo_path, changed_paths)
            self._artifact_store.write_json(
                f"logs/validations__{subtask.id}.json",
                {
                    "repo_path": str(repo_path),
                    "subtask_id": subtask.id,
                    "commands": [
                        {
                            "command": item.command,
                            "returncode": item.returncode,
                            "stdout": item.stdout,
                            "stderr": item.stderr,
                        }
                        for item in validations
                    ],
                },
            )
            failed = [command for command in validations if not command.ok]
            if failed:
                await self._revert(original_contents)
                return RepoMutationOutcome(
                    applied=False,
                    reverted=True,
                    validation_results=validations,
                    message="Validation failed; reverted applied changes.",
                )
            return RepoMutationOutcome(
                applied=True,
                reverted=False,
                validation_results=validations,
                message="Changes applied and validation passed.",
            )
        except Exception:
            await self._revert(original_contents)
            raise

    async def _revert(self, original_contents: dict[Path, str | None]) -> None:
        for path, original in original_contents.items():
            if original is None:
                if path.exists():
                    path.unlink()
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(original, encoding="utf-8")

    async def _run_validations(self, repo_path: Path, changed_paths: list[Path]) -> list[CommandResult]:
        validations: list[CommandResult] = []
        python_files = [path for path in changed_paths if path.suffix in self._PYTHON_SUFFIXES]
        if python_files:
            validations.append(await self._subprocess.run_py_compile(repo_path, python_files))
            validations.append(await self._subprocess.run_ruff(repo_path))
        if self._has_pytest_targets(repo_path):
            validations.append(await self._subprocess.run_pytest(repo_path))
        if self._has_mypy_targets(repo_path, changed_paths):
            validations.append(await self._subprocess.run_mypy(repo_path))
        if not validations:
            validations.append(
                CommandResult(
                    command=["validation-skip"],
                    returncode=0,
                    stdout="",
                    stderr="No applicable validations for current mutation.",
                )
            )
        return validations

    def _has_pytest_targets(self, repo_path: Path) -> bool:
        return any(self._iter_matching_files(repo_path, self._TEST_FILE_PATTERNS))

    def _has_mypy_targets(self, repo_path: Path, changed_paths: list[Path]) -> bool:
        if not any(path.suffix in self._PYTHON_SUFFIXES for path in changed_paths):
            return False
        if any((repo_path / filename).exists() for filename in self._MYPY_CONFIG_FILES):
            return True
        pyproject = repo_path / "pyproject.toml"
        if not pyproject.exists():
            return False
        content = pyproject.read_text(encoding="utf-8", errors="ignore")
        return "[tool.mypy]" in content

    def _iter_matching_files(self, repo_path: Path, patterns: Iterable[str]) -> Iterable[Path]:
        for pattern in patterns:
            yield from (
                path
                for path in repo_path.rglob(pattern)
                if path.is_file() and not self._is_ignored_path(path.relative_to(repo_path))
            )

    def _is_ignored_path(self, relative_path: Path) -> bool:
        blocked = {".git", ".venv", ".venv312", "__pycache__", "node_modules", "runs"}
        return any(part in blocked for part in relative_path.parts)

    def _write_conflict_artifacts(
        self,
        *,
        subtask: Subtask,
        result: WorkerResult,
        original_contents: dict[Path, str | None],
        conflict_details: list[dict[str, str | None]],
        repo_path: Path,
    ) -> None:
        for change in result.code_changes:
            absolute_path = repo_path / change.path
            original = original_contents.get(absolute_path) or ""
            updated = "" if change.action is ChangeAction.DELETE else (change.content or "")
            patch = self._fs.create_patch(path=absolute_path, original=original, updated=updated)
            self._artifact_store.write_patch(subtask.id, change.path, patch)
        self._artifact_store.write_json(
            f"logs/conflicts__{subtask.id}.json",
            {
                "repo_path": str(repo_path),
                "subtask_id": subtask.id,
                "message": "Repo files changed after context selection; mutation skipped.",
                "conflicts": conflict_details,
            },
        )
