from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

        original_contents: dict[Path, str | None] = {}
        expected_hashes = {item.path: item.sha256 for item in (repo_context.selected_files if repo_context else [])}

        for change in result.code_changes:
            absolute_path = repo_path / change.path
            if absolute_path.exists():
                original_contents[absolute_path] = absolute_path.read_text(encoding="utf-8", errors="ignore")
            else:
                original_contents[absolute_path] = None
            if change.path in expected_hashes and absolute_path.exists():
                current_hash = self._fs.fingerprint_file(absolute_path)
                if current_hash != expected_hashes[change.path]:
                    raise RepoMutationError(
                        f"Conflict applying {change.path}: file changed since context selection."
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

            validations = await self._run_validations(repo_path)
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

    async def _run_validations(self, repo_path: Path) -> list[CommandResult]:
        return [
            await self._subprocess.run_pytest(repo_path),
            await self._subprocess.run_ruff(repo_path),
            await self._subprocess.run_mypy(repo_path),
        ]
