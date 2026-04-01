from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from multiagent.adapters.filesystem import FileSystemAdapter
from multiagent.adapters.gemini import GeminiGateway
from multiagent.config import Settings
from multiagent.domain.models import ExecutionPhase, RepoContext, RepoContextFile, Subtask
from multiagent.services.prompts import PromptRegistry
from multiagent.utils import compact_text, sha256_file


class FileSelectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_paths: list[str] = Field(default_factory=list)
    rationale: str


class RepoContextSelector:
    def __init__(
        self,
        settings: Settings,
        fs: FileSystemAdapter,
        prompts: PromptRegistry,
        gateway: GeminiGateway,
    ) -> None:
        self._settings = settings
        self._fs = fs
        self._prompts = prompts
        self._gateway = gateway

    def summarize_repo(self, repo_path: Path, limit: int = 100) -> str:
        files = [
            path.relative_to(repo_path).as_posix()
            for path in self._fs.list_files(repo_path)
            if self._include_file(path)
        ][:limit]
        return "\n".join(files)

    async def select_context(self, *, goal: str, subtask: Subtask, repo_path: Path) -> RepoContext:
        files = [path for path in self._fs.list_files(repo_path) if self._include_file(path)]
        local_candidates = self._score_files(goal, subtask, repo_path, files)
        shortlisted = local_candidates[: self._settings.local_selection_candidate_limit]
        if len(shortlisted) > 6:
            selection_prompt = self._prompts.file_selector(
                goal=goal,
                subtask=subtask,
                file_summaries=[
                    {
                        "path": item.relative_to(repo_path).as_posix(),
                        "summary": compact_text(self._fs.read_file(item, max_chars=600), 320),
                    }
                    for item in shortlisted
                ],
            )
            response = await self._gateway.generate_structured(
                model=self._settings.planner_model,
                system_instruction=selection_prompt.system_instruction,
                user_prompt=selection_prompt.user_prompt,
                schema=FileSelectionResponse,
                phase=ExecutionPhase.PLANNING,
            )
            selected_lookup = set(response.parsed.selected_paths)
            final_paths = [
                path for path in shortlisted if path.relative_to(repo_path).as_posix() in selected_lookup
            ] or shortlisted[:6]
            rationale = response.parsed.rationale
        else:
            final_paths = shortlisted[:6]
            rationale = "Local heuristic selection based on path, keyword, and content relevance."
        selected_files = [
            RepoContextFile(
                path=path.relative_to(repo_path).as_posix(),
                sha256=sha256_file(path),
                summary=compact_text(self._fs.read_file(path, max_chars=1200), 300),
                excerpt=self._fs.read_file(path, max_chars=3000),
            )
            for path in final_paths
        ]
        return RepoContext(
            repo_path=str(repo_path),
            selected_files=selected_files,
            selection_reason=rationale,
        )

    def _score_files(self, goal: str, subtask: Subtask, repo_path: Path, files: list[Path]) -> list[Path]:
        keywords = {
            token.lower()
            for token in re.findall(r"[a-zA-Z_]{4,}", " ".join([goal, subtask.title, subtask.objective]))
        }
        scored: list[tuple[float, Path]] = []
        for path in files:
            rel = path.relative_to(repo_path).as_posix().lower()
            score = 0.0
            for keyword in keywords:
                if keyword in rel:
                    score += 2.0
            try:
                content = self._fs.read_file(path, max_chars=4000).lower()
            except OSError:
                continue
            for keyword in keywords:
                if keyword in content:
                    score += 1.0
            if path.suffix in {".py", ".toml", ".md", ".yaml", ".yml", ".json"}:
                score += 0.5
            if score > 0:
                scored.append((score, path))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [path for _, path in scored]

    def _include_file(self, path: Path) -> bool:
        blocked = {".git", ".venv", ".venv312", "__pycache__", "node_modules", "runs"}
        return not any(part in blocked for part in path.parts)
