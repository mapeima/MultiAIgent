from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from multiagent.adapters.filesystem import FileSystemAdapter
from multiagent.adapters.subprocess_tools import CommandResult
from multiagent.domain.models import (
    AgentRole,
    ChangeAction,
    ModelTier,
    RepoContext,
    RepoContextFile,
    Subtask,
    TokenUsage,
    WorkerResult,
)
from multiagent.services.artifact_store import ArtifactStore
from multiagent.services.repo_mutation import RepoMutationService


@dataclass
class FakeSubprocessTools:
    calls: list[str] = field(default_factory=list)

    async def run_py_compile(self, repo_path: Path, paths: list[Path], timeout: int = 120) -> CommandResult:
        self.calls.append("py_compile")
        return CommandResult(["py_compile"], 0, "", "")

    async def run_ruff(self, repo_path: Path, timeout: int = 300) -> CommandResult:
        self.calls.append("ruff")
        return CommandResult(["ruff", "check", "."], 0, "", "")

    async def run_pytest(self, repo_path: Path, timeout: int = 600) -> CommandResult:
        self.calls.append("pytest")
        return CommandResult(["pytest", "-q"], 0, "", "")

    async def run_mypy(self, repo_path: Path, timeout: int = 600) -> CommandResult:
        self.calls.append("mypy")
        return CommandResult(["mypy", "."], 0, "", "")


def _subtask() -> Subtask:
    return Subtask(
        id="backend",
        title="Backend scaffolding",
        role=AgentRole.BACKEND,
        objective="Create initial backend files",
        depends_on=[],
        deliverables=["scaffold"],
        acceptance_criteria=["files exist"],
        importance_score=8,
        complexity_score=6,
        parallelizable=True,
        recommended_candidate_count=1,
        recommended_model_tier=ModelTier.BALANCED,
        requires_review=True,
    )


def _result(path: str, content: str) -> WorkerResult:
    return WorkerResult(
        summary="Created files",
        detailed_result="Created initial project files",
        artifacts=[],
        suggested_files=[path],
        code_changes=[
            {
                "path": path,
                "action": ChangeAction.CREATE,
                "content": content,
                "reason": "Initial scaffold",
                "language": "python" if path.endswith(".py") else None,
            }
        ],
        risks=[],
        confidence=0.8,
        model_used="gemini-2.5-flash",
        prompt_variant="default",
        token_usage_estimate=TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20),
        follow_up_suggestions=[],
    )


def test_repo_mutation_greenfield_skips_pytest_and_mypy_without_targets(base_settings, tmp_path):
    repo_path = tmp_path / "new-repo"
    tools = FakeSubprocessTools()
    store = ArtifactStore(base_settings, "repo-mutation-skip", FileSystemAdapter())
    service = RepoMutationService(FileSystemAdapter(), tools, store)

    outcome = asyncio.run(
        service.apply_selected_result(
            repo_path=repo_path,
            subtask=_subtask(),
            repo_context=None,
            result=_result("backend/main.py", "print('ok')\n"),
        )
    )

    assert outcome.applied is True
    assert tools.calls == ["py_compile", "ruff"]
    assert (repo_path / "backend/main.py").exists()
    assert (store.root / "logs/validations__backend.json").exists()


def test_repo_mutation_runs_pytest_and_mypy_when_targets_exist(base_settings, tmp_path):
    repo_path = tmp_path / "configured-repo"
    (repo_path / "tests").mkdir(parents=True)
    (repo_path / "tests/test_smoke.py").write_text("def test_smoke():\n    assert True\n", encoding="utf-8")
    (repo_path / "pyproject.toml").write_text("[tool.mypy]\npython_version = '3.12'\n", encoding="utf-8")
    tools = FakeSubprocessTools()
    store = ArtifactStore(base_settings, "repo-mutation-targets", FileSystemAdapter())
    service = RepoMutationService(FileSystemAdapter(), tools, store)

    outcome = asyncio.run(
        service.apply_selected_result(
            repo_path=repo_path,
            subtask=_subtask(),
            repo_context=None,
            result=_result("backend/main.py", "print('ok')\n"),
        )
    )

    assert outcome.applied is True
    assert tools.calls == ["py_compile", "ruff", "pytest", "mypy"]


def test_repo_mutation_conflict_is_recorded_without_raising(base_settings, tmp_path):
    repo_path = tmp_path / "conflict-repo"
    target = repo_path / "docker-compose.yml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("version: '3'\nservices:\n  api:\n    image: old\n", encoding="utf-8")
    fs = FileSystemAdapter()
    expected_hash = fs.fingerprint_file(target)
    target.write_text("version: '3'\nservices:\n  api:\n    image: newer\n", encoding="utf-8")
    tools = FakeSubprocessTools()
    store = ArtifactStore(base_settings, "repo-mutation-conflict", fs)
    service = RepoMutationService(fs, tools, store)
    context = RepoContext(
        repo_path=str(repo_path),
        selected_files=[
            RepoContextFile(
                path="docker-compose.yml",
                sha256=expected_hash,
                summary="compose",
                excerpt="compose",
            )
        ],
        selection_reason="selected",
    )

    outcome = asyncio.run(
        service.apply_selected_result(
            repo_path=repo_path,
            subtask=_subtask(),
            repo_context=context,
            result=_result("docker-compose.yml", "version: '3'\nservices:\n  api:\n    image: proposed\n"),
        )
    )

    assert outcome.applied is False
    assert outcome.reverted is False
    assert tools.calls == []
    assert target.read_text(encoding="utf-8") == "version: '3'\nservices:\n  api:\n    image: newer\n"
    assert (store.root / "logs/conflicts__backend.json").exists()
    assert (store.root / "patches/backend__docker-compose.yml.patch").exists()
