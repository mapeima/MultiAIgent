from multiagent.adapters.filesystem import FileSystemAdapter
from multiagent.domain.models import AgentRole, CandidateSpec, RunMode, TokenUsage, WorkerResult
from multiagent.services.artifact_store import ArtifactStore
from multiagent.services.prompts import PromptRegistry
from multiagent.services.workers import WorkerService


def _worker_result_handler(**kwargs):
    return WorkerResult(
        summary="ok",
        detailed_result="done",
        artifacts=[],
        suggested_files=[],
        code_changes=[],
        risks=[],
        confidence=0.8,
        model_used="gemini-2.5-flash",
        prompt_variant="default",
        token_usage_estimate=TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20),
        follow_up_suggestions=[],
    )


def test_worker_retries_on_schema_failure(base_settings):
    from tests.conftest import FakeGateway, make_subtask

    gateway = FakeGateway(handlers={"WorkerResult": lambda **_: _worker_result_handler()}, fail_once={"WorkerResult"})
    store = ArtifactStore(base_settings, "worker-retry", FileSystemAdapter())
    service = WorkerService(base_settings, PromptRegistry(), gateway, store)
    candidate = CandidateSpec(
        candidate_id="c1",
        subtask_id="s1",
        agent_role=AgentRole.IMPLEMENTER,
        model="gemini-2.5-flash",
        temperature=0.1,
        prompt_variant="default",
        reasoning_style="lean",
        strictness_level="normal",
        benchmark_axes=[],
        mode=RunMode.EFFICIENT,
        repo_context_hashes={},
    )
    import asyncio

    execution = asyncio.run(
        service.execute_candidate(
            goal="test",
            subtask=make_subtask("s1"),
            mode=RunMode.EFFICIENT,
            prior_artifacts=[],
            repo_context=None,
            candidate_spec=candidate,
        )
    )
    assert execution.result.candidate_id == "c1"
    assert len(gateway.calls) == 2
