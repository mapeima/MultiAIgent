import asyncio

from multiagent.adapters.filesystem import FileSystemAdapter
from multiagent.adapters.logging import EventLogger, MetricsTracker
from multiagent.adapters.pricing import PriceBook
from multiagent.domain.models import (
    AgentRole,
    ModelTier,
    ReviewVerdict,
    RunMode,
    RunReport,
    RunRequest,
    Subtask,
    SubtaskSelection,
    TokenUsage,
    WorkerResult,
)
from multiagent.services.artifact_store import ArtifactStore
from multiagent.services.orchestrator import Orchestrator

from tests.conftest import DummyBudget, DummyReviewer, DummySynthesizer, make_plan


class StubPlanner:
    async def create_plan(self, request, repo_summary=None):
        return make_plan(
            [
                Subtask(
                    id="base",
                    title="Base task",
                    role=AgentRole.IMPLEMENTER,
                    objective="Do it",
                    depends_on=[],
                    deliverables=["x"],
                    acceptance_criteria=["done"],
                    importance_score=8,
                    complexity_score=6,
                    parallelizable=True,
                    recommended_candidate_count=1,
                    recommended_model_tier=ModelTier.BALANCED,
                    requires_review=True,
                )
            ]
        )


class StubScheduler:
    def __init__(self):
        self.calls = 0

    async def execute(self, *, plan, existing_selections=None, **kwargs):
        self.calls += 1
        subtask = plan.subtasks[0]
        selection = SubtaskSelection(
            subtask_id=subtask.id,
            selected_candidate_id=f"{subtask.id}-cand-1",
            selected_result=WorkerResult(
                summary="ok",
                detailed_result="detail",
                artifacts=[],
                suggested_files=[],
                code_changes=[],
                risks=[],
                confidence=0.8,
                model_used="gemini-2.5-flash",
                prompt_variant="default",
                token_usage_estimate=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                follow_up_suggestions=[],
            ),
            candidate_results=[],
            evaluations=[],
            merged_candidate_ids=[],
        )
        return [selection]


class StubBatch:
    async def submit(self, **kwargs):
        raise AssertionError("not used")

    async def reconcile(self, batch_id):
        raise AssertionError("not used")


class HarnessOrchestrator(Orchestrator):
    def __init__(self, settings):
        super().__init__(settings)
        self.scheduler = StubScheduler()
        self.reviewer = DummyReviewer()
        self.synth = DummySynthesizer()

    def _build_runtime(self, run_id):
        artifact_store = ArtifactStore(self._settings, run_id, FileSystemAdapter())
        logger = EventLogger(artifact_store.log_path(), run_id)
        metrics = MetricsTracker()
        planner = StubPlanner()
        return (
            artifact_store,
            logger,
            metrics,
            None,
            DummyBudget(),
            planner,
            self.scheduler,
            self.synth,
            self.reviewer,
            StubBatch(),
        )


def test_orchestrator_runs_corrective_loop(base_settings):
    orchestrator = HarnessOrchestrator(base_settings)
    request = RunRequest(goal="goal", mode=RunMode.AGGRESSIVE, apply_repo_changes=False)
    run_id, report = asyncio.run(orchestrator.run(request))
    assert run_id
    assert report.review_verdict.passed is True
    assert len(report.selected_results) == 2
    assert orchestrator.scheduler.calls == 2
