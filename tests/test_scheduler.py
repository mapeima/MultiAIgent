import asyncio

from multiagent.adapters.filesystem import FileSystemAdapter
from multiagent.adapters.logging import EventLogger
from multiagent.domain.models import RunMode, RunRequest
from multiagent.services.artifact_store import ArtifactStore
from multiagent.services.candidate_generator import CandidateGenerator
from multiagent.services.model_router import ModelRouter
from multiagent.services.scheduler import Scheduler
from multiagent.services.utilization import ExecutionBudgetTuning

from tests.conftest import DummyBudget, DummyEvaluator, DummySchedulerWorker, FakeGateway, make_plan, make_subtask


def test_scheduler_runs_candidates_in_parallel(base_settings):
    worker = DummySchedulerWorker(delay=0.05)
    gateway = FakeGateway()
    artifact_store = ArtifactStore(base_settings, "scheduler", FileSystemAdapter())
    logger = EventLogger(artifact_store.log_path(), "scheduler")
    router = ModelRouter(base_settings, FileSystemAdapter())
    scheduler = Scheduler(
        base_settings,
        gateway,
        logger,
        artifact_store,
        DummyBudget(),
        router,
        CandidateGenerator(base_settings, router),
        worker,
        DummyEvaluator(),
    )
    tuning = ExecutionBudgetTuning(max_concurrency=4, candidate_limit=4, benchmark_width=2, review_loops=1)
    request = RunRequest(goal="goal", mode=RunMode.AGGRESSIVE, apply_repo_changes=False)
    plan = make_plan([make_subtask("a"), make_subtask("b")])
    results = asyncio.run(scheduler.execute(plan=plan, request=request, tuning=tuning))
    assert len(results) == 2
    assert worker.max_active >= 2
