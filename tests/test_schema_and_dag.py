import pytest

from multiagent.domain.models import AgentRole, Plan, Subtask, TaskGraph
from multiagent.errors import SchedulerError
from multiagent.services.planner import PlannerService

from tests.conftest import make_plan, make_subtask


def test_strict_schema_rejects_extra():
    with pytest.raises(Exception):
        Plan.model_validate(
            {
                "task_summary": "x",
                "execution_strategy": "y",
                "assumptions": [],
                "subtasks": [],
                "final_acceptance_criteria": [],
                "recommended_global_strategy": {
                    "breadth_vs_depth": "balanced",
                    "expected_parallelism": 1,
                    "suggested_budget_allocation": {},
                },
                "unexpected": True,
            }
        )


def test_scheduler_cycle_detection(base_settings):
    from multiagent.services.scheduler import Scheduler
    from multiagent.adapters.filesystem import FileSystemAdapter
    from multiagent.adapters.logging import EventLogger
    from tests.conftest import DummyBudget, DummyEvaluator, DummySchedulerWorker, FakeGateway
    from multiagent.services.artifact_store import ArtifactStore
    from multiagent.services.candidate_generator import CandidateGenerator
    from multiagent.services.model_router import ModelRouter

    artifact_store = ArtifactStore(base_settings, "cycle", FileSystemAdapter())
    scheduler = Scheduler(
        base_settings,
        FakeGateway(),
        EventLogger(artifact_store.log_path(), "cycle"),
        artifact_store,
        DummyBudget(),
        ModelRouter(base_settings, FileSystemAdapter()),
        CandidateGenerator(base_settings, ModelRouter(base_settings, FileSystemAdapter())),
        DummySchedulerWorker(),
        DummyEvaluator(),
    )
    plan = make_plan(
        [
            make_subtask("a", depends_on=["b"]),
            make_subtask("b", depends_on=["a"]),
        ]
    )
    with pytest.raises(SchedulerError):
        scheduler._validate_dag(plan)


def test_agent_role_aliases_cover_planner_outputs():
    assert AgentRole("security") is AgentRole.SECURITY
    assert AgentRole("site_reliability_engineer") is AgentRole.SRE
    assert AgentRole("product_manager") is AgentRole.PM
    subtask = Subtask(
        id="role-test",
        title="Security Hardening",
        role="security",
        objective="Harden auth and abuse controls",
        depends_on=[],
        deliverables=["security checklist"],
        acceptance_criteria=["controls defined"],
        importance_score=8,
        complexity_score=6,
        parallelizable=True,
        recommended_candidate_count=1,
        recommended_model_tier="review",
        requires_review=True,
    )
    assert subtask.role is AgentRole.SECURITY
