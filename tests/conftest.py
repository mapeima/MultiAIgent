from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from multiagent.adapters.gemini import GatewayCallResult
from multiagent.adapters.logging import EventLogger, MetricsTracker
from multiagent.adapters.pricing import PriceBook
from multiagent.config import Settings
from multiagent.domain.models import (
    AgentRole,
    CandidateExecution,
    CandidateSpec,
    CriterionScore,
    EvaluationResult,
    EvaluationStrategy,
    ExecutionPhase,
    ModelTier,
    Plan,
    RecommendedGlobalStrategy,
    ReviewVerdict,
    RunMode,
    Subtask,
    SubtaskSelection,
    SynthesisResult,
    TaskEdge,
    TaskGraph,
    TokenUsage,
    WorkerResult,
)
from multiagent.errors import SchemaValidationError
from multiagent.services.prompts import PromptPayload
from multiagent.utils import utc_now


@pytest.fixture()
def base_settings(tmp_path: Path) -> Settings:
    return Settings(
        gemini_api_key="test-key",
        artifact_dir=tmp_path / "runs",
        router_state_dir=tmp_path / "state",
        credit_expiry_datetime=None,
    )


class FakeGateway:
    def __init__(self, handlers: dict[str, Any] | None = None, fail_once: set[str] | None = None) -> None:
        self.handlers = handlers or {}
        self.fail_once = set(fail_once or set())
        self.calls: list[tuple[str, str]] = []

    async def generate_structured(self, *, schema, model: str, phase: ExecutionPhase, **kwargs):
        schema_name = getattr(schema, "__name__", str(schema))
        self.calls.append((schema_name, phase.value))
        if schema_name in self.fail_once:
            self.fail_once.remove(schema_name)
            raise SchemaValidationError(f"forced schema failure for {schema_name}")
        handler = self.handlers.get(schema_name)
        if handler is None:
            raise AssertionError(f"No fake handler for {schema_name}")
        parsed = handler(schema=schema, model=model, phase=phase, **kwargs)
        return GatewayCallResult(
            parsed=parsed,
            text="fake",
            model=model,
            usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
            latency_ms=5,
            estimated_cost_usd=0.02,
            actual_cost_usd=0.02,
            raw={"fake": True},
        )

    async def generate_text(self, *, model: str, **kwargs):
        return GatewayCallResult(
            parsed="hello",
            text="hello",
            model=model,
            usage=TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20),
            latency_ms=5,
            estimated_cost_usd=0.001,
            actual_cost_usd=0.001,
            raw={"fake": True},
        )

    async def count_tokens(self, *, model: str, contents: str) -> int:
        return max(1, len(contents) // 4)

    async def create_batch_from_file(self, *, model: str, request_file: Path, display_name: str):
        return type("BatchJob", (), {"name": f"jobs/{display_name}", "state": "QUEUED"})()

    async def download_batch_output(self, name: str):
        return type(
            "BatchDownload",
            (),
            {
                "state": "SUCCEEDED",
                "lines": [{"key": "variant-1", "response": {"text": "ok"}}],
                "output_file_name": "files/output",
            },
        )()


def make_subtask(subtask_id: str, *, depends_on: list[str] | None = None) -> Subtask:
    return Subtask(
        id=subtask_id,
        title=f"Task {subtask_id}",
        role=AgentRole.IMPLEMENTER,
        objective="Do useful work",
        depends_on=depends_on or [],
        deliverables=["output"],
        acceptance_criteria=["works"],
        importance_score=8,
        complexity_score=6,
        parallelizable=True,
        recommended_candidate_count=2,
        recommended_model_tier=ModelTier.BALANCED,
        requires_review=True,
    )


def make_plan(subtasks: list[Subtask]) -> Plan:
    return Plan(
        task_summary="Plan",
        execution_strategy="Parallel",
        assumptions=["none"],
        subtasks=subtasks,
        final_acceptance_criteria=["done"],
        recommended_global_strategy=RecommendedGlobalStrategy(
            breadth_vs_depth="balanced",
            expected_parallelism=2,
            suggested_budget_allocation={"workers": 0.5},
        ),
    )


def make_candidate_execution(candidate_id: str, score_hint: float = 0.5) -> CandidateExecution:
    spec = CandidateSpec(
        candidate_id=candidate_id,
        subtask_id="s1",
        agent_role=AgentRole.IMPLEMENTER,
        model="gemini-2.5-flash",
        temperature=0.1,
        prompt_variant="default",
        reasoning_style="lean",
        strictness_level="normal",
        benchmark_axes=[],
        mode=RunMode.AGGRESSIVE,
        repo_context_hashes={},
    )
    result = WorkerResult(
        summary=f"Summary {candidate_id}",
        detailed_result=f"Detailed {candidate_id}",
        artifacts=[],
        suggested_files=[],
        code_changes=[],
        risks=[],
        confidence=score_hint,
        model_used=spec.model,
        prompt_variant="default",
        token_usage_estimate=TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20),
        follow_up_suggestions=[],
        candidate_id=candidate_id,
    )
    return CandidateExecution(
        spec=spec,
        result=result,
        latency_ms=10,
        estimated_cost_usd=0.01,
        actual_cost_usd=0.01,
    )


class DummySchedulerWorker:
    def __init__(self, delay: float = 0.01) -> None:
        self.delay = delay
        self.active = 0
        self.max_active = 0

    def build_prompt(self, **kwargs) -> PromptPayload:
        return PromptPayload(system_instruction="sys", user_prompt="prompt")

    async def execute_candidate(self, *, candidate_spec: CandidateSpec, **kwargs) -> CandidateExecution:
        import asyncio

        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(self.delay)
        self.active -= 1
        result = WorkerResult(
            summary=f"summary {candidate_spec.candidate_id}",
            detailed_result="detail",
            artifacts=[],
            suggested_files=[],
            code_changes=[],
            risks=[],
            confidence=0.6,
            model_used=candidate_spec.model,
            prompt_variant=candidate_spec.prompt_variant,
            token_usage_estimate=TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20),
            follow_up_suggestions=[],
            candidate_id=candidate_spec.candidate_id,
        )
        return CandidateExecution(
            spec=candidate_spec,
            result=result,
            latency_ms=10,
            estimated_cost_usd=0.01,
            actual_cost_usd=0.01,
        )


class DummyEvaluator:
    async def evaluate(self, *, candidates: list[CandidateExecution], **kwargs) -> list[EvaluationResult]:
        ranked = []
        for index, candidate in enumerate(candidates):
            ranked.append(
                EvaluationResult(
                    candidate_id=candidate.spec.candidate_id,
                    overall_score=10 - index,
                    rubric_scores=[CriterionScore(name="correctness", score=9.0, rationale="ok")],
                    strengths=["good"],
                    weaknesses=[],
                    recommended=index == 0,
                    comparison_notes="",
                    model_used=candidate.spec.model,
                    strategy=EvaluationStrategy.RUBRIC,
                )
            )
        return ranked


@dataclass(slots=True)
class DummyBudget:
    spent: float = 0.0

    def estimate_call_cost(self, **kwargs) -> float:
        return 0.01

    def reserve(self, **kwargs):
        return type("Reservation", (), {"estimated_cost_usd": 0.01, "reservation_id": "r1"})()

    def commit(self, reservation, actual_cost_usd: float) -> None:
        self.spent += actual_cost_usd

    def release(self, reservation) -> None:
        return None

    def recent_daily_spend(self) -> float:
        return self.spent

    def snapshot(self):
        from multiagent.domain.models import BudgetSnapshot

        return BudgetSnapshot(
            run_spent_usd=self.spent,
            reserved_usd=0.0,
            daily_spent_usd=self.spent,
            session_spent_usd=self.spent,
            by_phase_usd={},
            hard_cap_usd=10.0,
            soft_cap_usd=8.0,
            target_utilization_ratio=0.8,
            forecast_to_completion_usd=0.0,
        )


class DummyReviewer:
    def __init__(self) -> None:
        self.calls = 0

    async def review(self, **kwargs) -> ReviewVerdict:
        self.calls += 1
        if self.calls == 1:
            return ReviewVerdict(
                passed=False,
                issues=[],
                fixable_within_budget=True,
                suggested_followup_subtasks=[make_subtask("fix-1")],
                rationale="needs one more pass",
            )
        return ReviewVerdict(
            passed=True,
            issues=[],
            fixable_within_budget=False,
            suggested_followup_subtasks=[],
            rationale="passed",
        )


class DummySynthesizer:
    def __init__(self) -> None:
        self.calls = 0

    async def synthesize(self, *, selections: list[SubtaskSelection], **kwargs) -> SynthesisResult:
        self.calls += 1
        return SynthesisResult(
            final_response_markdown=f"# Result {self.calls}",
            executive_summary="done",
            merged_best_practices=[],
            unresolved_issues=[],
            confidence_map={item.subtask_id: 0.8 for item in selections},
            referenced_candidate_ids=[item.selected_candidate_id for item in selections],
        )
