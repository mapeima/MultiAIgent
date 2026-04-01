from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from multiagent.adapters.filesystem import FileSystemAdapter
from multiagent.adapters.gemini import GeminiGateway
from multiagent.adapters.logging import EventLogger, MetricsTracker
from multiagent.adapters.pricing import PriceBook
from multiagent.adapters.subprocess_tools import SubprocessTools
from multiagent.config import Settings
from multiagent.domain.models import Plan, ReviewVerdict, RunMode, RunReport, RunRequest, SynthesisResult
from multiagent.services.artifact_store import ArtifactStore
from multiagent.services.batch import BatchService
from multiagent.services.budget import BudgetManager
from multiagent.services.candidate_generator import CandidateGenerator
from multiagent.services.evaluator import EvaluatorService
from multiagent.services.model_router import ModelRouter
from multiagent.services.planner import PlannerService
from multiagent.services.prompts import PromptRegistry
from multiagent.services.repo_context import RepoContextSelector
from multiagent.services.repo_mutation import RepoMutationService
from multiagent.services.reviewer import ReviewerService
from multiagent.services.scheduler import Scheduler
from multiagent.services.synthesizer import SynthesizerService
from multiagent.services.utilization import UtilizationEngine
from multiagent.services.workers import WorkerService
from multiagent.utils import utc_now


class Orchestrator:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._fs = FileSystemAdapter()
        self._subprocess = SubprocessTools()
        self._prompts = PromptRegistry()
        self._router = ModelRouter(settings, self._fs)
        self._utilization = UtilizationEngine()

    async def plan(self, request: RunRequest) -> tuple[str, Plan]:
        run_id = str(uuid.uuid4())
        artifact_store, _logger, _metrics, _gateway, _budget, planner, _scheduler, _synth, _reviewer, _batch = (
            self._build_runtime(run_id)
        )
        artifact_store.snapshot_settings(self._settings)
        artifact_store.write_json("request.json", request.model_dump(mode="json"))
        repo_summary = None
        if request.repo_path and self._settings.enable_repo_tools:
            selector = RepoContextSelector(self._settings, self._fs, self._prompts, _gateway)
            repo_summary = selector.summarize_repo(Path(request.repo_path))
        plan = await planner.create_plan(request, repo_summary=repo_summary)
        return run_id, plan

    async def run(self, request: RunRequest) -> tuple[str, RunReport]:
        run_id = str(uuid.uuid4())
        started_at = utc_now()
        (
            artifact_store,
            logger,
            metrics,
            gateway,
            budget,
            planner,
            scheduler,
            synthesizer,
            reviewer,
            _batch,
        ) = self._build_runtime(run_id)
        artifact_store.snapshot_settings(self._settings)
        artifact_store.write_json("request.json", request.model_dump(mode="json"))
        repo_summary = None
        if request.repo_path and self._settings.enable_repo_tools:
            selector = RepoContextSelector(self._settings, self._fs, self._prompts, gateway)
            repo_summary = selector.summarize_repo(Path(request.repo_path))
        recommendation = self._utilization.recommend(
            settings=self._settings,
            mode=request.mode,
            recent_daily_spend_usd=budget.recent_daily_spend(),
        )
        tuning = self._utilization.tune_execution(
            settings=self._settings,
            mode=request.mode,
            recommendation=recommendation,
        )
        plan = await planner.create_plan(request, repo_summary=repo_summary)
        selections = await scheduler.execute(plan=plan, request=request, tuning=tuning)
        synthesis = await synthesizer.synthesize(
            goal=request.goal,
            plan_summary=plan.task_summary,
            selections=selections,
            mode=request.mode,
        )
        verdict = await reviewer.review(
            goal=request.goal,
            acceptance_criteria=plan.final_acceptance_criteria,
            synthesis_markdown=synthesis.final_response_markdown,
            mode=request.mode,
        )
        max_loops = min(tuning.review_loops, self._settings.max_review_loops)
        loop_count = 0
        while (
            not verdict.passed
            and verdict.fixable_within_budget
            and verdict.suggested_followup_subtasks
            and loop_count < max_loops
        ):
            loop_count += 1
            corrective_plan = plan.model_copy(
                update={
                    "task_summary": f"Corrective loop {loop_count}",
                    "subtasks": [
                        subtask.model_copy(update={"depends_on": []})
                        for subtask in verdict.suggested_followup_subtasks
                    ],
                }
            )
            corrective = await scheduler.execute(
                plan=corrective_plan,
                request=request,
                tuning=tuning,
                existing_selections=selections,
            )
            selections.extend(corrective)
            synthesis = await synthesizer.synthesize(
                goal=request.goal,
                plan_summary=plan.task_summary,
                selections=selections,
                mode=request.mode,
            )
            verdict = await reviewer.review(
                goal=request.goal,
                acceptance_criteria=plan.final_acceptance_criteria,
                synthesis_markdown=synthesis.final_response_markdown,
                mode=request.mode,
            )

        completed_at = utc_now()
        budget_snapshot = budget.snapshot()
        report = RunReport(
            run_id=run_id,
            status="passed" if verdict.passed else "needs_attention",
            goal=request.goal,
            mode=request.mode,
            plan=plan,
            selected_results=selections,
            synthesis=synthesis,
            review_verdict=verdict,
            budget_snapshot=budget_snapshot,
            started_at=started_at,
            completed_at=completed_at,
            artifact_root=str(artifact_store.root),
            source_run_id=request.source_run_id,
            benchmark_summary={
                "benchmark_models": request.benchmark_models,
                "prompt_variants": request.prompt_variants,
                "temperatures": request.temperatures,
            }
            if request.benchmark_models or request.prompt_variants or request.temperatures
            else None,
        )
        artifact_store.write_json("run_report.json", report.model_dump(mode="json"))
        artifact_store.write_json("cost_report.json", budget_snapshot.model_dump(mode="json"))
        metrics.write(artifact_store.metrics_path())
        logger.log(level="INFO", phase="run", event_type="run_complete", status=report.status)
        return run_id, report

    async def benchmark(
        self,
        *,
        goal: str,
        models: list[str] | None = None,
        prompt_variants: list[str] | None = None,
        temperatures: list[float] | None = None,
        repo_path: str | None = None,
    ) -> tuple[str, RunReport]:
        request = RunRequest(
            goal=goal,
            mode=RunMode.AGGRESSIVE,
            repo_path=repo_path,
            apply_repo_changes=False,
            interactive=False,
            batch_enabled=False,
            benchmark_models=models or [],
            prompt_variants=prompt_variants or [],
            temperatures=temperatures or [],
            metadata={"benchmark": True},
        )
        return await self.run(request)

    async def replay(self, run_id: str) -> tuple[str, RunReport]:
        request_path = self._settings.artifact_dir / run_id / "request.json"
        request = RunRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
        replay_request = request.model_copy(update={"source_run_id": run_id})
        return await self.run(replay_request)

    async def inspect(self, run_id: str) -> dict[str, Any]:
        report_path = self._settings.artifact_dir / run_id / "run_report.json"
        report = RunReport.model_validate_json(report_path.read_text(encoding="utf-8"))
        return {
            "run_id": report.run_id,
            "status": report.status,
            "goal": report.goal,
            "mode": report.mode.value,
            "artifact_root": report.artifact_root,
            "spent_usd": report.budget_snapshot.run_spent_usd,
            "subtasks": len(report.plan.subtasks),
            "selected_candidates": [item.selected_candidate_id for item in report.selected_results],
            "review_passed": report.review_verdict.passed,
        }

    async def submit_batch(
        self,
        *,
        goal: str,
        count: int,
        model: str,
        role,
        prompt_variant: str,
    ):
        run_id = str(uuid.uuid4())
        _artifact_store, _logger, _metrics, _gateway, _budget, _planner, _scheduler, _synth, _reviewer, batch = (
            self._build_runtime(run_id)
        )
        return await batch.submit(
            goal=goal,
            count=count,
            model=model,
            role=role,
            prompt_variant=prompt_variant,
        )

    async def reconcile_batch(self, batch_id: str) -> dict[str, object]:
        run_id = str(uuid.uuid4())
        _artifact_store, _logger, _metrics, _gateway, _budget, _planner, _scheduler, _synth, _reviewer, batch = (
            self._build_runtime(run_id)
        )
        return await batch.reconcile(batch_id)

    def budget_recommendation(self, mode: RunMode) -> dict[str, object]:
        temp_artifact = ArtifactStore(self._settings, "temp-budget", self._fs)
        temp_logger = EventLogger(temp_artifact.log_path(), "temp-budget")
        budget = BudgetManager(self._settings, PriceBook(self._settings), self._fs, temp_logger)
        recommendation = self._utilization.recommend(
            settings=self._settings,
            mode=mode,
            recent_daily_spend_usd=budget.recent_daily_spend(),
        )
        tuning = self._utilization.tune_execution(
            settings=self._settings,
            mode=mode,
            recommendation=recommendation,
        )
        return {
            "recommendation": recommendation.model_dump(mode="json"),
            "tuning": {
                "max_concurrency": tuning.max_concurrency,
                "candidate_limit": tuning.candidate_limit,
                "benchmark_width": tuning.benchmark_width,
                "review_loops": tuning.review_loops,
            },
        }

    def _build_runtime(self, run_id: str):
        artifact_store = ArtifactStore(self._settings, run_id, self._fs)
        logger = EventLogger(artifact_store.log_path(), run_id)
        metrics = MetricsTracker()
        price_book = PriceBook(self._settings)
        gateway = GeminiGateway(self._settings, price_book, logger, metrics)
        budget = BudgetManager(self._settings, price_book, self._fs, logger)
        planner = PlannerService(self._settings, self._prompts, gateway, artifact_store)
        selector = RepoContextSelector(self._settings, self._fs, self._prompts, gateway)
        worker = WorkerService(self._settings, self._prompts, gateway, artifact_store)
        evaluator = EvaluatorService(self._settings, self._prompts, gateway, artifact_store)
        synthesizer = SynthesizerService(self._settings, self._prompts, gateway, artifact_store)
        reviewer = ReviewerService(self._settings, self._prompts, gateway)
        mutation = RepoMutationService(self._fs, self._subprocess, artifact_store)
        scheduler = Scheduler(
            self._settings,
            gateway,
            logger,
            artifact_store,
            budget,
            self._router,
            CandidateGenerator(self._settings, self._router),
            worker,
            evaluator,
            selector if self._settings.enable_repo_tools else None,
            mutation,
        )
        batch = BatchService(self._settings, gateway, self._prompts)
        return (
            artifact_store,
            logger,
            metrics,
            gateway,
            budget,
            planner,
            scheduler,
            synthesizer,
            reviewer,
            batch,
        )
