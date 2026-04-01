from __future__ import annotations

import asyncio
import heapq
from collections import defaultdict
from pathlib import Path

from multiagent.adapters.gemini import GeminiGateway
from multiagent.adapters.logging import EventLogger
from multiagent.config import Settings
from multiagent.domain.models import (
    CandidateExecution,
    ExecutionPhase,
    Plan,
    RunRequest,
    Subtask,
    SubtaskSelection,
)
from multiagent.errors import BudgetExceededError, SchedulerError
from multiagent.services.artifact_store import ArtifactStore
from multiagent.services.budget import BudgetManager
from multiagent.services.candidate_generator import CandidateGenerator, ExecutionTuning
from multiagent.services.evaluator import EvaluatorService
from multiagent.services.model_router import ModelRouter
from multiagent.services.repo_context import RepoContextSelector
from multiagent.services.repo_mutation import RepoMutationService
from multiagent.services.workers import WorkerService


class Scheduler:
    def __init__(
        self,
        settings: Settings,
        gateway: GeminiGateway,
        logger: EventLogger,
        artifact_store: ArtifactStore,
        budget: BudgetManager,
        router: ModelRouter,
        candidate_generator: CandidateGenerator,
        worker_service: WorkerService,
        evaluator_service: EvaluatorService,
        repo_context_selector: RepoContextSelector | None = None,
        repo_mutation_service: RepoMutationService | None = None,
    ) -> None:
        self._settings = settings
        self._gateway = gateway
        self._logger = logger
        self._artifact_store = artifact_store
        self._budget = budget
        self._router = router
        self._candidate_generator = candidate_generator
        self._worker_service = worker_service
        self._evaluator_service = evaluator_service
        self._repo_context_selector = repo_context_selector
        self._repo_mutation_service = repo_mutation_service

    async def execute(
        self,
        *,
        plan: Plan,
        request: RunRequest,
        tuning: ExecutionTuning,
        existing_selections: list[SubtaskSelection] | None = None,
    ) -> list[SubtaskSelection]:
        existing_selections = existing_selections or []
        self._validate_dag(plan)
        subtask_lookup = {subtask.id: subtask for subtask in plan.subtasks}
        remaining_dependencies = {
            subtask.id: len(subtask.depends_on)
            for subtask in plan.subtasks
        }
        dependents: dict[str, list[str]] = defaultdict(list)
        for subtask in plan.subtasks:
            for dependency in subtask.depends_on:
                dependents[dependency].append(subtask.id)
        ready: list[tuple[int, int, str]] = []
        for subtask in plan.subtasks:
            if remaining_dependencies[subtask.id] == 0:
                heapq.heappush(ready, (-subtask.importance_score, -subtask.complexity_score, subtask.id))
        global_sem = asyncio.Semaphore(tuning.max_concurrency)
        role_caps: dict[str, asyncio.Semaphore] = {}
        model_caps: dict[str, asyncio.Semaphore] = {}
        running: dict[asyncio.Task[SubtaskSelection], str] = {}
        results = {item.subtask_id: item for item in existing_selections}

        while ready or running:
            while ready and len(running) < tuning.max_concurrency:
                _, _, subtask_id = heapq.heappop(ready)
                subtask = subtask_lookup[subtask_id]
                task = asyncio.create_task(
                    self._execute_subtask(
                        subtask=subtask,
                        request=request,
                        tuning=tuning,
                        prior_results=results,
                        global_sem=global_sem,
                        role_caps=role_caps,
                        model_caps=model_caps,
                    )
                )
                running[task] = subtask_id
            if not running:
                break
            done, _pending = await asyncio.wait(running.keys(), return_when=asyncio.FIRST_COMPLETED)
            for finished in done:
                subtask_id = running.pop(finished)
                selection = finished.result()
                results[subtask_id] = selection
                for dependent_id in dependents.get(subtask_id, []):
                    remaining_dependencies[dependent_id] -= 1
                    if remaining_dependencies[dependent_id] == 0:
                        dependent = subtask_lookup[dependent_id]
                        heapq.heappush(
                            ready,
                            (-dependent.importance_score, -dependent.complexity_score, dependent_id),
                        )
        return [results[subtask.id] for subtask in plan.subtasks if subtask.id in results]

    async def _execute_subtask(
        self,
        *,
        subtask: Subtask,
        request: RunRequest,
        tuning: ExecutionTuning,
        prior_results: dict[str, SubtaskSelection],
        global_sem: asyncio.Semaphore,
        role_caps: dict[str, asyncio.Semaphore],
        model_caps: dict[str, asyncio.Semaphore],
    ) -> SubtaskSelection:
        prior_artifacts = [
            selection.selected_result.model_dump(mode="json")
            for selection in prior_results.values()
        ]
        repo_context = None
        if request.repo_path and self._repo_context_selector and self._settings.enable_repo_tools:
            repo_context = await self._repo_context_selector.select_context(
                goal=request.goal,
                subtask=subtask,
                repo_path=Path(request.repo_path),
            )
        repo_hashes = {
            item.path: item.sha256
            for item in (repo_context.selected_files if repo_context else [])
        }
        candidates = self._candidate_generator.generate(
            subtask=subtask,
            mode=request.mode,
            tuning=tuning,
            benchmark_models=request.benchmark_models,
            prompt_variants=request.prompt_variants or None,
            temperatures=request.temperatures or None,
            repo_context_hashes=repo_hashes,
        )
        executions = await asyncio.gather(
            *[
                self._run_candidate(
                    subtask=subtask,
                    request=request,
                    prior_artifacts=prior_artifacts,
                    repo_context=repo_context,
                    candidate=candidate,
                    global_sem=global_sem,
                    role_caps=role_caps,
                    model_caps=model_caps,
                )
                for candidate in candidates
            ],
            return_exceptions=True,
        )
        successful = [item for item in executions if isinstance(item, CandidateExecution)]
        failures = [item for item in executions if not isinstance(item, CandidateExecution)]
        if not successful:
            raise SchedulerError(f"All candidates failed for subtask {subtask.id}: {failures}")
        evaluations = await self._evaluator_service.evaluate(
            goal=request.goal,
            subtask=subtask,
            mode=request.mode,
            candidates=successful,
        )
        selected_candidate_id = evaluations[0].candidate_id if evaluations else successful[0].spec.candidate_id
        selected = next(item for item in successful if item.spec.candidate_id == selected_candidate_id)
        for evaluation in evaluations:
            self._router.record_outcome(
                model=next(item for item in successful if item.spec.candidate_id == evaluation.candidate_id).spec.model,
                score=evaluation.overall_score,
                cost_usd=next(
                    item for item in successful if item.spec.candidate_id == evaluation.candidate_id
                ).actual_cost_usd
                or next(item for item in successful if item.spec.candidate_id == evaluation.candidate_id).estimated_cost_usd,
                won=evaluation.candidate_id == selected_candidate_id,
            )
        if (
            request.apply_repo_changes
            and request.repo_path
            and self._repo_mutation_service is not None
            and selected.result.code_changes
        ):
            mutation = await self._repo_mutation_service.apply_selected_result(
                repo_path=Path(request.repo_path),
                subtask=subtask,
                repo_context=repo_context,
                result=selected.result,
            )
            if not mutation.applied:
                selected = selected.model_copy(
                    update={
                        "result": selected.result.model_copy(
                            update={
                                "risks": [
                                    *selected.result.risks,
                                    f"Repo mutation was reverted: {mutation.message}",
                                ]
                            }
                        )
                    }
                )
        selection = SubtaskSelection(
            subtask_id=subtask.id,
            selected_candidate_id=selected_candidate_id,
            selected_result=selected.result,
            candidate_results=successful,
            evaluations=evaluations,
            merged_candidate_ids=[],
        )
        self._artifact_store.write_json(
            f"summaries/{subtask.id}.json",
            selection.model_dump(mode="json"),
        )
        return selection

    async def _run_candidate(
        self,
        *,
        subtask: Subtask,
        request: RunRequest,
        prior_artifacts: list[dict[str, object]],
        repo_context,
        candidate,
        global_sem: asyncio.Semaphore,
        role_caps: dict[str, asyncio.Semaphore],
        model_caps: dict[str, asyncio.Semaphore],
    ) -> CandidateExecution:
        role_key = candidate.agent_role.value
        role_sem = role_caps.setdefault(
            role_key,
            asyncio.Semaphore(max(1, self._settings.max_concurrency // 2)),
        )
        model_sem = model_caps.setdefault(
            candidate.model,
            asyncio.Semaphore(max(1, self._settings.max_concurrency // 2)),
        )
        prompt = self._worker_service.build_prompt(
            goal=request.goal,
            subtask=subtask,
            mode=request.mode,
            prior_artifacts=prior_artifacts,
            repo_context=repo_context,
            candidate_spec=candidate,
        )
        prompt_text = f"{prompt.system_instruction}\n\n{prompt.user_prompt}"
        input_tokens = await self._gateway.count_tokens(model=candidate.model, contents=prompt_text)
        estimated_cost = self._budget.estimate_call_cost(
            phase=ExecutionPhase.WORKERS,
            model=candidate.model,
            input_tokens=input_tokens,
            output_tokens=self._settings.prompt_max_output_tokens // 2,
        )
        reservation = self._budget.reserve(
            phase=ExecutionPhase.WORKERS,
            estimated_cost_usd=estimated_cost,
            note=f"{subtask.id}:{candidate.candidate_id}",
        )
        try:
            async with global_sem, role_sem, model_sem:
                execution = await self._worker_service.execute_candidate(
                    goal=request.goal,
                    subtask=subtask,
                    mode=request.mode,
                    prior_artifacts=prior_artifacts,
                    repo_context=repo_context,
                    candidate_spec=candidate,
                )
                actual_cost = execution.actual_cost_usd or execution.estimated_cost_usd
                self._budget.commit(reservation, actual_cost)
                return execution
        except BudgetExceededError:
            self._budget.release(reservation)
            raise
        except Exception:
            self._budget.release(reservation)
            raise

    def _validate_dag(self, plan: Plan) -> None:
        graph = {subtask.id: subtask.depends_on[:] for subtask in plan.subtasks}
        visited: set[str] = set()
        active: set[str] = set()

        def visit(node: str) -> None:
            if node in active:
                raise SchedulerError(f"Cycle detected at subtask {node}")
            if node in visited:
                return
            active.add(node)
            for dependency in graph.get(node, []):
                visit(dependency)
            active.remove(node)
            visited.add(node)

        for subtask in plan.subtasks:
            visit(subtask.id)
