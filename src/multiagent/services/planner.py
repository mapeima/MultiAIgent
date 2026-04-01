from __future__ import annotations

from collections import Counter

from multiagent.adapters.gemini import GeminiGateway
from multiagent.config import Settings
from multiagent.domain.models import ExecutionPhase, Plan, RunRequest, TaskEdge, TaskGraph
from multiagent.errors import SchedulerError, SchemaValidationError
from multiagent.services.artifact_store import ArtifactStore
from multiagent.services.prompts import PromptRegistry


class PlannerService:
    def __init__(
        self,
        settings: Settings,
        prompts: PromptRegistry,
        gateway: GeminiGateway,
        artifact_store: ArtifactStore,
    ) -> None:
        self._settings = settings
        self._prompts = prompts
        self._gateway = gateway
        self._artifact_store = artifact_store

    async def create_plan(self, request: RunRequest, repo_summary: str | None = None) -> Plan:
        prompt = self._prompts.planner(
            goal=request.goal,
            constraints=request.constraints,
            mode=request.mode,
            repo_summary=repo_summary,
        )
        last_error: Exception | None = None
        reminder = ""
        for _ in range(self._settings.max_retries + 1):
            try:
                response = await self._gateway.generate_structured(
                    model=self._settings.planner_model,
                    system_instruction=prompt.system_instruction,
                    user_prompt=prompt.user_prompt + reminder,
                    schema=Plan,
                    phase=ExecutionPhase.PLANNING,
                    max_output_tokens=self._settings.planner_max_output_tokens,
                )
                plan = self._normalize_plan(response.parsed, request)
                self._artifact_store.write_json("plan.json", plan.model_dump(mode="json"))
                self._artifact_store.write_json("task_graph.json", self.task_graph(plan).model_dump(mode="json"))
                return plan
            except (SchemaValidationError, SchedulerError) as exc:
                last_error = exc
                reminder = "\nReturn valid structured JSON with unique task ids and valid dependencies."
        assert last_error is not None
        raise last_error

    def task_graph(self, plan: Plan) -> TaskGraph:
        edges = [
            TaskEdge(source=dependency, target=subtask.id)
            for subtask in plan.subtasks
            for dependency in subtask.depends_on
        ]
        return TaskGraph(nodes=[subtask.id for subtask in plan.subtasks], edges=edges)

    def _normalize_plan(self, plan: Plan, request: RunRequest) -> Plan:
        ids = [subtask.id for subtask in plan.subtasks]
        duplicates = [item for item, count in Counter(ids).items() if count > 1]
        if duplicates:
            raise SchedulerError(f"Duplicate subtask ids: {duplicates}")
        valid_ids = set(ids)
        for subtask in plan.subtasks:
            unknown_dependencies = [item for item in subtask.depends_on if item not in valid_ids]
            if unknown_dependencies:
                raise SchedulerError(f"Unknown dependencies for {subtask.id}: {unknown_dependencies}")
        normalized_subtasks = []
        for subtask in plan.subtasks:
            recommended_candidates = min(
                max(1, subtask.recommended_candidate_count),
                self._settings.max_candidates_per_subtask,
            )
            normalized_subtasks.append(
                subtask.model_copy(update={"recommended_candidate_count": recommended_candidates})
            )
        return plan.model_copy(update={"subtasks": normalized_subtasks})
