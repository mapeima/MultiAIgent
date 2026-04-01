from __future__ import annotations

from multiagent.adapters.gemini import GeminiGateway
from multiagent.config import Settings
from multiagent.domain.models import (
    CandidateExecution,
    CandidateSpec,
    ExecutionPhase,
    RepoContext,
    RunMode,
    Subtask,
    WorkerResult,
)
from multiagent.errors import SchemaValidationError
from multiagent.services.artifact_store import ArtifactStore
from multiagent.services.prompts import PromptPayload, PromptRegistry


class WorkerService:
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

    def build_prompt(
        self,
        *,
        goal: str,
        subtask: Subtask,
        mode: RunMode,
        prior_artifacts: list[dict[str, object]],
        repo_context: RepoContext | None,
        candidate_spec: CandidateSpec,
    ) -> PromptPayload:
        return self._prompts.worker(
            role=subtask.role,
            goal=goal,
            subtask=subtask,
            mode=mode,
            prior_artifacts=prior_artifacts,
            repo_context=repo_context,
            candidate_metadata=candidate_spec.model_dump(mode="json"),
            variant=candidate_spec.prompt_variant,
        )

    async def execute_candidate(
        self,
        *,
        goal: str,
        subtask: Subtask,
        mode: RunMode,
        prior_artifacts: list[dict[str, object]],
        repo_context: RepoContext | None,
        candidate_spec: CandidateSpec,
    ) -> CandidateExecution:
        prompt = self.build_prompt(
            goal=goal,
            subtask=subtask,
            mode=mode,
            prior_artifacts=prior_artifacts,
            repo_context=repo_context,
            candidate_spec=candidate_spec,
        )
        last_error: Exception | None = None
        reminder = ""
        for _ in range(self._settings.max_retries + 1):
            try:
                response = await self._gateway.generate_structured(
                    model=candidate_spec.model,
                    system_instruction=prompt.system_instruction,
                    user_prompt=prompt.user_prompt + reminder,
                    schema=WorkerResult,
                    phase=ExecutionPhase.WORKERS,
                    temperature=candidate_spec.temperature,
                )
                worker_result = response.parsed.model_copy(
                    update={
                        "candidate_id": candidate_spec.candidate_id,
                        "model_used": candidate_spec.model,
                        "prompt_variant": candidate_spec.prompt_variant,
                        "token_usage_estimate": response.usage,
                    }
                )
                execution = CandidateExecution(
                    spec=candidate_spec,
                    result=worker_result,
                    latency_ms=response.latency_ms,
                    estimated_cost_usd=response.estimated_cost_usd,
                    actual_cost_usd=response.actual_cost_usd,
                )
                self._artifact_store.write_agent_output(
                    subtask.id,
                    candidate_spec.candidate_id,
                    execution.model_dump(mode="json"),
                )
                return execution
            except SchemaValidationError as exc:
                last_error = exc
                reminder = "\nReturn valid structured JSON matching the schema exactly."
        assert last_error is not None
        raise last_error
