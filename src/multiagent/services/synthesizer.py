from __future__ import annotations

from multiagent.adapters.gemini import GeminiGateway
from multiagent.config import Settings
from multiagent.domain.models import ExecutionPhase, RunMode, SubtaskSelection, SynthesisResult
from multiagent.errors import SchemaValidationError
from multiagent.services.artifact_store import ArtifactStore
from multiagent.services.prompts import PromptRegistry


class SynthesizerService:
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

    async def synthesize(
        self,
        *,
        goal: str,
        plan_summary: str,
        selections: list[SubtaskSelection],
        mode: RunMode,
    ) -> SynthesisResult:
        prompt = self._prompts.synthesizer(
            goal=goal,
            plan_summary=plan_summary,
            selections=[item.model_dump(mode="json") for item in selections],
            mode=mode,
        )
        last_error: Exception | None = None
        reminder = ""
        for _ in range(self._settings.max_retries + 1):
            try:
                response = await self._gateway.generate_structured(
                    model=self._settings.synthesis_model,
                    system_instruction=prompt.system_instruction,
                    user_prompt=prompt.user_prompt + reminder,
                    schema=SynthesisResult,
                    phase=ExecutionPhase.SYNTHESIS,
                )
                result = response.parsed
                self._artifact_store.write_markdown("final_result.md", result.final_response_markdown)
                self._artifact_store.write_json("final_result.json", result.model_dump(mode="json"))
                return result
            except SchemaValidationError as exc:
                last_error = exc
                reminder = "\nReturn valid structured JSON matching the synthesis schema."
        assert last_error is not None
        raise last_error
