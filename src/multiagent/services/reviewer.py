from __future__ import annotations

from multiagent.adapters.gemini import GeminiGateway
from multiagent.config import Settings
from multiagent.domain.models import ExecutionPhase, ReviewVerdict, RunMode
from multiagent.errors import SchemaValidationError
from multiagent.services.prompts import PromptRegistry


class ReviewerService:
    def __init__(
        self,
        settings: Settings,
        prompts: PromptRegistry,
        gateway: GeminiGateway,
    ) -> None:
        self._settings = settings
        self._prompts = prompts
        self._gateway = gateway

    async def review(
        self,
        *,
        goal: str,
        acceptance_criteria: list[str],
        synthesis_markdown: str,
        mode: RunMode,
    ) -> ReviewVerdict:
        prompt = self._prompts.reviewer(
            goal=goal,
            acceptance_criteria=acceptance_criteria,
            synthesis_markdown=synthesis_markdown,
            mode=mode,
        )
        last_error: Exception | None = None
        reminder = ""
        for _ in range(self._settings.max_retries + 1):
            try:
                response = await self._gateway.generate_structured(
                    model=self._settings.reviewer_model,
                    system_instruction=prompt.system_instruction,
                    user_prompt=prompt.user_prompt + reminder,
                    schema=ReviewVerdict,
                    phase=ExecutionPhase.REVIEW,
                    temperature=0.0,
                )
                return response.parsed
            except SchemaValidationError as exc:
                last_error = exc
                reminder = "\nReturn valid structured JSON with a clear verdict and issues list."
        assert last_error is not None
        raise last_error
