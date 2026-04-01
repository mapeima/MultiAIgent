from __future__ import annotations

from itertools import combinations

from pydantic import BaseModel, ConfigDict, Field

from multiagent.adapters.gemini import GeminiGateway
from multiagent.config import Settings
from multiagent.domain.models import (
    CandidateExecution,
    EvaluationResult,
    EvaluationStrategy,
    ExecutionPhase,
    RunMode,
    Subtask,
)
from multiagent.errors import SchemaValidationError
from multiagent.services.artifact_store import ArtifactStore
from multiagent.services.prompts import PromptRegistry


class EvaluationBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluations: list[EvaluationResult]


class PairwiseDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    winner_candidate_id: str
    rationale: str


class EvaluatorService:
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

    async def evaluate(
        self,
        *,
        goal: str,
        subtask: Subtask,
        mode: RunMode,
        candidates: list[CandidateExecution],
    ) -> list[EvaluationResult]:
        if len(candidates) == 1:
            only = candidates[0]
            evaluation = EvaluationResult(
                candidate_id=only.spec.candidate_id,
                overall_score=round(only.result.confidence * 10, 2),
                rubric_scores=[],
                strengths=["Only candidate; confidence-based default score."],
                weaknesses=[],
                recommended=True,
                comparison_notes="Single candidate execution.",
                model_used=only.spec.model,
                strategy=EvaluationStrategy.RUBRIC,
            )
            self._artifact_store.write_evaluation(subtask.id, [evaluation.model_dump(mode="json")])
            return [evaluation]

        prompt = self._prompts.evaluator(goal=goal, subtask=subtask, candidates=candidates)
        bundle = await self._call_bundle(prompt.system_instruction, prompt.user_prompt)
        evaluations = {item.candidate_id: item for item in bundle.evaluations}
        if mode is not RunMode.EFFICIENT and len(candidates) > 1:
            pairwise_bonus = await self._pairwise_scores(goal=goal, subtask=subtask, candidates=candidates)
            for candidate_id, bonus in pairwise_bonus.items():
                if candidate_id in evaluations:
                    evaluations[candidate_id] = evaluations[candidate_id].model_copy(
                        update={"overall_score": min(10.0, evaluations[candidate_id].overall_score + bonus)}
                    )
        ranked = sorted(evaluations.values(), key=lambda item: item.overall_score, reverse=True)
        if ranked:
            top_id = ranked[0].candidate_id
            ranked = [
                item.model_copy(update={"recommended": item.candidate_id == top_id})
                for item in ranked
            ]
        self._artifact_store.write_evaluation(
            subtask.id,
            [item.model_dump(mode="json") for item in ranked],
        )
        return ranked

    async def _call_bundle(self, system_instruction: str, user_prompt: str) -> EvaluationBundle:
        last_error: Exception | None = None
        reminder = ""
        for _ in range(self._settings.max_retries + 1):
            try:
                response = await self._gateway.generate_structured(
                    model=self._settings.evaluator_models[0],
                    system_instruction=system_instruction,
                    user_prompt=user_prompt + reminder,
                    schema=EvaluationBundle,
                    phase=ExecutionPhase.EVALUATION,
                )
                return response.parsed
            except SchemaValidationError as exc:
                last_error = exc
                reminder = "\nReturn valid evaluations for every candidate."
        assert last_error is not None
        raise last_error

    async def _pairwise_scores(
        self,
        *,
        goal: str,
        subtask: Subtask,
        candidates: list[CandidateExecution],
    ) -> dict[str, float]:
        scores = {item.spec.candidate_id: 0.0 for item in candidates}
        pairs = list(combinations(candidates, 2))
        for left, right in pairs[:6]:
            prompt = self._prompts.pairwise(goal=goal, subtask=subtask, left=left, right=right)
            response = await self._gateway.generate_structured(
                model=self._settings.evaluator_models[0],
                system_instruction=prompt.system_instruction,
                user_prompt=prompt.user_prompt,
                schema=PairwiseDecision,
                phase=ExecutionPhase.EVALUATION,
                temperature=0.0,
            )
            winner = response.parsed.winner_candidate_id
            if winner in scores:
                scores[winner] += 0.3
        return scores
