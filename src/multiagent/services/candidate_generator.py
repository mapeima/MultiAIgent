from __future__ import annotations

from dataclasses import dataclass

from multiagent.config import Settings
from multiagent.domain.models import CandidateSpec, RunMode, Subtask
from multiagent.services.model_router import ModelRouter
from multiagent.services.utilization import ExecutionBudgetTuning


ExecutionTuning = ExecutionBudgetTuning


@dataclass(slots=True)
class CandidateGenerator:
    settings: Settings
    router: ModelRouter

    def generate(
        self,
        *,
        subtask: Subtask,
        mode: RunMode,
        tuning: ExecutionTuning,
        benchmark_models: list[str] | None = None,
        prompt_variants: list[str] | None = None,
        temperatures: list[float] | None = None,
        repo_context_hashes: dict[str, str] | None = None,
    ) -> list[CandidateSpec]:
        desired = self._candidate_count(subtask=subtask, mode=mode, tuning=tuning)
        models = self.router.models_for_subtask(
            subtask=subtask,
            mode=mode,
            desired_count=max(desired, tuning.benchmark_width),
            benchmark_models=benchmark_models,
        )
        variants = prompt_variants or self._default_variants(mode=mode, subtask=subtask)
        temps = temperatures or self._default_temperatures(mode=mode)
        repo_context_hashes = repo_context_hashes or {}

        candidates: list[CandidateSpec] = []
        for index in range(desired):
            model = models[index % len(models)]
            variant = variants[index % len(variants)]
            temperature = temps[index % len(temps)]
            benchmark_axes = []
            if len(models) > 1:
                benchmark_axes.append("model")
            if len(variants) > 1:
                benchmark_axes.append("prompt")
            if len(temps) > 1:
                benchmark_axes.append("temperature")
            candidates.append(
                CandidateSpec(
                    candidate_id=f"{subtask.id}-cand-{index + 1}",
                    subtask_id=subtask.id,
                    agent_role=subtask.role,
                    model=model,
                    temperature=temperature,
                    prompt_variant=variant,
                    reasoning_style="deliberate" if mode is not RunMode.EFFICIENT else "lean",
                    strictness_level="high" if variant == "strict" else "normal",
                    benchmark_axes=benchmark_axes,
                    mode=mode,
                    repo_context_hashes=repo_context_hashes,
                )
            )
        return candidates

    def _candidate_count(self, *, subtask: Subtask, mode: RunMode, tuning: ExecutionTuning) -> int:
        profile = self.settings.mode_profiles[mode]
        base = max(subtask.recommended_candidate_count, profile.base_candidates)
        if subtask.importance_score >= 8:
            base += 1
        if mode is RunMode.EXHAUST and subtask.importance_score >= 7:
            base += 1
        return max(1, min(base, tuning.candidate_limit, self.settings.max_candidates_per_subtask))

    def _default_variants(self, *, mode: RunMode, subtask: Subtask) -> list[str]:
        variants = ["default"]
        if mode is not RunMode.EFFICIENT or subtask.requires_review:
            variants.append("strict")
        if subtask.role.value in {
            "pm",
            "architect",
            "backend",
            "frontend",
            "infra",
            "security",
            "sre",
            "implementer",
            "optimizer",
        }:
            variants.append("architect")
        if mode is RunMode.EXHAUST:
            variants.append("critic")
        return list(dict.fromkeys(variants))

    def _default_temperatures(self, *, mode: RunMode) -> list[float]:
        if mode is RunMode.EFFICIENT:
            return [0.1]
        if mode is RunMode.AGGRESSIVE:
            return [0.1, 0.4]
        return [0.1, 0.4, 0.7]
