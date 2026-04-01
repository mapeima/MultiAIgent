from __future__ import annotations

from multiagent.adapters.filesystem import FileSystemAdapter
from multiagent.config import Settings
from multiagent.domain.models import ModelTier, RunMode, Subtask
from multiagent.utils import ensure_directory


class ModelRouter:
    def __init__(self, settings: Settings, fs: FileSystemAdapter) -> None:
        self._settings = settings
        self._fs = fs
        self._history_path = ensure_directory(settings.router_state_dir) / "router_history.json"
        self._history = self._load_history()

    def models_for_subtask(
        self,
        *,
        subtask: Subtask,
        mode: RunMode,
        desired_count: int,
        benchmark_models: list[str] | None = None,
    ) -> list[str]:
        if subtask.model_override:
            return [subtask.model_override]
        if benchmark_models:
            return benchmark_models[:desired_count]
        pool = self._pool_for_tier(subtask.recommended_model_tier)
        if subtask.complexity_score >= 8 and self._settings.premium_models:
            pool = [*self._settings.premium_models, *pool]
        if subtask.importance_score <= 4 and self._settings.cheap_worker_models:
            pool = [*self._settings.cheap_worker_models, *pool]
        if mode is RunMode.EXHAUST:
            pool = [*pool, *self._settings.synthesis_models, *self._settings.review_models]
        ordered = self._sort_by_history(pool)
        return list(dict.fromkeys(ordered))[:desired_count] if desired_count > 0 else ordered

    def fallback_chain(self, primary_model: str) -> list[str]:
        pool = [
            primary_model,
            *self._settings.balanced_worker_models,
            *self._settings.premium_models,
            *self._settings.cheap_worker_models,
        ]
        return list(dict.fromkeys(pool))

    def record_outcome(self, *, model: str, score: float, cost_usd: float, won: bool) -> None:
        entry = self._history.setdefault(
            model,
            {"calls": 0, "wins": 0, "score_total": 0.0, "cost_total": 0.0},
        )
        entry["calls"] += 1
        entry["score_total"] += score
        entry["cost_total"] += cost_usd
        if won:
            entry["wins"] += 1
        self._persist()

    def _pool_for_tier(self, tier: ModelTier) -> list[str]:
        return {
            ModelTier.CHEAP: self._settings.cheap_worker_models,
            ModelTier.BALANCED: self._settings.balanced_worker_models,
            ModelTier.PREMIUM: self._settings.premium_models,
            ModelTier.SYNTHESIS: self._settings.synthesis_models,
            ModelTier.REVIEW: self._settings.review_models,
            ModelTier.EVALUATOR: self._settings.evaluator_models,
        }[tier]

    def _load_history(self) -> dict[str, dict[str, float]]:
        if self._history_path.exists():
            return self._fs.read_json(self._history_path)
        return {}

    def _persist(self) -> None:
        self._fs.write_json(self._history_path, self._history)

    def _sort_by_history(self, models: list[str]) -> list[str]:
        def score(model: str) -> tuple[float, float]:
            entry = self._history.get(model, {})
            calls = max(float(entry.get("calls", 0.0)), 1.0)
            wins = float(entry.get("wins", 0.0))
            score_total = float(entry.get("score_total", 0.0))
            cost_total = float(entry.get("cost_total", 0.0))
            average_score = score_total / calls
            win_rate = wins / calls
            efficiency = average_score / max(cost_total, 0.01)
            return (win_rate + efficiency, average_score)

        return sorted(models, key=score, reverse=True)
