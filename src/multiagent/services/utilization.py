from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from multiagent.config import Settings
from multiagent.domain.models import RunMode, UtilizationRecommendation


@dataclass(slots=True)
class ExecutionBudgetTuning:
    max_concurrency: int
    candidate_limit: int
    benchmark_width: int
    review_loops: int


class UtilizationEngine:
    def recommend(
        self,
        *,
        settings: Settings,
        mode: RunMode,
        recent_daily_spend_usd: float,
    ) -> UtilizationRecommendation:
        if settings.credit_expiry_datetime is None or settings.estimated_remaining_credit_usd <= 0:
            base_loops = settings.mode_profiles[mode].review_loops
            return UtilizationRecommendation(
                recommended_spend_velocity_usd_per_day=recent_daily_spend_usd,
                urgency_score=0.0,
                concurrency_multiplier=1.0,
                candidate_multiplier=1.0,
                benchmark_multiplier=1.0,
                review_loop_target=base_loops,
                enable_batch_mode=settings.enable_batch_mode and mode is RunMode.EXHAUST,
                rationale="No expiry window configured; keep default utilization settings.",
            )

        now = datetime.now(tz=UTC)
        expiry = settings.credit_expiry_datetime
        hours_remaining = max((expiry - now).total_seconds() / 3600, 1.0)
        days_remaining = max(hours_remaining / 24, 0.05)
        target_velocity = settings.estimated_remaining_credit_usd / days_remaining
        urgency = min(1.0, max(0.0, 1.0 - (days_remaining / 14)))
        pace_gap_ratio = 0.0
        if target_velocity > 0:
            pace_gap_ratio = max(0.0, (target_velocity - recent_daily_spend_usd) / target_velocity)
        pressure = max(urgency, pace_gap_ratio)
        profile = settings.mode_profiles[mode]
        return UtilizationRecommendation(
            recommended_spend_velocity_usd_per_day=target_velocity,
            urgency_score=pressure,
            concurrency_multiplier=1.0 + pressure,
            candidate_multiplier=1.0 + pressure,
            benchmark_multiplier=1.0 + pressure if settings.enable_benchmarking else 1.0,
            review_loop_target=max(profile.review_loops, round(profile.review_loops * (1.0 + pressure))),
            enable_batch_mode=settings.enable_batch_mode and pressure > 0.35,
            rationale=(
                f"Expiry-driven utilization pressure={pressure:.2f}; "
                f"target daily spend=${target_velocity:.2f}."
            ),
        )

    def tune_execution(
        self,
        *,
        settings: Settings,
        mode: RunMode,
        recommendation: UtilizationRecommendation,
    ) -> ExecutionBudgetTuning:
        profile = settings.mode_profiles[mode]
        max_concurrency = max(
            1,
            min(
                round(settings.max_concurrency * recommendation.concurrency_multiplier),
                settings.max_concurrency * 2,
            ),
        )
        candidate_limit = max(
            profile.base_candidates,
            min(
                round(settings.max_candidates_per_subtask * recommendation.candidate_multiplier),
                settings.max_candidates_per_subtask,
            ),
        )
        benchmark_width = max(
            1,
            min(
                round(profile.benchmark_width * recommendation.benchmark_multiplier),
                settings.max_candidates_per_subtask,
            ),
        )
        return ExecutionBudgetTuning(
            max_concurrency=max_concurrency,
            candidate_limit=candidate_limit,
            benchmark_width=benchmark_width,
            review_loops=min(recommendation.review_loop_target, settings.max_review_loops),
        )
