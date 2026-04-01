from __future__ import annotations

from dataclasses import dataclass

from multiagent.config import PricingEntry, Settings
from multiagent.domain.models import TokenUsage


@dataclass(slots=True)
class CostEstimate:
    model: str
    input_cost_usd: float
    output_cost_usd: float
    cached_input_cost_usd: float
    total_cost_usd: float


class PriceBook:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pricing = settings.pricing_table

    def resolve(self, model: str) -> PricingEntry:
        if model in self._pricing:
            return self._pricing[model]
        for key, pricing in self._pricing.items():
            if model.startswith(key):
                return pricing
        return PricingEntry(
            input_per_million_usd=0.0,
            output_per_million_usd=0.0,
            cached_input_per_million_usd=0.0,
            batch_discount_ratio=0.5,
        )

    def estimate_cost(
        self,
        *,
        model: str,
        usage: TokenUsage,
        batch: bool = False,
    ) -> CostEstimate:
        pricing = self.resolve(model)
        multiplier = pricing.batch_discount_ratio if batch else 1.0
        input_cost = usage.input_tokens * pricing.input_per_million_usd / 1_000_000
        output_cost = usage.output_tokens * pricing.output_per_million_usd / 1_000_000
        cached_cost = usage.cached_input_tokens * pricing.cached_input_per_million_usd / 1_000_000
        total = (input_cost + output_cost + cached_cost) * multiplier
        return CostEstimate(
            model=model,
            input_cost_usd=input_cost * multiplier,
            output_cost_usd=output_cost * multiplier,
            cached_input_cost_usd=cached_cost * multiplier,
            total_cost_usd=total,
        )
