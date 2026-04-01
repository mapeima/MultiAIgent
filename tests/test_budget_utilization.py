from datetime import UTC, datetime, timedelta

import pytest

from multiagent.adapters.filesystem import FileSystemAdapter
from multiagent.adapters.logging import EventLogger
from multiagent.adapters.pricing import PriceBook
from multiagent.domain.models import ExecutionPhase, RunMode
from multiagent.errors import BudgetExceededError
from multiagent.services.artifact_store import ArtifactStore
from multiagent.services.budget import BudgetManager
from multiagent.services.utilization import UtilizationEngine


def test_budget_enforces_hard_cap(base_settings):
    settings = base_settings.model_copy(
        update={"total_budget_cap_usd": 0.01, "session_budget_cap_usd": 0.01, "daily_budget_cap_usd": 1.0}
    )
    store = ArtifactStore(settings, "budget-test", FileSystemAdapter())
    logger = EventLogger(store.log_path(), "budget-test")
    budget = BudgetManager(settings, PriceBook(settings), FileSystemAdapter(), logger)
    with pytest.raises(BudgetExceededError):
        budget.reserve(phase=ExecutionPhase.WORKERS, estimated_cost_usd=0.02, note="too much")


def test_utilization_increases_pressure_near_expiry(base_settings):
    settings = base_settings.model_copy(
        update={
            "credit_expiry_datetime": datetime.now(tz=UTC) + timedelta(hours=12),
            "estimated_remaining_credit_usd": 120.0,
        }
    )
    engine = UtilizationEngine()
    recommendation = engine.recommend(
        settings=settings,
        mode=RunMode.EXHAUST,
        recent_daily_spend_usd=1.0,
    )
    assert recommendation.urgency_score > 0
    assert recommendation.concurrency_multiplier > 1
