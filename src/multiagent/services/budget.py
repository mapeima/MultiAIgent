from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime

from multiagent.adapters.filesystem import FileSystemAdapter
from multiagent.adapters.logging import EventLogger
from multiagent.adapters.pricing import PriceBook
from multiagent.config import Settings
from multiagent.domain.models import BudgetSnapshot, ExecutionPhase, TokenUsage
from multiagent.errors import BudgetExceededError


@dataclass(slots=True)
class BudgetReservation:
    reservation_id: str
    phase: ExecutionPhase
    estimated_cost_usd: float
    note: str


class BudgetManager:
    def __init__(
        self,
        settings: Settings,
        price_book: PriceBook,
        fs: FileSystemAdapter,
        logger: EventLogger,
    ) -> None:
        self._settings = settings
        self._price_book = price_book
        self._fs = fs
        self._logger = logger
        self._phase_allocations = {
            ExecutionPhase.PLANNING: 0.10,
            ExecutionPhase.WORKERS: 0.45,
            ExecutionPhase.CANDIDATE_GENERATION: 0.10,
            ExecutionPhase.EVALUATION: 0.15,
            ExecutionPhase.SYNTHESIS: 0.10,
            ExecutionPhase.REVIEW: 0.10,
            ExecutionPhase.BATCH: 0.10,
        }
        self._run_spent_usd = 0.0
        self._reserved_usd = 0.0
        self._session_spent_usd = 0.0
        self._phase_spend: dict[str, float] = {}
        self._active_reservations: dict[str, BudgetReservation] = {}
        self._state_path = settings.router_state_dir / "budget_state.json"
        self._state = self._load_state()

    def reserve(self, *, phase: ExecutionPhase, estimated_cost_usd: float, note: str) -> BudgetReservation:
        self._check_can_spend(phase=phase, estimated_cost_usd=estimated_cost_usd)
        reservation = BudgetReservation(
            reservation_id=str(uuid.uuid4()),
            phase=phase,
            estimated_cost_usd=estimated_cost_usd,
            note=note,
        )
        self._active_reservations[reservation.reservation_id] = reservation
        self._reserved_usd += estimated_cost_usd
        self._logger.log(
            level="INFO",
            phase=phase.value,
            event_type="budget_reservation",
            status="reserved",
            estimated_cost_usd=estimated_cost_usd,
            note=note,
        )
        return reservation

    def commit(self, reservation: BudgetReservation, actual_cost_usd: float) -> None:
        self._active_reservations.pop(reservation.reservation_id, None)
        self._reserved_usd = max(0.0, self._reserved_usd - reservation.estimated_cost_usd)
        self._run_spent_usd += actual_cost_usd
        self._session_spent_usd += actual_cost_usd
        self._phase_spend[reservation.phase.value] = (
            self._phase_spend.get(reservation.phase.value, 0.0) + actual_cost_usd
        )
        self._state.setdefault("days", {})
        today = date.today().isoformat()
        self._state["days"][today] = float(self._state["days"].get(today, 0.0)) + actual_cost_usd
        self._persist_state()

    def release(self, reservation: BudgetReservation) -> None:
        if reservation.reservation_id in self._active_reservations:
            self._active_reservations.pop(reservation.reservation_id, None)
            self._reserved_usd = max(0.0, self._reserved_usd - reservation.estimated_cost_usd)

    def estimate_call_cost(
        self,
        *,
        phase: ExecutionPhase,
        model: str,
        input_tokens: int,
        output_tokens: int,
        batch: bool = False,
    ) -> float:
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        estimate = self._price_book.estimate_cost(model=model, usage=usage, batch=batch)
        self._logger.log(
            level="INFO",
            phase=phase.value,
            event_type="cost_estimate",
            status="computed",
            model=model,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            estimated_cost_usd=estimate.total_cost_usd,
        )
        return estimate.total_cost_usd

    def recent_daily_spend(self) -> float:
        today = date.today().isoformat()
        return float(self._state.get("days", {}).get(today, 0.0))

    def snapshot(
        self,
        *,
        forecast_to_completion_usd: float = 0.0,
        blocked_reason: str | None = None,
    ) -> BudgetSnapshot:
        today = date.today().isoformat()
        return BudgetSnapshot(
            run_spent_usd=self._run_spent_usd,
            reserved_usd=self._reserved_usd,
            daily_spent_usd=float(self._state.get("days", {}).get(today, 0.0)),
            session_spent_usd=self._session_spent_usd,
            by_phase_usd=self._phase_spend.copy(),
            hard_cap_usd=self._settings.total_budget_cap_usd,
            soft_cap_usd=self._settings.soft_budget_cap_usd,
            target_utilization_ratio=self._settings.target_utilization_ratio,
            forecast_to_completion_usd=forecast_to_completion_usd,
            blocked_reason=blocked_reason,
        )

    def _check_can_spend(self, *, phase: ExecutionPhase, estimated_cost_usd: float) -> None:
        projected_total = self._run_spent_usd + self._reserved_usd + estimated_cost_usd
        projected_session = self._session_spent_usd + self._reserved_usd + estimated_cost_usd
        today = date.today().isoformat()
        projected_daily = float(self._state.get("days", {}).get(today, 0.0)) + estimated_cost_usd
        phase_cap = self._settings.total_budget_cap_usd * self._phase_allocations.get(phase, 1.0)
        projected_phase = self._phase_spend.get(phase.value, 0.0) + estimated_cost_usd
        if projected_total > self._settings.total_budget_cap_usd:
            raise BudgetExceededError("Run hard budget cap exceeded")
        if projected_session > self._settings.session_budget_cap_usd:
            raise BudgetExceededError("Session budget cap exceeded")
        if projected_daily > self._settings.daily_budget_cap_usd:
            raise BudgetExceededError("Daily budget cap exceeded")
        if projected_phase > phase_cap * 1.5:
            raise BudgetExceededError(f"Phase budget for {phase.value} exceeded")

    def _load_state(self) -> dict[str, object]:
        if self._state_path.exists():
            return self._fs.read_json(self._state_path)
        return {"days": {}, "updated_at": datetime.now(tz=UTC).isoformat()}

    def _persist_state(self) -> None:
        self._state["updated_at"] = datetime.now(tz=UTC).isoformat()
        self._fs.write_json(self._state_path, self._state)
