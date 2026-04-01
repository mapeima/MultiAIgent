from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from multiagent.domain.models import RunMode
from multiagent.utils import parse_csv


class BaseConfigModel(BaseModel):
    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
    }


class PricingEntry(BaseConfigModel):
    input_per_million_usd: float
    output_per_million_usd: float
    cached_input_per_million_usd: float = 0.0
    batch_discount_ratio: float = 0.5


class ModeProfile(BaseConfigModel):
    base_candidates: int
    benchmark_width: int
    review_loops: int
    enable_pairwise: bool
    enable_tournament: bool
    enable_batch: bool


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
        validate_by_name=True,
        enable_decoding=False,
    )

    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    default_mode: RunMode = Field(default=RunMode.EFFICIENT, validation_alias="DEFAULT_MODE")
    planner_model: str = Field(default="gemini-2.5-flash", validation_alias="PLANNER_MODEL")
    synthesis_model: str = Field(default="gemini-2.5-pro", validation_alias="SYNTHESIS_MODEL")
    reviewer_model: str = Field(default="gemini-2.5-flash", validation_alias="REVIEWER_MODEL")
    cheap_worker_models: list[str] = Field(
        default_factory=lambda: ["gemini-2.5-flash-lite"],
        validation_alias="CHEAP_WORKER_MODELS",
    )
    balanced_worker_models: list[str] = Field(
        default_factory=lambda: ["gemini-2.5-flash"],
        validation_alias="BALANCED_WORKER_MODELS",
    )
    premium_models: list[str] = Field(
        default_factory=lambda: ["gemini-2.5-pro"],
        validation_alias="PREMIUM_MODELS",
    )
    synthesis_models: list[str] = Field(
        default_factory=lambda: ["gemini-2.5-pro", "gemini-2.5-flash"],
        validation_alias="SYNTHESIS_MODELS",
    )
    review_models: list[str] = Field(
        default_factory=lambda: ["gemini-2.5-flash", "gemini-2.5-pro"],
        validation_alias="REVIEW_MODELS",
    )
    evaluator_models: list[str] = Field(
        default_factory=lambda: ["gemini-2.5-flash", "gemini-2.5-pro"],
        validation_alias="EVALUATOR_MODELS",
    )
    max_concurrency: int = Field(default=12, validation_alias="MAX_CONCURRENCY", ge=1)
    max_candidates_per_subtask: int = Field(
        default=6,
        validation_alias="MAX_CANDIDATES_PER_SUBTASK",
        ge=1,
    )
    max_review_loops: int = Field(default=2, validation_alias="MAX_REVIEW_LOOPS", ge=0)
    per_call_timeout_seconds: int = Field(
        default=180,
        validation_alias="PER_CALL_TIMEOUT_SECONDS",
        ge=5,
    )
    max_retries: int = Field(default=2, validation_alias="MAX_RETRIES", ge=0)
    total_budget_cap_usd: float = Field(default=25.0, validation_alias="TOTAL_BUDGET_CAP_USD", gt=0)
    soft_budget_cap_usd: float = Field(default=20.0, validation_alias="SOFT_BUDGET_CAP_USD", gt=0)
    daily_budget_cap_usd: float = Field(default=100.0, validation_alias="DAILY_BUDGET_CAP_USD", gt=0)
    session_budget_cap_usd: float = Field(default=50.0, validation_alias="SESSION_BUDGET_CAP_USD", gt=0)
    target_utilization_ratio: float = Field(
        default=0.8,
        validation_alias="TARGET_UTILIZATION_RATIO",
        gt=0,
        le=1,
    )
    estimated_remaining_credit_usd: float = Field(
        default=100.0,
        validation_alias="ESTIMATED_REMAINING_CREDIT_USD",
        ge=0,
    )
    credit_expiry_datetime: datetime | None = Field(
        default=None,
        validation_alias="CREDIT_EXPIRY_DATETIME",
    )
    enable_batch_mode: bool = Field(default=True, validation_alias="ENABLE_BATCH_MODE")
    artifact_dir: Path = Field(default=Path("runs"), validation_alias="ARTIFACT_DIR")
    enable_repo_tools: bool = Field(default=True, validation_alias="ENABLE_REPO_TOOLS")
    enable_benchmarking: bool = Field(default=True, validation_alias="ENABLE_BENCHMARKING")
    enable_exhaust_mode: bool = Field(default=True, validation_alias="ENABLE_EXHAUST_MODE")
    router_state_dir: Path = Field(default=Path("runs") / "_state")
    prompt_max_output_tokens: int = Field(default=8192, ge=256)
    planner_max_output_tokens: int = Field(
        default=16384,
        validation_alias="PLANNER_MAX_OUTPUT_TOKENS",
        ge=512,
    )
    structured_thinking_budget: int = Field(
        default=512,
        validation_alias="STRUCTURED_THINKING_BUDGET",
        ge=1,
    )
    local_selection_candidate_limit: int = Field(default=20, ge=1)

    def __init__(self, **values: Any) -> None:
        remapped: dict[str, Any] = {}
        for key, value in values.items():
            field = self.__class__.model_fields.get(key)
            if field is not None and isinstance(field.validation_alias, str):
                remapped[field.validation_alias] = value
            else:
                remapped[key] = value
        super().__init__(**remapped)

    @field_validator(
        "cheap_worker_models",
        "balanced_worker_models",
        "premium_models",
        "synthesis_models",
        "review_models",
        "evaluator_models",
        mode="before",
    )
    @classmethod
    def _parse_model_lists(cls, value: Any) -> list[str]:
        return parse_csv(value)

    @model_validator(mode="after")
    def _validate_budget_caps(self) -> "Settings":
        if self.soft_budget_cap_usd > self.total_budget_cap_usd:
            raise ValueError("soft_budget_cap_usd cannot exceed total_budget_cap_usd")
        if not self.enable_exhaust_mode and self.default_mode is RunMode.EXHAUST:
            raise ValueError("default_mode cannot be exhaust when enable_exhaust_mode is false")
        return self

    @property
    def mode_profiles(self) -> dict[RunMode, ModeProfile]:
        return {
            RunMode.EFFICIENT: ModeProfile(
                base_candidates=1,
                benchmark_width=1,
                review_loops=1,
                enable_pairwise=False,
                enable_tournament=False,
                enable_batch=False,
            ),
            RunMode.AGGRESSIVE: ModeProfile(
                base_candidates=3,
                benchmark_width=2,
                review_loops=min(2, self.max_review_loops),
                enable_pairwise=True,
                enable_tournament=False,
                enable_batch=self.enable_batch_mode,
            ),
            RunMode.EXHAUST: ModeProfile(
                base_candidates=5,
                benchmark_width=4,
                review_loops=min(4, self.max_review_loops),
                enable_pairwise=True,
                enable_tournament=True,
                enable_batch=self.enable_batch_mode,
            ),
        }

    @property
    def pricing_table(self) -> dict[str, PricingEntry]:
        return {
            "gemini-2.5-flash-lite": PricingEntry(
                input_per_million_usd=0.10,
                output_per_million_usd=0.40,
                cached_input_per_million_usd=0.025,
                batch_discount_ratio=0.5,
            ),
            "gemini-2.5-flash": PricingEntry(
                input_per_million_usd=0.30,
                output_per_million_usd=2.50,
                cached_input_per_million_usd=0.075,
                batch_discount_ratio=0.5,
            ),
            "gemini-2.5-pro": PricingEntry(
                input_per_million_usd=1.25,
                output_per_million_usd=10.00,
                cached_input_per_million_usd=0.3125,
                batch_discount_ratio=0.5,
            ),
        }

    @property
    def batch_artifact_dir(self) -> Path:
        return self.artifact_dir / "batches"


def load_settings() -> Settings:
    return Settings()
