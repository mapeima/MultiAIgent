from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, use_enum_values=False)


class RunMode(str, Enum):
    EFFICIENT = "efficient"
    AGGRESSIVE = "aggressive"
    EXHAUST = "exhaust"

    @classmethod
    def _missing_(cls, value: object) -> "RunMode" | None:
        if isinstance(value, str):
            normalized = value.strip().lower()
            for member in cls:
                if member.value == normalized:
                    return member
        return None


class AgentRole(str, Enum):
    RESEARCHER = "researcher"
    PM = "pm"
    ARCHITECT = "architect"
    BACKEND = "backend"
    FRONTEND = "frontend"
    INFRA = "infra"
    SECURITY = "security"
    SRE = "sre"
    QA = "qa"
    IMPLEMENTER = "implementer"
    DEBUGGER = "debugger"
    TESTER = "tester"
    REVIEWER = "reviewer"
    CRITIC = "critic"
    WRITER = "writer"
    OPTIMIZER = "optimizer"

    @classmethod
    def _missing_(cls, value: object) -> "AgentRole" | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "product_manager": cls.PM,
            "project_manager": cls.PM,
            "program_manager": cls.PM,
            "software_architect": cls.ARCHITECT,
            "system_architect": cls.ARCHITECT,
            "backend_engineer": cls.BACKEND,
            "frontend_engineer": cls.FRONTEND,
            "infrastructure": cls.INFRA,
            "devops": cls.INFRA,
            "devops_engineer": cls.INFRA,
            "security_engineer": cls.SECURITY,
            "appsec": cls.SECURITY,
            "application_security": cls.SECURITY,
            "security_sre": cls.SRE,
            "site_reliability_engineer": cls.SRE,
            "reliability_engineer": cls.SRE,
            "quality_assurance": cls.QA,
            "test_engineer": cls.QA,
            "tester": cls.TESTER,
            "implementation": cls.IMPLEMENTER,
        }
        if normalized in aliases:
            return aliases[normalized]
        for member in cls:
            if member.value == normalized:
                return member
        return None


class ModelTier(str, Enum):
    CHEAP = "cheap"
    BALANCED = "balanced"
    PREMIUM = "premium"
    SYNTHESIS = "synthesis"
    REVIEW = "review"
    EVALUATOR = "evaluator"

    @classmethod
    def _missing_(cls, value: object) -> "ModelTier" | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "cheap_worker_models": cls.CHEAP,
            "balanced_worker_models": cls.BALANCED,
            "premium_models": cls.PREMIUM,
            "synthesis_models": cls.SYNTHESIS,
            "review_models": cls.REVIEW,
            "evaluator_models": cls.EVALUATOR,
        }
        if normalized in aliases:
            return aliases[normalized]
        for member in cls:
            if member.value == normalized:
                return member
        return None


class ExecutionPhase(str, Enum):
    PLANNING = "planning"
    WORKERS = "workers"
    CANDIDATE_GENERATION = "candidate_generation"
    EVALUATION = "evaluation"
    SYNTHESIS = "synthesis"
    REVIEW = "review"
    BATCH = "batch"


class EvaluationStrategy(str, Enum):
    RUBRIC = "rubric"
    LLM_JUDGE = "llm_judge"
    PAIRWISE = "pairwise"
    TOURNAMENT = "tournament"


class ChangeAction(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


class ArtifactRef(StrictModel):
    kind: str
    path: str
    description: str


class TokenUsage(StrictModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    thoughts_tokens: int = 0
    cached_input_tokens: int = 0


class CodeChange(StrictModel):
    path: str
    action: ChangeAction
    content: str | None = None
    reason: str
    language: str | None = None


class RepoContextFile(StrictModel):
    path: str
    sha256: str
    summary: str
    excerpt: str


class RepoContext(StrictModel):
    repo_path: str
    selected_files: list[RepoContextFile] = Field(default_factory=list)
    selection_reason: str


class Subtask(StrictModel):
    id: str
    title: str
    role: AgentRole
    objective: str
    depends_on: list[str] = Field(default_factory=list)
    deliverables: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    importance_score: int = Field(ge=1, le=10)
    complexity_score: int = Field(ge=1, le=10)
    parallelizable: bool
    recommended_candidate_count: int = Field(ge=1, le=32)
    recommended_model_tier: ModelTier
    requires_review: bool
    model_override: str | None = None


class RecommendedGlobalStrategy(StrictModel):
    breadth_vs_depth: str
    expected_parallelism: int = Field(ge=1)
    suggested_budget_allocation: dict[str, float]


class TaskEdge(StrictModel):
    source: str
    target: str


class TaskGraph(StrictModel):
    nodes: list[str]
    edges: list[TaskEdge]


class Plan(StrictModel):
    task_summary: str
    execution_strategy: str
    assumptions: list[str] = Field(default_factory=list)
    subtasks: list[Subtask]
    final_acceptance_criteria: list[str] = Field(default_factory=list)
    recommended_global_strategy: RecommendedGlobalStrategy


class CandidateSpec(StrictModel):
    candidate_id: str
    subtask_id: str
    agent_role: AgentRole
    model: str
    temperature: float = Field(ge=0, le=2)
    prompt_variant: str
    reasoning_style: str
    strictness_level: str
    benchmark_axes: list[str] = Field(default_factory=list)
    mode: RunMode
    repo_context_hashes: dict[str, str] = Field(default_factory=dict)


class WorkerResult(StrictModel):
    summary: str
    detailed_result: str
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    suggested_files: list[str] = Field(default_factory=list)
    code_changes: list[CodeChange] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    model_used: str
    prompt_variant: str
    token_usage_estimate: TokenUsage
    follow_up_suggestions: list[str] = Field(default_factory=list)
    candidate_id: str | None = None


class CandidateExecution(StrictModel):
    spec: CandidateSpec
    result: WorkerResult
    latency_ms: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)
    actual_cost_usd: float | None = Field(default=None, ge=0)
    status: str = "completed"


class CriterionScore(StrictModel):
    name: str
    score: float = Field(ge=0, le=10)
    rationale: str


class EvaluationResult(StrictModel):
    candidate_id: str
    overall_score: float = Field(ge=0, le=10)
    rubric_scores: list[CriterionScore] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    recommended: bool = False
    comparison_notes: str = ""
    model_used: str
    strategy: EvaluationStrategy


class SubtaskSelection(StrictModel):
    subtask_id: str
    selected_candidate_id: str
    selected_result: WorkerResult
    candidate_results: list[CandidateExecution] = Field(default_factory=list)
    evaluations: list[EvaluationResult] = Field(default_factory=list)
    merged_candidate_ids: list[str] = Field(default_factory=list)


class SynthesisResult(StrictModel):
    final_response_markdown: str
    executive_summary: str
    merged_best_practices: list[str] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)
    confidence_map: dict[str, float] = Field(default_factory=dict)
    referenced_candidate_ids: list[str] = Field(default_factory=list)


class ReviewIssue(StrictModel):
    subtask_id: str | None = None
    description: str
    severity: str
    suggested_fix: str


class ReviewVerdict(StrictModel):
    passed: bool
    issues: list[ReviewIssue] = Field(default_factory=list)
    fixable_within_budget: bool
    suggested_followup_subtasks: list[Subtask] = Field(default_factory=list)
    rationale: str


class BudgetSnapshot(StrictModel):
    run_spent_usd: float = Field(ge=0)
    reserved_usd: float = Field(ge=0)
    daily_spent_usd: float = Field(ge=0)
    session_spent_usd: float = Field(ge=0)
    by_phase_usd: dict[str, float] = Field(default_factory=dict)
    hard_cap_usd: float = Field(ge=0)
    soft_cap_usd: float = Field(ge=0)
    target_utilization_ratio: float = Field(ge=0, le=1)
    forecast_to_completion_usd: float = Field(ge=0)
    blocked_reason: str | None = None


class RunRequest(StrictModel):
    goal: str
    mode: RunMode
    repo_path: str | None = None
    constraints: list[str] = Field(default_factory=list)
    apply_repo_changes: bool = True
    interactive: bool = True
    batch_enabled: bool = False
    benchmark_models: list[str] = Field(default_factory=list)
    prompt_variants: list[str] = Field(default_factory=list)
    temperatures: list[float] = Field(default_factory=list)
    source_run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchManifest(StrictModel):
    batch_id: str
    created_at: datetime
    model: str
    prompt_variant: str
    role: AgentRole
    request_count: int = Field(ge=1)
    local_request_path: str
    remote_job_name: str
    status: str
    run_id: str | None = None


class UtilizationRecommendation(StrictModel):
    recommended_spend_velocity_usd_per_day: float = Field(ge=0)
    urgency_score: float = Field(ge=0, le=1)
    concurrency_multiplier: float = Field(ge=0.1)
    candidate_multiplier: float = Field(ge=0.1)
    benchmark_multiplier: float = Field(ge=0.1)
    review_loop_target: int = Field(ge=0)
    enable_batch_mode: bool
    rationale: str


class RunReport(StrictModel):
    run_id: str
    status: str
    goal: str
    mode: RunMode
    plan: Plan
    selected_results: list[SubtaskSelection]
    synthesis: SynthesisResult
    review_verdict: ReviewVerdict
    budget_snapshot: BudgetSnapshot
    started_at: datetime
    completed_at: datetime
    artifact_root: str
    source_run_id: str | None = None
    benchmark_summary: dict[str, Any] | None = None


JsonDict = dict[str, Any]
PathLike = str | Path
