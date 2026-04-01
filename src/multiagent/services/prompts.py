from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from multiagent.domain.models import AgentRole, CandidateExecution, RepoContext, RunMode, Subtask
from multiagent.utils import compact_text


@dataclass(slots=True)
class PromptPayload:
    system_instruction: str
    user_prompt: str


class PromptRegistry:
    def planner(
        self,
        *,
        goal: str,
        constraints: list[str],
        mode: RunMode,
        repo_summary: str | None,
        variant: str = "default",
    ) -> PromptPayload:
        subtask_hint = {
            RunMode.EFFICIENT: "Prefer 5-8 subtasks.",
            RunMode.AGGRESSIVE: "Prefer 8-12 subtasks.",
            RunMode.EXHAUST: "Prefer 10-16 subtasks.",
        }[mode]
        repo_block = f"Repository summary:\n{repo_summary}\n\n" if repo_summary else ""
        constraints_block = "\n".join(f"- {item}" for item in constraints) or "- None"
        user_prompt = (
            f"Goal:\n{goal}\n\n"
            f"Mode: {mode.value}\n\n"
            f"{repo_block}"
            f"Constraints:\n{constraints_block}\n\n"
            "Produce a strict execution plan that decomposes work into DAG-friendly subtasks. "
            "In aggressive and exhaust modes, prefer smaller subtasks, more parallelism, and more candidate recommendations. "
            "Keep the JSON concise: use short phrases, not paragraphs, for assumptions, deliverables, and acceptance criteria. "
            "Keep task_summary and execution_strategy under 120 words each. "
            f"{subtask_hint} "
            f"{self._variant_hint(variant)}"
        )
        system_instruction = (
            "You are a planning model for production engineering workflows. "
            "Return only data that matches the response schema. "
            "Be explicit, compact, and technically rigorous."
        )
        return PromptPayload(system_instruction=system_instruction, user_prompt=user_prompt)

    def worker(
        self,
        *,
        role: AgentRole,
        goal: str,
        subtask: Subtask,
        mode: RunMode,
        prior_artifacts: list[dict[str, Any]],
        repo_context: RepoContext | None,
        candidate_metadata: dict[str, Any],
        variant: str,
    ) -> PromptPayload:
        repo_block = ""
        if repo_context and repo_context.selected_files:
            file_lines = []
            for item in repo_context.selected_files:
                file_lines.append(f"Path: {item.path}\nSummary: {item.summary}\nExcerpt:\n{item.excerpt}")
            repo_block = "\n\nRelevant repo context:\n" + "\n\n".join(file_lines)
        artifacts_block = compact_text(str(prior_artifacts), limit=6000)
        system_instruction = (
            f"You are the {role.value} worker in a production multi-agent system. "
            "Be direct, practical, and non-roleplay. "
            "Return only output that matches the schema. "
            "When proposing code changes, provide full intended file contents in code_changes[].content."
        )
        user_prompt = (
            f"Global goal:\n{goal}\n\n"
            f"Execution mode: {mode.value}\n\n"
            f"Subtask:\n{subtask.model_dump_json(indent=2)}\n\n"
            f"Prior artifacts:\n{artifacts_block}\n\n"
            f"Candidate strategy metadata:\n{candidate_metadata}\n"
            f"{repo_block}\n\n"
            f"{self._variant_hint(variant)}\n"
            "Meet the acceptance criteria. Surface risks and follow-up suggestions. "
            "If repo context is insufficient, state the gap in risks[]."
        )
        return PromptPayload(system_instruction=system_instruction, user_prompt=user_prompt)

    def evaluator(
        self,
        *,
        goal: str,
        subtask: Subtask,
        candidates: list[CandidateExecution],
        variant: str = "default",
    ) -> PromptPayload:
        candidate_block = "\n\n".join(
            f"Candidate {item.spec.candidate_id}:\n"
            f"Model={item.spec.model}; Variant={item.spec.prompt_variant}; "
            f"Summary={item.result.summary}\n"
            f"Detailed={compact_text(item.result.detailed_result, 1800)}"
            for item in candidates
        )
        system_instruction = (
            "You are an evaluation model for engineering work. "
            "Score candidates against correctness, completeness, coherence, adherence, implementability, testability, confidence, and redundancy."
        )
        user_prompt = (
            f"Goal:\n{goal}\n\n"
            f"Subtask:\n{subtask.model_dump_json(indent=2)}\n\n"
            f"Candidates:\n{candidate_block}\n\n"
            f"{self._variant_hint(variant)}"
        )
        return PromptPayload(system_instruction=system_instruction, user_prompt=user_prompt)

    def pairwise(
        self,
        *,
        goal: str,
        subtask: Subtask,
        left: CandidateExecution,
        right: CandidateExecution,
    ) -> PromptPayload:
        system_instruction = (
            "You compare two candidate engineering outputs and pick the stronger one. "
            "Return only structured output."
        )
        user_prompt = (
            f"Goal:\n{goal}\n\n"
            f"Subtask:\n{subtask.model_dump_json(indent=2)}\n\n"
            f"Candidate A ({left.spec.candidate_id}):\n{left.result.model_dump_json(indent=2)}\n\n"
            f"Candidate B ({right.spec.candidate_id}):\n{right.result.model_dump_json(indent=2)}"
        )
        return PromptPayload(system_instruction=system_instruction, user_prompt=user_prompt)

    def synthesizer(
        self,
        *,
        goal: str,
        plan_summary: str,
        selections: list[dict[str, Any]],
        mode: RunMode,
    ) -> PromptPayload:
        system_instruction = (
            "You synthesize multi-agent outputs into a single coherent engineering deliverable. "
            "Resolve contradictions, expose unresolved issues, and preserve the best ideas."
        )
        user_prompt = (
            f"Goal:\n{goal}\n\n"
            f"Mode: {mode.value}\n\n"
            f"Plan summary:\n{plan_summary}\n\n"
            f"Selected results:\n{compact_text(str(selections), 14000)}"
        )
        return PromptPayload(system_instruction=system_instruction, user_prompt=user_prompt)

    def reviewer(
        self,
        *,
        goal: str,
        acceptance_criteria: list[str],
        synthesis_markdown: str,
        mode: RunMode,
    ) -> PromptPayload:
        criteria_block = "\n".join(f"- {item}" for item in acceptance_criteria)
        system_instruction = (
            "You review a synthesized engineering result against the original goal and acceptance criteria. "
            "Return a pass/fail verdict, issues, and follow-up subtasks if needed."
        )
        user_prompt = (
            f"Goal:\n{goal}\n\n"
            f"Mode: {mode.value}\n\n"
            f"Acceptance criteria:\n{criteria_block}\n\n"
            f"Synthesized result:\n{synthesis_markdown}"
        )
        return PromptPayload(system_instruction=system_instruction, user_prompt=user_prompt)

    def file_selector(
        self,
        *,
        goal: str,
        subtask: Subtask,
        file_summaries: list[dict[str, str]],
    ) -> PromptPayload:
        system_instruction = (
            "You pick the smallest set of repository files needed for a worker subtask. "
            "Optimize for relevance and minimal context volume."
        )
        user_prompt = (
            f"Goal:\n{goal}\n\n"
            f"Subtask:\n{subtask.model_dump_json(indent=2)}\n\n"
            f"File summaries:\n{file_summaries}"
        )
        return PromptPayload(system_instruction=system_instruction, user_prompt=user_prompt)

    def batch_variant(
        self,
        *,
        goal: str,
        role: AgentRole,
        variant_index: int,
    ) -> PromptPayload:
        system_instruction = (
            f"You are a {role.value} worker producing one distinct high-quality variant. "
            "Focus on usefulness over stylistic novelty."
        )
        user_prompt = (
            f"Task:\n{goal}\n\n"
            f"Produce variant number {variant_index}. Make it meaningfully distinct from common defaults."
        )
        return PromptPayload(system_instruction=system_instruction, user_prompt=user_prompt)

    def _variant_hint(self, variant: str) -> str:
        hints = {
            "default": "Default variant: balanced brevity and completeness.",
            "strict": "Strict variant: emphasize acceptance criteria, testability, and explicit tradeoffs.",
            "architect": "Architect variant: emphasize system design, interfaces, and failure modes.",
            "critic": "Critic variant: emphasize edge cases, flaws, and corrective actions.",
            "creative": "Creative variant: broaden exploration while remaining practical.",
        }
        return hints.get(variant, f"Variant hint: {variant}")
