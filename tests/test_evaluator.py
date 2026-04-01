import asyncio

from multiagent.adapters.filesystem import FileSystemAdapter
from multiagent.services.artifact_store import ArtifactStore
from multiagent.services.evaluator import EvaluatorService
from multiagent.services.prompts import PromptRegistry

from tests.conftest import FakeGateway, make_candidate_execution, make_subtask


def test_evaluator_ranks_candidates(base_settings):
    def bundle_handler(**kwargs):
        from multiagent.services.evaluator import EvaluationBundle
        from multiagent.domain.models import EvaluationResult, EvaluationStrategy

        return EvaluationBundle(
            evaluations=[
                EvaluationResult(
                    candidate_id="c1",
                    overall_score=7.5,
                    rubric_scores=[],
                    strengths=["good"],
                    weaknesses=[],
                    recommended=False,
                    comparison_notes="",
                    model_used="gemini-2.5-flash",
                    strategy=EvaluationStrategy.LLM_JUDGE,
                ),
                EvaluationResult(
                    candidate_id="c2",
                    overall_score=8.5,
                    rubric_scores=[],
                    strengths=["better"],
                    weaknesses=[],
                    recommended=False,
                    comparison_notes="",
                    model_used="gemini-2.5-flash",
                    strategy=EvaluationStrategy.LLM_JUDGE,
                ),
            ]
        )

    def pairwise_handler(**kwargs):
        from multiagent.services.evaluator import PairwiseDecision

        return PairwiseDecision(winner_candidate_id="c2", rationale="better")

    gateway = FakeGateway(
        handlers={
            "EvaluationBundle": bundle_handler,
            "PairwiseDecision": pairwise_handler,
        }
    )
    store = ArtifactStore(base_settings, "evaluator", FileSystemAdapter())
    service = EvaluatorService(base_settings, PromptRegistry(), gateway, store)
    ranked = asyncio.run(
        service.evaluate(
            goal="goal",
            subtask=make_subtask("s1"),
            mode=__import__("multiagent.domain.models", fromlist=["RunMode"]).RunMode.AGGRESSIVE,
            candidates=[make_candidate_execution("c1", 0.5), make_candidate_execution("c2", 0.7)],
        )
    )
    assert ranked[0].candidate_id == "c2"
    assert ranked[0].recommended is True
