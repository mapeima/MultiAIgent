from multiagent.adapters.filesystem import FileSystemAdapter
from multiagent.domain.models import RunMode
from multiagent.services.candidate_generator import CandidateGenerator
from multiagent.services.model_router import ModelRouter
from multiagent.services.utilization import ExecutionBudgetTuning

from tests.conftest import make_subtask


def test_candidate_generator_expands_for_exhaust(base_settings):
    router = ModelRouter(base_settings, FileSystemAdapter())
    generator = CandidateGenerator(base_settings, router)
    tuning = ExecutionBudgetTuning(
        max_concurrency=8,
        candidate_limit=6,
        benchmark_width=4,
        review_loops=2,
    )
    subtask = make_subtask("s1")
    candidates = generator.generate(subtask=subtask, mode=RunMode.EXHAUST, tuning=tuning)
    assert len(candidates) >= 4
    assert len({item.model for item in candidates}) >= 1
    assert any(item.prompt_variant == "strict" for item in candidates)
