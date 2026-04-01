import os

import pytest

from multiagent.adapters.filesystem import FileSystemAdapter
from multiagent.adapters.gemini import GeminiGateway
from multiagent.adapters.logging import EventLogger, MetricsTracker
from multiagent.adapters.pricing import PriceBook
from multiagent.config import Settings
from multiagent.domain.models import ExecutionPhase
from multiagent.services.artifact_store import ArtifactStore


pytestmark = pytest.mark.live


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
def test_live_gemini_text_generation(tmp_path):
    settings = Settings(
        gemini_api_key=os.environ["GEMINI_API_KEY"],
        artifact_dir=tmp_path / "runs",
        router_state_dir=tmp_path / "state",
    )
    store = ArtifactStore(settings, "live", FileSystemAdapter())
    gateway = GeminiGateway(settings, PriceBook(settings), EventLogger(store.log_path(), "live"), MetricsTracker())
    result = __import__("asyncio").run(
        gateway.generate_text(
            model=settings.balanced_worker_models[0],
            system_instruction="Answer in one sentence.",
            user_prompt="Say hello to a test harness.",
            phase=ExecutionPhase.PLANNING,
            temperature=0.0,
            max_output_tokens=64,
        )
    )
    assert result.text
