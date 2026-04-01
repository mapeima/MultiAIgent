from types import SimpleNamespace

from pydantic import BaseModel, ConfigDict

from multiagent.adapters.filesystem import FileSystemAdapter
from multiagent.adapters.gemini import GeminiGateway
from multiagent.adapters.logging import EventLogger, MetricsTracker
from multiagent.adapters.pricing import PriceBook
from multiagent.config import Settings
from multiagent.domain.models import ExecutionPhase
from multiagent.services.artifact_store import ArtifactStore


class DictSchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    mapping: dict[str, float]


def test_gateway_falls_back_when_schema_uses_additional_properties(tmp_path):
    settings = Settings(
        gemini_api_key="test-key",
        artifact_dir=tmp_path / "runs",
        router_state_dir=tmp_path / "state",
    )
    store = ArtifactStore(settings, "gateway", FileSystemAdapter())
    gateway = GeminiGateway(
        settings,
        PriceBook(settings),
        EventLogger(store.log_path(), "gateway"),
        MetricsTracker(),
    )
    captured = {}

    async def fake_call_with_retry(*, contents, config, **kwargs):
        captured["contents"] = contents
        captured["response_schema"] = getattr(config, "response_schema", None)
        return (
            SimpleNamespace(
                text='```json\n{"name":"ok","mapping":{"a":1.5}}\n```',
                parsed=None,
                usage_metadata=None,
            ),
            3,
        )

    gateway._call_with_retry = fake_call_with_retry  # type: ignore[method-assign]
    result = __import__("asyncio").run(
        gateway.generate_structured(
            model="gemini-2.5-flash",
            system_instruction="Return JSON.",
            user_prompt="Give me the payload.",
            schema=DictSchemaModel,
            phase=ExecutionPhase.PLANNING,
        )
    )
    assert captured["response_schema"] is None
    assert "The JSON must satisfy this schema" in captured["contents"]
    assert result.parsed.mapping == {"a": 1.5}
