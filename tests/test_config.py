from multiagent.config import Settings


def test_settings_parse_csv(monkeypatch, tmp_path):
    monkeypatch.setenv("CHEAP_WORKER_MODELS", "a,b,c")
    settings = Settings(
        artifact_dir=tmp_path / "runs",
        router_state_dir=tmp_path / "state",
        gemini_api_key="x",
    )
    assert settings.cheap_worker_models == ["a", "b", "c"]


def test_settings_enforce_budget_order(tmp_path):
    try:
        Settings(
            gemini_api_key="x",
            artifact_dir=tmp_path / "runs",
            router_state_dir=tmp_path / "state",
            total_budget_cap_usd=5,
            soft_budget_cap_usd=6,
        )
    except ValueError as exc:
        assert "soft_budget_cap_usd" in str(exc)
    else:
        raise AssertionError("Expected settings validation error")
