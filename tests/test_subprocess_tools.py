import asyncio
from pathlib import Path

from multiagent.adapters.subprocess_tools import CommandResult, SubprocessTools


def test_resolve_optional_command_prefers_python_module(monkeypatch):
    tools = SubprocessTools()
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object() if name == "pytest" else None)
    command, reason = tools._resolve_optional_command("pytest", ["pytest", "-q"])
    assert reason is None
    assert command is not None
    assert command[1:3] == ["-m", "pytest"]
    assert command[3:] == ["-q"]


def test_resolve_optional_command_skips_external_batch_shim(monkeypatch):
    tools = SubprocessTools()
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    monkeypatch.setattr("shutil.which", lambda name: r"C:\shim\ruff.BAT")
    command, reason = tools._resolve_optional_command("ruff", ["ruff", "check", "."])
    assert command is None
    assert reason is not None
    assert "external shim" in reason


def test_run_returns_127_for_missing_command(tmp_path):
    result = asyncio.run(
        SubprocessTools().run(["definitely-missing-command-xyz"], cwd=tmp_path)
    )
    assert result.returncode == 127
    assert "cannot find the file" in result.stderr.lower() or "no such file" in result.stderr.lower()
