from __future__ import annotations

import asyncio
import importlib.util
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(slots=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class SubprocessTools:
    _PYTHON_TOOL_MODULES = {
        "pytest": "pytest",
        "ruff": "ruff",
        "mypy": "mypy",
    }

    async def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        timeout: int = 300,
    ) -> CommandResult:
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            return CommandResult(
                command=list(command),
                returncode=127,
                stdout="",
                stderr=str(exc),
            )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()
            return CommandResult(
                command=list(command),
                returncode=124,
                stdout=stdout.decode(),
                stderr=(stderr.decode() + "\nTimed out").strip(),
            )
        return CommandResult(
            command=list(command),
            returncode=process.returncode or 0,
            stdout=stdout.decode(),
            stderr=stderr.decode(),
        )

    async def run_pytest(self, repo_path: Path, timeout: int = 600) -> CommandResult:
        return await self._run_optional("pytest", ["pytest", "-q"], repo_path, timeout)

    async def run_ruff(self, repo_path: Path, timeout: int = 300) -> CommandResult:
        return await self._run_optional("ruff", ["ruff", "check", "."], repo_path, timeout)

    async def run_mypy(self, repo_path: Path, timeout: int = 600) -> CommandResult:
        return await self._run_optional("mypy", ["mypy", "."], repo_path, timeout)

    async def run_uv_command(
        self,
        repo_path: Path,
        *args: str,
        timeout: int = 600,
    ) -> CommandResult:
        uv_executable = shutil.which("uv")
        if uv_executable is None:
            return CommandResult(["uv", *args], 127, "", "uv is not installed")
        return await self.run([uv_executable, *args], cwd=repo_path, timeout=timeout)

    async def run_py_compile(
        self,
        repo_path: Path,
        paths: Sequence[Path],
        timeout: int = 120,
    ) -> CommandResult:
        python_files = [str(path) for path in paths if path.suffix == ".py"]
        if not python_files:
            return CommandResult(["py_compile"], 0, "", "No Python files changed; skipped")
        return await self.run([sys.executable, "-m", "py_compile", *python_files], cwd=repo_path, timeout=timeout)

    async def _run_optional(
        self,
        executable: str,
        command: list[str],
        repo_path: Path,
        timeout: int,
    ) -> CommandResult:
        resolved_command, skip_reason = self._resolve_optional_command(executable, command)
        if resolved_command is None:
            return CommandResult(command, 0, "", skip_reason or f"{executable} is not installed; skipped")
        return await self.run(resolved_command, cwd=repo_path, timeout=timeout)

    def _resolve_optional_command(
        self,
        executable: str,
        command: list[str],
    ) -> tuple[list[str] | None, str | None]:
        module_name = self._PYTHON_TOOL_MODULES.get(executable)
        if module_name and importlib.util.find_spec(module_name) is not None:
            return [sys.executable, "-m", module_name, *command[1:]], None

        resolved = shutil.which(executable)
        if resolved is None:
            return None, f"{executable} is not installed; skipped"

        suffix = Path(resolved).suffix.lower()
        if suffix in {".bat", ".cmd"} and module_name is not None:
            return (
                None,
                f"{executable} is not installed in the current Python environment; found external shim {resolved}; skipped",
            )

        return [resolved, *command[1:]], None
