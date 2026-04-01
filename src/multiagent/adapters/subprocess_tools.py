from __future__ import annotations

import asyncio
import shutil
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
    async def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        timeout: int = 300,
    ) -> CommandResult:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
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
        if shutil.which("uv") is None:
            return CommandResult(["uv", *args], 127, "", "uv is not installed")
        return await self.run(["uv", *args], cwd=repo_path, timeout=timeout)

    async def _run_optional(
        self,
        executable: str,
        command: list[str],
        repo_path: Path,
        timeout: int,
    ) -> CommandResult:
        if shutil.which(executable) is None:
            return CommandResult(command, 0, "", f"{executable} is not installed; skipped")
        return await self.run(command, cwd=repo_path, timeout=timeout)
