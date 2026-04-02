from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.json import JSON
from rich.markdown import Markdown
from rich.table import Table

from multiagent.config import load_settings
from multiagent.domain.models import AgentRole, RunMode, RunRequest
from multiagent.services.orchestrator import Orchestrator
from multiagent.utils import parse_csv


app = typer.Typer(no_args_is_help=True)
config_app = typer.Typer()
batch_app = typer.Typer()
budget_app = typer.Typer()
app.add_typer(config_app, name="config")
app.add_typer(batch_app, name="batch")
app.add_typer(budget_app, name="budget")
console = Console()


def _parse_temperatures(value: str | None) -> list[float]:
    if not value:
        return []
    return [float(item) for item in parse_csv(value)]


def _console_safe_text(value: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    try:
        value.encode(encoding)
        return value
    except UnicodeEncodeError:
        return value.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _print_markdown(value: str) -> None:
    safe_value = _console_safe_text(value)
    try:
        console.print(Markdown(safe_value))
    except UnicodeEncodeError:
        console.print(safe_value)


@app.command()
def run(
    goal: str,
    mode: RunMode = typer.Option(None),
    repo: Path | None = typer.Option(None, file_okay=False),
    constraint: list[str] = typer.Option(None),
    no_apply: bool = typer.Option(False, "--no-apply"),
    models: str | None = typer.Option(None, "--models"),
    prompt_variants: str | None = typer.Option(None, "--variants"),
    temperatures: str | None = typer.Option(None, "--temperatures"),
) -> None:
    settings = load_settings()
    orchestrator = Orchestrator(settings)
    request = RunRequest(
        goal=goal,
        mode=mode or settings.default_mode,
        repo_path=str(repo) if repo else None,
        constraints=constraint or [],
        apply_repo_changes=not no_apply,
        benchmark_models=parse_csv(models),
        prompt_variants=parse_csv(prompt_variants),
        temperatures=_parse_temperatures(temperatures),
    )
    run_id, report = asyncio.run(orchestrator.run(request))
    console.print(f"Run: [bold]{run_id}[/bold]")
    console.print(f"Status: {report.status}")
    console.print(f"Artifacts: {report.artifact_root}")
    _print_markdown(report.synthesis.final_response_markdown)


@app.command()
def plan(
    goal: str,
    mode: RunMode = typer.Option(None),
    repo: Path | None = typer.Option(None, file_okay=False),
    constraint: list[str] = typer.Option(None),
) -> None:
    settings = load_settings()
    orchestrator = Orchestrator(settings)
    request = RunRequest(
        goal=goal,
        mode=mode or settings.default_mode,
        repo_path=str(repo) if repo else None,
        constraints=constraint or [],
        apply_repo_changes=False,
    )
    run_id, plan_result = asyncio.run(orchestrator.plan(request))
    console.print(f"Run: [bold]{run_id}[/bold]")
    console.print(JSON.from_data(plan_result.model_dump(mode='json')))


@app.command("benchmark")
def benchmark_command(
    goal: str,
    models: str = typer.Option(..., "--models"),
    repo: Path | None = typer.Option(None, file_okay=False),
    temperatures: str | None = typer.Option(None, "--temperatures"),
) -> None:
    settings = load_settings()
    orchestrator = Orchestrator(settings)
    run_id, report = asyncio.run(
        orchestrator.benchmark(
            goal=goal,
            models=parse_csv(models),
            temperatures=_parse_temperatures(temperatures),
            repo_path=str(repo) if repo else None,
        )
    )
    console.print(f"Benchmark run: [bold]{run_id}[/bold]")
    console.print(JSON.from_data(report.benchmark_summary or {}))


@app.command("benchmark-prompts")
def benchmark_prompts(
    goal: str,
    variants: str = typer.Option(..., "--variants"),
    model: str | None = typer.Option(None, "--model"),
    repo: Path | None = typer.Option(None, file_okay=False),
) -> None:
    settings = load_settings()
    orchestrator = Orchestrator(settings)
    run_id, report = asyncio.run(
        orchestrator.benchmark(
            goal=goal,
            models=[model] if model else None,
            prompt_variants=parse_csv(variants),
            repo_path=str(repo) if repo else None,
        )
    )
    console.print(f"Prompt benchmark run: [bold]{run_id}[/bold]")
    console.print(JSON.from_data(report.benchmark_summary or {}))


@app.command()
def inspect(run_id: str) -> None:
    settings = load_settings()
    orchestrator = Orchestrator(settings)
    summary = asyncio.run(orchestrator.inspect(run_id))
    table = Table(show_header=False)
    table.add_column("Field")
    table.add_column("Value")
    for key, value in summary.items():
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def replay(run_id: str) -> None:
    settings = load_settings()
    orchestrator = Orchestrator(settings)
    new_run_id, report = asyncio.run(orchestrator.replay(run_id))
    console.print(f"Replay run: [bold]{new_run_id}[/bold]")
    console.print(f"Status: {report.status}")
    console.print(f"Artifacts: {report.artifact_root}")


@app.command()
def resume(
    run_id: str,
    repo: Path | None = typer.Option(None, file_okay=False),
    no_apply: bool = typer.Option(False, "--no-apply"),
) -> None:
    settings = load_settings()
    orchestrator = Orchestrator(settings)
    new_run_id, report = asyncio.run(
        orchestrator.resume(
            run_id,
            repo_path=str(repo) if repo else None,
            apply_repo_changes=False if no_apply else None,
        )
    )
    console.print(f"Resume run: [bold]{new_run_id}[/bold]")
    console.print(f"Status: {report.status}")
    console.print(f"Artifacts: {report.artifact_root}")


@config_app.command("show")
def config_show() -> None:
    settings = load_settings()
    console.print(JSON.from_data(settings.model_dump(mode="json")))


@budget_app.command("recommend")
def budget_recommend(
    mode: RunMode = typer.Option(RunMode.AGGRESSIVE),
) -> None:
    settings = load_settings()
    orchestrator = Orchestrator(settings)
    recommendation = orchestrator.budget_recommendation(mode)
    console.print(JSON.from_data(recommendation))


@batch_app.command("submit")
def batch_submit(
    goal: str,
    count: int = typer.Option(20, min=1),
    model: str | None = typer.Option(None, "--model"),
    role: AgentRole = typer.Option(AgentRole.WRITER, "--role"),
    prompt_variant: str = typer.Option("default", "--variant"),
) -> None:
    settings = load_settings()
    orchestrator = Orchestrator(settings)
    manifest = asyncio.run(
        orchestrator.submit_batch(
            goal=goal,
            count=count,
            model=model or settings.balanced_worker_models[0],
            role=role,
            prompt_variant=prompt_variant,
        )
    )
    console.print(JSON.from_data(manifest.model_dump(mode="json")))


@batch_app.command("reconcile")
def batch_reconcile(batch_id: str) -> None:
    settings = load_settings()
    orchestrator = Orchestrator(settings)
    payload = asyncio.run(orchestrator.reconcile_batch(batch_id))
    console.print(JSON.from_data(payload))
