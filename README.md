# Multiagent Gemini Executor

`multiagent` is a local-first Python 3.12 project for running high-throughput, artifact-rich multi-agent workflows on the Google Gemini API. It is designed to turn limited-expiry API credits into useful work: broad planning, parallel execution, model benchmarking, candidate comparison, synthesis, review loops, and detailed audit trails.

## Features

- Three run modes: `efficient`, `aggressive`, `exhaust`
- DAG-aware async scheduling with candidate fan-out and model/prompt sweeps
- Strict Pydantic schemas for planner, workers, evaluator, synthesizer, and reviewer
- Configurable model routing, retries, timeouts, budget caps, and deadline-aware spend acceleration
- Local artifact store for plans, events, metrics, costs, candidate outputs, evaluations, and final reports
- Optional repo-aware workflows with guarded code application and validation
- Optional Gemini Batch API support for non-urgent sweeps and reconciled artifacts
- CLI built with Typer and Rich

## Quick Start

```powershell
uv sync
Copy-Item .env.example .env
uv run multiagent config show
uv run multiagent run "Build a FastAPI CRUD service for products with tests"
```

If you do not use `uv`, install with:

```powershell
pip install -e .[dev]
```

## Required Environment

Set `GEMINI_API_KEY` in `.env` or your shell environment. The project uses the official `google-genai` SDK directly.

## Common Commands

```powershell
uv run multiagent plan "Create a SaaS MVP architecture for X"
uv run multiagent run --mode aggressive "Design and implement JWT auth in this repo"
uv run multiagent run --mode exhaust --repo . "Analyze this repo, propose improvements, implement the top 3, and test them"
uv run multiagent benchmark "Implement feature flags in Python" --models gemini-2.5-flash,gemini-2.5-pro
uv run multiagent benchmark-prompts "Write a production FastAPI service" --variants default,strict,architect
uv run multiagent inspect <run_id>
uv run multiagent replay <run_id>
uv run multiagent budget recommend
uv run multiagent batch submit "Generate 20 architectural variants for X" --count 20
uv run multiagent batch reconcile <batch_id>
```

## Architecture

- `multiagent.domain`: strict schemas, enums, and run models
- `multiagent.adapters`: Gemini gateway, pricing, filesystem, subprocess, logging
- `multiagent.services`: orchestration, scheduling, planning, workers, evaluation, synthesis, review, budget, batch, repo support
- `multiagent.cli`: terminal UX

## Testing

```powershell
uv run pytest
```

The default suite uses fake Gemini adapters. Optional live smoke tests run only when `GEMINI_API_KEY` is present:

```powershell
uv run pytest -m live
```

## Limitations

- v1 is local-first and single-node only
- No web UI and no database
- Repo mutation is guarded, but it still depends on model-generated code changes and locally available validation tooling
- Price tables are configurable and may need updates as Gemini pricing changes
