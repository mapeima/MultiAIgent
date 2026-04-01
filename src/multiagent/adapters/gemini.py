from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, TypeAdapter

from multiagent.adapters.logging import EventLogger, MetricsTracker
from multiagent.adapters.pricing import PriceBook
from multiagent.config import Settings
from multiagent.domain.models import ExecutionPhase, TokenUsage
from multiagent.errors import ConfigurationError, SchemaValidationError
from multiagent.utils import estimate_tokens_from_text

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover
    genai = None
    types = None


T = TypeVar("T")


@dataclass(slots=True)
class GatewayCallResult(Generic[T]):
    parsed: T
    text: str
    model: str
    usage: TokenUsage
    latency_ms: int
    estimated_cost_usd: float
    actual_cost_usd: float | None
    raw: Any


@dataclass(slots=True)
class BatchDownloadResult:
    job_name: str
    state: str
    lines: list[dict[str, Any]]
    output_file_name: str | None
    raw_bytes: bytes | None


class GeminiGateway:
    def __init__(
        self,
        settings: Settings,
        price_book: PriceBook,
        logger: EventLogger,
        metrics: MetricsTracker,
    ) -> None:
        if genai is None or types is None:
            raise ConfigurationError("google-genai is not installed")
        self._settings = settings
        self._price_book = price_book
        self._logger = logger
        self._metrics = metrics
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not self._settings.gemini_api_key:
                raise ConfigurationError("GEMINI_API_KEY is required for Gemini operations")
            self._client = genai.Client(api_key=self._settings.gemini_api_key)
        return self._client

    async def generate_structured(
        self,
        *,
        model: str,
        system_instruction: str,
        user_prompt: str,
        schema: type[T] | Any,
        phase: ExecutionPhase,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> GatewayCallResult[T]:
        adapter = TypeAdapter(schema)
        use_native_schema = self._supports_native_response_schema(adapter)
        prompt = user_prompt
        thinking_budget = self._settings.structured_thinking_budget
        config = self._structured_config(
            system_instruction=system_instruction,
            schema=schema if use_native_schema else None,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_budget=thinking_budget,
        )
        if not use_native_schema:
            prompt = self._augment_prompt_with_schema(user_prompt, adapter)
            self._logger.log(
                level="INFO",
                phase=phase.value,
                model=model,
                event_type="structured_output_mode",
                status="json_only_fallback",
            )
        try:
            response, latency_ms = await self._call_with_retry(
                phase=phase,
                model=model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001
            suggested_budget = self._extract_suggested_thinking_budget(exc)
            if suggested_budget is not None and suggested_budget != thinking_budget:
                retry_config = self._structured_config(
                    system_instruction=system_instruction,
                    schema=schema if use_native_schema else None,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    thinking_budget=suggested_budget,
                )
                self._logger.log(
                    level="WARNING",
                    phase=phase.value,
                    model=model,
                    event_type="structured_output_mode",
                    status="thinking_budget_adjusted",
                    previous_budget=thinking_budget,
                    suggested_budget=suggested_budget,
                    error=str(exc),
                )
                response, latency_ms = await self._call_with_retry(
                    phase=phase,
                    model=model,
                    contents=prompt,
                    config=retry_config,
                )
            elif use_native_schema and self._is_schema_compatibility_error(exc):
                fallback_prompt = self._augment_prompt_with_schema(user_prompt, adapter)
                fallback_config = self._structured_config(
                    system_instruction=system_instruction,
                    schema=None,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    thinking_budget=thinking_budget,
                )
                self._logger.log(
                    level="WARNING",
                    phase=phase.value,
                    model=model,
                    event_type="structured_output_mode",
                    status="json_only_runtime_fallback",
                    error=str(exc),
                )
                response, latency_ms = await self._call_with_retry(
                    phase=phase,
                    model=model,
                    contents=fallback_prompt,
                    config=fallback_config,
                )
            else:
                raise
        try:
            parsed = self._parse_structured_response(adapter, response)
        except SchemaValidationError as exc:
            finish_reason = self._primary_finish_reason(response)
            if self._should_attempt_json_repair(exc, response):
                repair_status = (
                    "repair_attempt_truncated"
                    if self._is_truncated_response(response)
                    else "repair_attempt_malformed_json"
                )
                self._logger.log(
                    level="WARNING",
                    phase=phase.value,
                    model=model,
                    event_type="structured_output_parse",
                    status=repair_status,
                    finish_reason=finish_reason,
                    error=str(exc),
                )
                repair_prompt = self._repair_prompt(
                    user_prompt=prompt,
                    schema_adapter=adapter,
                    broken_json=response.text or "",
                )
                repair_config = self._structured_config(
                    system_instruction=system_instruction,
                    schema=None,
                    temperature=0.0,
                    max_output_tokens=max(
                        max_output_tokens or self._settings.prompt_max_output_tokens,
                        self._settings.planner_max_output_tokens,
                    ),
                    thinking_budget=thinking_budget,
                )
                repair_response, repair_latency_ms = await self._call_with_retry(
                    phase=phase,
                    model=model,
                    contents=repair_prompt,
                    config=repair_config,
                )
                parsed = self._parse_structured_response(adapter, repair_response)
                response = repair_response
                latency_ms += repair_latency_ms
            else:
                raise
        return self._build_call_result(
            model=model,
            response=response,
            parsed=parsed,
            latency_ms=latency_ms,
        )

    async def generate_text(
        self,
        *,
        model: str,
        system_instruction: str,
        user_prompt: str,
        phase: ExecutionPhase,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> GatewayCallResult[str]:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens or self._settings.prompt_max_output_tokens,
        )
        response, latency_ms = await self._call_with_retry(
            phase=phase,
            model=model,
            contents=user_prompt,
            config=config,
        )
        return self._build_call_result(
            model=model,
            response=response,
            parsed=response.text or "",
            latency_ms=latency_ms,
        )

    async def count_tokens(self, *, model: str, contents: str) -> int:
        try:
            response = await asyncio.wait_for(
                self.client.aio.models.count_tokens(model=model, contents=contents),
                timeout=self._settings.per_call_timeout_seconds,
            )
            total = response.total_tokens or 0
            return total if total > 0 else estimate_tokens_from_text(contents)
        except Exception:  # noqa: BLE001
            return estimate_tokens_from_text(contents)

    async def create_batch_from_file(
        self,
        *,
        model: str,
        request_file: Path,
        display_name: str,
    ) -> Any:
        uploaded = await self.client.aio.files.upload(file=request_file)
        job = await self.client.aio.batches.create(
            model=model,
            src=uploaded.name,
            config=types.CreateBatchJobConfig(display_name=display_name),
        )
        return job

    async def get_batch(self, name: str) -> Any:
        return await self.client.aio.batches.get(name=name)

    async def list_batches(self) -> list[Any]:
        pager = await self.client.aio.batches.list()
        return [item async for item in pager]

    async def download_batch_output(self, name: str) -> BatchDownloadResult:
        job = await self.get_batch(name)
        state = str(job.state)
        if job.dest is None:
            return BatchDownloadResult(job_name=name, state=state, lines=[], output_file_name=None, raw_bytes=None)
        if job.dest.inlined_responses:
            return BatchDownloadResult(
                job_name=name,
                state=state,
                lines=[item.model_dump(mode="json") for item in job.dest.inlined_responses],
                output_file_name=None,
                raw_bytes=None,
            )
        if job.dest.file_name:
            payload = await self.client.aio.files.download(file=job.dest.file_name)
            lines = []
            for line in payload.decode("utf-8").splitlines():
                if line.strip():
                    lines.append(json.loads(line))
            return BatchDownloadResult(
                job_name=name,
                state=state,
                lines=lines,
                output_file_name=job.dest.file_name,
                raw_bytes=payload,
            )
        return BatchDownloadResult(job_name=name, state=state, lines=[], output_file_name=None, raw_bytes=None)

    async def _call_with_retry(
        self,
        *,
        phase: ExecutionPhase,
        model: str,
        contents: str,
        config: Any,
    ) -> tuple[Any, int]:
        attempts = self._settings.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            start = time.perf_counter()
            try:
                response = await asyncio.wait_for(
                    self.client.aio.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    ),
                    timeout=self._settings.per_call_timeout_seconds,
                )
                latency_ms = int((time.perf_counter() - start) * 1000)
                self._metrics.increment(f"calls.model.{model}")
                self._metrics.observe(f"latency_ms.model.{model}", latency_ms)
                self._logger.log(
                    level="INFO",
                    phase=phase.value,
                    model=model,
                    event_type="gemini_call",
                    status="success",
                    retry_count=attempt - 1,
                    latency_ms=latency_ms,
                )
                return response, latency_ms
            except Exception as exc:  # noqa: BLE001
                latency_ms = int((time.perf_counter() - start) * 1000)
                self._metrics.increment(f"failures.model.{model}")
                self._logger.log(
                    level="ERROR",
                    phase=phase.value,
                    model=model,
                    event_type="gemini_call",
                    status="error",
                    retry_count=attempt - 1,
                    latency_ms=latency_ms,
                    error=str(exc),
                )
                last_error = exc
                if self._is_non_retryable_error(exc):
                    raise
                if attempt >= attempts:
                    raise
                await asyncio.sleep(min(2**attempt, 8))
        assert last_error is not None
        raise last_error

    def _build_call_result[TResult](
        self,
        *,
        model: str,
        response: Any,
        parsed: TResult,
        latency_ms: int,
    ) -> GatewayCallResult[TResult]:
        usage = self._extract_usage(response)
        estimated = self._price_book.estimate_cost(model=model, usage=usage)
        return GatewayCallResult(
            parsed=parsed,
            text=response.text or "",
            model=model,
            usage=usage,
            latency_ms=latency_ms,
            estimated_cost_usd=estimated.total_cost_usd,
            actual_cost_usd=estimated.total_cost_usd,
            raw=response,
        )

    def _extract_usage(self, response: Any) -> TokenUsage:
        metadata = getattr(response, "usage_metadata", None)
        if metadata is None:
            text = response.text or ""
            estimate = estimate_tokens_from_text(text)
            return TokenUsage(input_tokens=estimate, output_tokens=estimate, total_tokens=estimate * 2)
        prompt_tokens = getattr(metadata, "prompt_token_count", 0) or 0
        output_tokens = getattr(metadata, "candidates_token_count", 0) or 0
        total_tokens = getattr(metadata, "total_token_count", 0) or prompt_tokens + output_tokens
        thoughts_tokens = getattr(metadata, "thoughts_token_count", 0) or 0
        cached_tokens = getattr(metadata, "cached_content_token_count", 0) or 0
        return TokenUsage(
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            thoughts_tokens=thoughts_tokens,
            cached_input_tokens=cached_tokens,
        )

    def _structured_config(
        self,
        *,
        system_instruction: str,
        schema: type[T] | Any | None,
        temperature: float,
        max_output_tokens: int | None,
        thinking_budget: int | None,
    ) -> Any:
        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=schema,
            thinking_config=types.ThinkingConfig(
                thinking_budget=thinking_budget,
                include_thoughts=False,
            )
            if thinking_budget is not None
            else None,
            temperature=temperature,
            max_output_tokens=max_output_tokens or self._settings.prompt_max_output_tokens,
        )

    def _supports_native_response_schema(self, adapter: TypeAdapter[Any]) -> bool:
        try:
            schema = adapter.json_schema()
        except Exception:  # noqa: BLE001
            return False
        return not self._schema_contains_unsupported_keywords(schema)

    def _schema_contains_unsupported_keywords(self, payload: Any) -> bool:
        if isinstance(payload, dict):
            if "additionalProperties" in payload or "patternProperties" in payload:
                return True
            return any(self._schema_contains_unsupported_keywords(value) for value in payload.values())
        if isinstance(payload, list):
            return any(self._schema_contains_unsupported_keywords(item) for item in payload)
        return False

    def _augment_prompt_with_schema(self, user_prompt: str, adapter: TypeAdapter[Any]) -> str:
        try:
            schema_payload = adapter.json_schema()
        except Exception:  # noqa: BLE001
            schema_payload = {"type": "object"}
        schema_text = json.dumps(schema_payload, indent=2, ensure_ascii=True)
        return (
            f"{user_prompt}\n\n"
            "Return only valid JSON with no prose and no markdown fences. "
            "The JSON must satisfy this schema:\n"
            f"{schema_text}"
        )

    def _parse_structured_response(self, adapter: TypeAdapter[T], response: Any) -> T:
        validation_errors: list[str] = []
        try:
            if getattr(response, "parsed", None) is not None:
                return adapter.validate_python(response.parsed)
        except Exception as exc:  # noqa: BLE001
            validation_errors.append(str(exc))

        text = response.text or ""
        for candidate in self._candidate_json_payloads(text):
            try:
                return adapter.validate_json(candidate)
            except Exception as exc:  # noqa: BLE001
                validation_errors.append(str(exc))
                continue
        details = validation_errors[0] if validation_errors else "unknown validation error"
        raise SchemaValidationError(
            f"Unable to parse structured JSON response: {details}. Raw prefix: {text[:500]}"
        )

    def _candidate_json_payloads(self, text: str) -> list[str]:
        stripped = text.strip()
        candidates: list[str] = []
        if stripped:
            candidates.append(stripped)
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
        if fence_match:
            candidates.append(fence_match.group(1).strip())
        for opening, closing in (("{", "}"), ("[", "]")):
            start = stripped.find(opening)
            end = stripped.rfind(closing)
            if start != -1 and end > start:
                candidates.append(stripped[start : end + 1].strip())
        seen: list[str] = []
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.append(candidate)
        return seen

    def _is_schema_compatibility_error(self, exc: Exception) -> bool:
        message = str(exc)
        return "not supported in the Gemini API" in message or "additionalProperties" in message

    def _is_non_retryable_error(self, exc: Exception) -> bool:
        return isinstance(exc, ValueError) and self._is_schema_compatibility_error(exc)

    def _extract_suggested_thinking_budget(self, exc: Exception) -> int | None:
        message = str(exc)
        match = re.search(r"thinking budget \d+ is invalid\. Please choose a value between (\d+) and (\d+)", message, re.IGNORECASE)
        if match:
            return int(match.group(1))
        if "Budget 0 is invalid" in message:
            return max(512, self._settings.structured_thinking_budget)
        return None

    def _primary_finish_reason(self, response: Any) -> str | None:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        return str(getattr(candidates[0], "finish_reason", None))

    def _is_truncated_response(self, response: Any) -> bool:
        finish_reason = self._primary_finish_reason(response)
        return finish_reason is not None and "MAX_TOKENS" in finish_reason

    def _should_attempt_json_repair(self, exc: SchemaValidationError, response: Any) -> bool:
        text = (response.text or "").strip()
        if not text:
            return False
        if self._is_truncated_response(response):
            return True
        return self._is_invalid_json_error(exc)

    def _is_invalid_json_error(self, exc: Exception) -> bool:
        message = str(exc)
        return "Invalid JSON" in message or "json_invalid" in message

    def _repair_prompt(
        self,
        *,
        user_prompt: str,
        schema_adapter: TypeAdapter[Any],
        broken_json: str,
    ) -> str:
        try:
            schema_payload = schema_adapter.json_schema()
        except Exception:  # noqa: BLE001
            schema_payload = {"type": "object"}
        return (
            "Repair and complete the following malformed or truncated JSON so it becomes valid JSON matching the schema exactly. "
            "Output only valid JSON.\n\n"
            f"Original task prompt:\n{user_prompt}\n\n"
            "Required JSON schema:\n"
            f"{json.dumps(schema_payload, indent=2, ensure_ascii=True)}\n\n"
            "Broken JSON to repair:\n"
            f"{broken_json}"
        )
