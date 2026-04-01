from __future__ import annotations

import json
import uuid
from pathlib import Path

from multiagent.adapters.gemini import GeminiGateway
from multiagent.config import Settings
from multiagent.domain.models import AgentRole, BatchManifest
from multiagent.services.prompts import PromptRegistry
from multiagent.utils import ensure_directory, utc_now


class BatchService:
    def __init__(
        self,
        settings: Settings,
        gateway: GeminiGateway,
        prompts: PromptRegistry,
    ) -> None:
        self._settings = settings
        self._gateway = gateway
        self._prompts = prompts

    async def submit(
        self,
        *,
        goal: str,
        count: int,
        model: str,
        role: AgentRole,
        prompt_variant: str,
    ) -> BatchManifest:
        batch_id = str(uuid.uuid4())
        batch_dir = ensure_directory(self._settings.batch_artifact_dir / batch_id)
        request_path = batch_dir / "requests.jsonl"
        with request_path.open("w", encoding="utf-8") as handle:
            for index in range(1, count + 1):
                prompt = self._prompts.batch_variant(goal=goal, role=role, variant_index=index)
                request = {
                    "key": f"variant-{index}",
                    "request": {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [
                                    {
                                        "text": f"{prompt.system_instruction}\n\n{prompt.user_prompt}",
                                    }
                                ],
                            }
                        ],
                        "generation_config": {
                            "temperature": 0.7,
                            "response_mime_type": "text/plain",
                        },
                    },
                }
                handle.write(json.dumps(request, ensure_ascii=True))
                handle.write("\n")
        job = await self._gateway.create_batch_from_file(
            model=model,
            request_file=request_path,
            display_name=f"multiagent-{batch_id}",
        )
        manifest = BatchManifest(
            batch_id=batch_id,
            created_at=utc_now(),
            model=model,
            prompt_variant=prompt_variant,
            role=role,
            request_count=count,
            local_request_path=str(request_path),
            remote_job_name=job.name,
            status=str(job.state),
        )
        (batch_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return manifest

    async def reconcile(self, batch_id: str) -> dict[str, object]:
        batch_dir = self._settings.batch_artifact_dir / batch_id
        manifest_path = batch_dir / "manifest.json"
        manifest = BatchManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        download = await self._gateway.download_batch_output(manifest.remote_job_name)
        payload = {
            "batch_id": batch_id,
            "job_name": manifest.remote_job_name,
            "state": download.state,
            "line_count": len(download.lines),
            "output_file_name": download.output_file_name,
            "lines": download.lines,
        }
        (batch_dir / "responses.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summary_lines = [
            f"# Batch {batch_id}",
            "",
            f"- State: {download.state}",
            f"- Job name: {manifest.remote_job_name}",
            f"- Responses: {len(download.lines)}",
        ]
        (batch_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
        return payload
