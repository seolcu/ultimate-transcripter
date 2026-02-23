from __future__ import annotations

import mimetypes
import random
import time
from pathlib import Path
from typing import Any

import requests


class TranscriptionAPIError(RuntimeError):
    pass


class OpenAITranscriptionClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        api_base: str,
        timeout_seconds: int,
        max_retries: int,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def transcribe_file(
        self,
        *,
        audio_path: Path,
        language: str | None,
        prompt: str | None,
    ) -> dict[str, Any]:
        url = f"{self.api_base}/audio/transcriptions"
        mime_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"

        base_data: list[tuple[str, str]] = [
            ("model", self.model),
            ("response_format", "verbose_json"),
            ("temperature", "0"),
            ("timestamp_granularities[]", "segment"),
        ]
        if language:
            base_data.append(("language", language))
        if prompt:
            base_data.append(("prompt", prompt))

        headers = {"Authorization": f"Bearer {self.api_key}"}

        retriable_status = {408, 409, 429, 500, 502, 503, 504}
        attempt = 0
        last_error: Exception | None = None

        while attempt <= self.max_retries:
            attempt += 1
            try:
                with audio_path.open("rb") as file_obj:
                    response = requests.post(
                        url,
                        headers=headers,
                        data=base_data,
                        files={"file": (audio_path.name, file_obj, mime_type)},
                        timeout=self.timeout_seconds,
                    )
            except requests.RequestException as exc:
                last_error = exc
                if attempt > self.max_retries:
                    break
                self._sleep_backoff(attempt)
                continue

            if response.status_code in retriable_status and attempt <= self.max_retries:
                last_error = TranscriptionAPIError(
                    f"Transient API error ({response.status_code}): {self._extract_error(response)}"
                )
                self._sleep_backoff(attempt)
                continue

            if response.status_code >= 400:
                raise TranscriptionAPIError(
                    f"API error ({response.status_code}): {self._extract_error(response)}"
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise TranscriptionAPIError("API returned non-JSON response") from exc

            if not isinstance(payload, dict):
                raise TranscriptionAPIError("API returned unexpected payload type")
            return payload

        if last_error is not None:
            raise TranscriptionAPIError(
                f"Failed to transcribe after retries: {last_error}"
            ) from last_error
        raise TranscriptionAPIError("Failed to transcribe after retries")

    @staticmethod
    def _extract_error(response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return (response.text or "").strip() or "Unknown error"

        if isinstance(payload, dict):
            error_value = payload.get("error")
            if isinstance(error_value, dict):
                message = error_value.get("message")
                if isinstance(message, str) and message.strip():
                    return message.strip()
            if isinstance(error_value, str) and error_value.strip():
                return error_value.strip()
        return "Unknown error"

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        base = min(2 ** (attempt - 1), 30)
        jitter = random.uniform(0.0, 0.35)
        time.sleep(base + jitter)
