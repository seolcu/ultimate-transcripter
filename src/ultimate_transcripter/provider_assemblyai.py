from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import requests


class TranscriptionAPIError(RuntimeError):
    pass


class AssemblyAITranscriptionClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        api_base: str | None,
        timeout_seconds: int,
        max_retries: int,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.api_base = (api_base or "https://api.assemblyai.com/v2").rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def transcribe_file(
        self,
        *,
        audio_path: Path,
        language: str | None,
        prompt: str | None,
    ) -> dict[str, Any]:
        audio_url = self._upload_file(audio_path)

        payload: dict[str, Any] = {
            "audio_url": audio_url,
            "speech_models": _parse_speech_models(self.model),
        }
        if language:
            payload["language_code"] = language
        if prompt:
            payload["word_boost"] = _prompt_to_word_boost(prompt)

        transcript_id = self._create_transcript(payload)
        completed = self._poll_transcript(transcript_id)

        text = str(completed.get("text") or "").strip()
        segments = _segments_from_words(completed.get("words"))
        result: dict[str, Any] = {
            "text": text,
        }
        if segments:
            result["segments"] = segments
        return result

    def _upload_file(self, audio_path: Path) -> str:
        url = f"{self.api_base}/upload"
        headers = {"authorization": self.api_key}

        attempt = 0
        last_error: Exception | None = None
        retriable_status = {408, 409, 429, 500, 502, 503, 504}

        while attempt <= self.max_retries:
            attempt += 1
            try:
                with audio_path.open("rb") as file_obj:
                    response = requests.post(
                        url,
                        headers=headers,
                        data=file_obj,
                        timeout=self.timeout_seconds,
                    )
            except requests.RequestException as exc:
                last_error = exc
                if attempt > self.max_retries:
                    break
                _sleep_backoff(attempt)
                continue

            if response.status_code in retriable_status and attempt <= self.max_retries:
                last_error = TranscriptionAPIError(
                    f"Transient upload error ({response.status_code}): {_extract_error(response)}"
                )
                _sleep_backoff(attempt)
                continue

            if response.status_code >= 400:
                raise TranscriptionAPIError(
                    f"Upload API error ({response.status_code}): {_extract_error(response)}"
                )

            payload = _safe_json(response)
            upload_url = payload.get("upload_url")
            if not isinstance(upload_url, str) or not upload_url.strip():
                raise TranscriptionAPIError("Upload API did not return upload_url")
            return upload_url

        if last_error is not None:
            raise TranscriptionAPIError(f"Failed upload after retries: {last_error}") from last_error
        raise TranscriptionAPIError("Failed upload after retries")

    def _create_transcript(self, payload: dict[str, Any]) -> str:
        url = f"{self.api_base}/transcript"
        headers = {
            "authorization": self.api_key,
            "content-type": "application/json",
        }
        retriable_status = {408, 409, 429, 500, 502, 503, 504}

        attempt = 0
        last_error: Exception | None = None
        while attempt <= self.max_retries:
            attempt += 1
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt > self.max_retries:
                    break
                _sleep_backoff(attempt)
                continue

            if response.status_code in retriable_status and attempt <= self.max_retries:
                last_error = TranscriptionAPIError(
                    f"Transient create error ({response.status_code}): {_extract_error(response)}"
                )
                _sleep_backoff(attempt)
                continue

            if response.status_code >= 400:
                raise TranscriptionAPIError(
                    f"Create transcript API error ({response.status_code}): {_extract_error(response)}"
                )

            body = _safe_json(response)
            transcript_id = body.get("id")
            if not isinstance(transcript_id, str) or not transcript_id.strip():
                raise TranscriptionAPIError("Create transcript API did not return transcript id")
            return transcript_id

        if last_error is not None:
            raise TranscriptionAPIError(
                f"Failed to create transcript after retries: {last_error}"
            ) from last_error
        raise TranscriptionAPIError("Failed to create transcript after retries")

    def _poll_transcript(self, transcript_id: str) -> dict[str, Any]:
        url = f"{self.api_base}/transcript/{transcript_id}"
        headers = {"authorization": self.api_key}
        poll_seconds = 3.0
        retriable_status = {408, 409, 429, 500, 502, 503, 504}

        started = time.monotonic()
        hard_timeout_seconds = max(60, self.timeout_seconds) * 20
        transient_failures = 0

        while True:
            try:
                response = requests.get(url, headers=headers, timeout=self.timeout_seconds)
            except requests.RequestException as exc:
                transient_failures += 1
                if transient_failures > self.max_retries:
                    raise TranscriptionAPIError(f"Polling failed: {exc}") from exc
                _sleep_backoff(transient_failures)
                continue

            if response.status_code in retriable_status:
                transient_failures += 1
                if transient_failures > self.max_retries:
                    raise TranscriptionAPIError(
                        f"Polling API error ({response.status_code}): {_extract_error(response)}"
                    )
                _sleep_backoff(transient_failures)
                continue

            if response.status_code >= 400:
                raise TranscriptionAPIError(
                    f"Polling API error ({response.status_code}): {_extract_error(response)}"
                )

            transient_failures = 0

            body = _safe_json(response)
            status = str(body.get("status") or "").strip().lower()
            if status == "completed":
                return body
            if status == "error":
                error_msg = str(body.get("error") or "Unknown AssemblyAI error")
                raise TranscriptionAPIError(f"AssemblyAI transcription failed: {error_msg}")

            elapsed = time.monotonic() - started
            if elapsed > hard_timeout_seconds:
                raise TranscriptionAPIError(
                    "Polling timed out while waiting for AssemblyAI transcript completion"
                )
            time.sleep(poll_seconds)


def _extract_error(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        text = (response.text or "").strip()
        return text or "Unknown error"
    if not isinstance(payload, dict):
        return "Unknown error"
    error_value = payload.get("error")
    if isinstance(error_value, str) and error_value.strip():
        return error_value.strip()
    if isinstance(error_value, dict):
        message = error_value.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    detail = payload.get("detail")
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    text = (response.text or "").strip()
    return text or "Unknown error"


def _safe_json(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise TranscriptionAPIError("API returned non-JSON response") from exc
    if not isinstance(payload, dict):
        raise TranscriptionAPIError("API returned unexpected payload type")
    return payload


def _sleep_backoff(attempt: int) -> None:
    base = min(2 ** (attempt - 1), 30)
    jitter = random.uniform(0.0, 0.35)
    time.sleep(base + jitter)


def _prompt_to_word_boost(prompt: str) -> list[str]:
    words = [token.strip() for token in prompt.replace("\n", " ").split(" ") if token.strip()]
    unique: list[str] = []
    seen: set[str] = set()
    for word in words:
        normalized = word.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(word)
        if len(unique) >= 50:
            break
    return unique


def _segments_from_words(raw_words: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_words, list):
        return []

    segments: list[dict[str, Any]] = []
    cur_start_ms: int | None = None
    cur_end_ms: int | None = None
    cur_words: list[str] = []
    max_span_ms = 8000

    for item in raw_words:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue

        start_ms = _as_int(item.get("start"), -1)
        end_ms = _as_int(item.get("end"), -1)
        if start_ms < 0:
            continue
        if end_ms < start_ms:
            end_ms = start_ms

        if cur_start_ms is None:
            cur_start_ms = start_ms
            cur_end_ms = end_ms
            cur_words = [text]
            continue

        span_ms = end_ms - cur_start_ms
        should_split = span_ms >= max_span_ms or _is_sentence_terminator(cur_words[-1])
        if should_split:
            segments.append(
                {
                    "start": cur_start_ms / 1000.0,
                    "end": max(cur_end_ms or cur_start_ms, cur_start_ms) / 1000.0,
                    "text": " ".join(cur_words).strip(),
                }
            )
            cur_start_ms = start_ms
            cur_words = [text]
            cur_end_ms = end_ms
            continue

        cur_words.append(text)
        cur_end_ms = max(cur_end_ms or end_ms, end_ms)

    if cur_start_ms is not None and cur_words:
        segments.append(
            {
                "start": cur_start_ms / 1000.0,
                "end": max(cur_end_ms or cur_start_ms, cur_start_ms) / 1000.0,
                "text": " ".join(cur_words).strip(),
            }
        )

    return [segment for segment in segments if segment["text"]]


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _is_sentence_terminator(word: str) -> bool:
    return word.endswith(".") or word.endswith("?") or word.endswith("!")


def _parse_speech_models(raw_model: str) -> list[str]:
    models = [part.strip() for part in raw_model.split(",") if part.strip()]
    if not models:
        return ["universal-2"]
    return models
