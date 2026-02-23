from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ultimate_transcripter.audio import (
    extract_chunk,
    format_srt_timestamp,
    plan_chunks,
    probe_audio,
    require_binary,
)
from ultimate_transcripter.provider_openai import OpenAITranscriptionClient
from ultimate_transcripter.types import (
    ChunkSpec,
    PipelineConfig,
    RunSummary,
    TranscriptSegment,
)


def run_transcription(
    *,
    config: PipelineConfig,
    api_key: str,
    logger: Callable[[str], None] = print,
) -> RunSummary:
    if not config.input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {config.input_path}")
    if not config.input_path.is_file():
        raise RuntimeError(f"Input is not a file: {config.input_path}")

    require_binary("ffmpeg")
    require_binary("ffprobe")

    info = probe_audio(config.input_path)
    chunks = plan_chunks(
        duration_seconds=info.duration_seconds,
        chunk_seconds=config.chunk_seconds,
        overlap_seconds=config.overlap_seconds,
    )

    client = OpenAITranscriptionClient(
        api_key=api_key,
        model=config.model,
        api_base=config.api_base,
        timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = config.output_dir / ".state"
    chunk_dir = config.output_dir / ".chunks"
    state_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    _write_run_manifest(config=config, state_dir=state_dir, duration_seconds=info.duration_seconds)

    total = len(chunks)
    chunk_results: list[tuple[ChunkSpec, dict[str, Any]]] = []
    previous_tail = ""

    for idx, chunk in enumerate(chunks, start=1):
        state_path = state_dir / f"chunk_{chunk.index:05d}.json"
        state_payload: dict[str, Any] | None = None

        if config.resume:
            state_payload = _safe_load_json(state_path)

        if state_payload is not None and isinstance(state_payload.get("result"), dict):
            result_payload = state_payload["result"]
            logger(
                f"[{idx}/{total}] Reusing checkpoint chunk {chunk.index + 1}/{total} "
                f"({chunk.logical_start:.1f}s-{chunk.logical_end:.1f}s)"
            )
        else:
            logger(
                f"[{idx}/{total}] Transcribing chunk {chunk.index + 1}/{total} "
                f"({chunk.logical_start:.1f}s-{chunk.logical_end:.1f}s)"
            )

            chunk_audio_path = chunk_dir / f"chunk_{chunk.index:05d}.m4a"
            extract_chunk(
                input_path=config.input_path,
                output_path=chunk_audio_path,
                start_seconds=chunk.extract_start,
                duration_seconds=chunk.extract_duration,
            )

            request_prompt = _build_chunk_prompt(
                user_prompt=config.prompt,
                previous_tail=previous_tail,
            )
            result_payload = client.transcribe_file(
                audio_path=chunk_audio_path,
                language=config.language,
                prompt=request_prompt,
            )

            _write_json(
                state_path,
                {
                    "saved_at": _utc_now_iso(),
                    "chunk": asdict(chunk),
                    "result": result_payload,
                },
            )

            if not config.keep_temp_chunks and chunk_audio_path.exists():
                chunk_audio_path.unlink()

        previous_tail = _tail_text(str(result_payload.get("text") or ""), limit=320)
        chunk_results.append((chunk, result_payload))

    segments = _merge_segments(chunk_results)
    final_text = _compose_text(segments)

    base_name = config.input_path.stem
    text_path: Path | None = None
    srt_path: Path | None = None
    json_path: Path | None = None

    if "txt" in config.formats:
        text_path = config.output_dir / f"{base_name}.txt"
        text_path.write_text(final_text.strip() + "\n", encoding="utf-8")

    if "srt" in config.formats:
        srt_path = config.output_dir / f"{base_name}.srt"
        srt_path.write_text(_to_srt(segments), encoding="utf-8")

    if "json" in config.formats:
        json_path = config.output_dir / f"{base_name}.json"
        json_payload = {
            "input_file": str(config.input_path),
            "created_at": _utc_now_iso(),
            "model": config.model,
            "language_requested": config.language,
            "duration_seconds": info.duration_seconds,
            "chunk_seconds": config.chunk_seconds,
            "overlap_seconds": config.overlap_seconds,
            "segments": [
                {"start": seg.start, "end": seg.end, "text": seg.text}
                for seg in segments
            ],
            "text": final_text,
        }
        _write_json(json_path, json_payload)

    return RunSummary(
        input_path=config.input_path,
        output_dir=config.output_dir,
        text_path=text_path,
        srt_path=srt_path,
        json_path=json_path,
        chunk_count=len(chunks),
        duration_seconds=info.duration_seconds,
    )


def _write_run_manifest(
    *, config: PipelineConfig, state_dir: Path, duration_seconds: float
) -> None:
    manifest_path = state_dir / "run_manifest.json"
    payload = {
        "saved_at": _utc_now_iso(),
        "input_file": str(config.input_path),
        "output_dir": str(config.output_dir),
        "model": config.model,
        "language": config.language,
        "chunk_seconds": config.chunk_seconds,
        "overlap_seconds": config.overlap_seconds,
        "duration_seconds": duration_seconds,
    }
    _write_json(manifest_path, payload)


def _build_chunk_prompt(user_prompt: str | None, previous_tail: str) -> str | None:
    parts: list[str] = []
    if user_prompt:
        parts.append(user_prompt.strip())
    if previous_tail:
        parts.append(
            "Context from previous chunk (for continuity only; do not repeat verbatim):\n"
            + previous_tail
        )
    if not parts:
        return None
    return "\n\n".join(parts)


def _tail_text(text: str, *, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[-limit:]


def _merge_segments(
    chunk_results: list[tuple[ChunkSpec, dict[str, Any]]],
) -> list[TranscriptSegment]:
    merged: list[TranscriptSegment] = []

    for chunk, payload in chunk_results:
        raw_segments = payload.get("segments")
        if isinstance(raw_segments, list) and raw_segments:
            for raw in raw_segments:
                if not isinstance(raw, dict):
                    continue
                text = str(raw.get("text") or "").strip()
                if not text:
                    continue

                rel_start = _as_float(raw.get("start"), 0.0)
                rel_end = _as_float(raw.get("end"), rel_start)
                abs_start = chunk.extract_start + rel_start
                abs_end = chunk.extract_start + max(rel_end, rel_start)
                candidate = _fit_segment_to_logical_window(
                    start=abs_start,
                    end=abs_end,
                    text=text,
                    chunk_index=chunk.index,
                    logical_start=chunk.logical_start,
                    logical_end=chunk.logical_end,
                )
                if candidate is not None:
                    merged.append(candidate)
        else:
            text = _sanitize_text(str(payload.get("text") or ""))
            if text:
                merged.append(
                    TranscriptSegment(
                        start=chunk.logical_start,
                        end=chunk.logical_end,
                        text=text,
                        chunk_index=chunk.index,
                    )
                )

    merged.sort(key=lambda seg: (seg.start, seg.end))
    return _dedupe_segments(merged)


def _fit_segment_to_logical_window(
    *,
    start: float,
    end: float,
    text: str,
    chunk_index: int,
    logical_start: float,
    logical_end: float,
) -> TranscriptSegment | None:
    if end <= logical_start:
        return None
    if start >= logical_end:
        return None
    clipped_start = max(start, logical_start)
    clipped_end = min(end, logical_end)
    if clipped_end <= clipped_start:
        clipped_end = clipped_start + 0.01
    return TranscriptSegment(
        start=clipped_start,
        end=clipped_end,
        text=_normalize_spaces(text),
        chunk_index=chunk_index,
    )


def _dedupe_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    if not segments:
        return []

    deduped: list[TranscriptSegment] = [segments[0]]
    for current in segments[1:]:
        previous = deduped[-1]
        same_text = _canonical_text(previous.text) == _canonical_text(current.text)
        near_start = abs(previous.start - current.start) <= 1.0
        overlaps = current.start <= previous.end

        if same_text and (near_start or overlaps):
            deduped[-1] = TranscriptSegment(
                start=previous.start,
                end=max(previous.end, current.end),
                text=previous.text,
                chunk_index=previous.chunk_index,
            )
            continue

        deduped.append(current)
    return deduped


def _compose_text(segments: list[TranscriptSegment]) -> str:
    pieces: list[str] = []
    for segment in segments:
        text = _sanitize_text(segment.text)
        if not text:
            continue
        if not pieces:
            pieces.append(text)
            continue

        prev = pieces[-1]
        if text[0] in ".,!?;:)":
            pieces[-1] = prev + text
        elif prev.endswith(("-", "'", "\"", "(")):
            pieces[-1] = prev + text
        else:
            pieces.append(text)

    merged = " ".join(pieces)
    return _normalize_spaces(merged)


def _to_srt(segments: list[TranscriptSegment]) -> str:
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        lines.append(str(index))
        lines.append(
            f"{format_srt_timestamp(segment.start)} --> {format_srt_timestamp(segment.end)}"
        )
        lines.append(segment.text)
        lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _sanitize_text(value: str) -> str:
    normalized = _normalize_spaces(value)
    if not normalized:
        return ""
    return _trim_repeated_tail_words(normalized)


def _trim_repeated_tail_words(text: str) -> str:
    words = text.split(" ")
    if len(words) < 6:
        return text

    max_window = min(24, len(words) // 3)
    while True:
        trimmed = False
        for window in range(1, max_window + 1):
            repeat_count = _count_tail_repeats(words, window)
            if repeat_count < 3:
                continue

            keep_until = len(words) - (window * (repeat_count - 1))
            words = words[:keep_until]
            trimmed = True
            break

        if not trimmed:
            break
        if len(words) < 6:
            break
        max_window = min(24, len(words) // 3)

    return " ".join(words)


def _count_tail_repeats(words: list[str], window: int) -> int:
    if window <= 0 or len(words) < window * 2:
        return 1

    target = words[-window:]
    repeats = 1
    cursor = len(words) - (window * 2)
    while cursor >= 0:
        candidate = words[cursor : cursor + window]
        if candidate != target:
            break
        repeats += 1
        cursor -= window
    return repeats


def _canonical_text(value: str) -> str:
    lowered = _normalize_spaces(value).lower()
    return re.sub(r"[^\w]+", "", lowered, flags=re.UNICODE)


def _as_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
