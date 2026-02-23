from __future__ import annotations

import json
import re
from collections import Counter
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
from ultimate_transcripter.provider_assemblyai import AssemblyAITranscriptionClient
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

    client = _build_transcription_client(config=config, api_key=api_key)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = config.output_dir / ".state"
    chunk_dir = config.output_dir / ".chunks"
    state_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    _validate_resume_compatibility(
        config=config,
        state_dir=state_dir,
        duration_seconds=info.duration_seconds,
    )
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
        "provider": config.provider,
        "model": config.model,
        "language": config.language,
        "chunk_seconds": config.chunk_seconds,
        "overlap_seconds": config.overlap_seconds,
        "duration_seconds": duration_seconds,
    }
    _write_json(manifest_path, payload)


def _validate_resume_compatibility(
    *, config: PipelineConfig, state_dir: Path, duration_seconds: float
) -> None:
    manifest_path = state_dir / "run_manifest.json"
    existing = _safe_load_json(manifest_path)
    if not existing:
        return

    existing_provider = existing.get("provider")
    if existing_provider is None:
        existing_provider = "openai"

    checks: list[tuple[str, Any, Any]] = [
        ("input_file", str(config.input_path), existing.get("input_file")),
        ("provider", config.provider, existing_provider),
        ("model", config.model, existing.get("model")),
        ("language", config.language, existing.get("language")),
        ("chunk_seconds", config.chunk_seconds, existing.get("chunk_seconds")),
        ("overlap_seconds", config.overlap_seconds, existing.get("overlap_seconds")),
    ]

    mismatches: list[str] = []
    for key, expected, actual in checks:
        if expected != actual:
            mismatches.append(f"{key}={actual!r} (current: {expected!r})")

    saved_duration = _as_float(existing.get("duration_seconds"), duration_seconds)
    if abs(saved_duration - duration_seconds) > 1.0:
        mismatches.append(
            f"duration_seconds={saved_duration:.3f} (current: {duration_seconds:.3f})"
        )

    if mismatches and config.resume:
        details = ", ".join(mismatches)
        raise RuntimeError(
            "Existing checkpoint manifest does not match current run options: "
            f"{details}. Use --no-resume or a different --output-dir."
        )


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


def _build_transcription_client(*, config: PipelineConfig, api_key: str) -> Any:
    provider = config.provider.strip().lower()
    if provider == "openai":
        return OpenAITranscriptionClient(
            api_key=api_key,
            model=config.model,
            api_base=config.api_base,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
        )
    if provider == "assemblyai":
        return AssemblyAITranscriptionClient(
            api_key=api_key,
            model=config.model,
            api_base=config.api_base,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
        )
    raise RuntimeError(f"Unsupported provider: {config.provider}")


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
    return _filter_segments(_dedupe_segments(merged))


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
    cleaned_text = _sanitize_text(text)
    if not cleaned_text:
        return None

    return TranscriptSegment(
        start=clipped_start,
        end=clipped_end,
        text=cleaned_text,
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
        if _canonical_text(prev) == _canonical_text(text):
            continue
        if text[0] in ".,!?;:)":
            pieces[-1] = prev + text
        elif prev.endswith(("-", "'", "\"", "(")):
            pieces[-1] = prev + text
        else:
            pieces.append(text)

    merged = " ".join(pieces)
    return _sanitize_document_text(merged)


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
    normalized = normalized.replace("\uFFFD", "").strip()
    if not normalized:
        return ""

    collapsed = _collapse_repeated_spans(normalized, max_window=12, min_repeats=3)
    trimmed = _trim_repeated_tail_words(collapsed)
    if not trimmed:
        return ""
    if _is_low_information_text(trimmed):
        return ""
    return _trim_fragment_tail(trimmed)


def _sanitize_document_text(value: str) -> str:
    normalized = _normalize_spaces(value)
    normalized = normalized.replace("\uFFFD", "").strip()
    if not normalized:
        return ""
    collapsed = _collapse_repeated_spans(normalized, max_window=20, min_repeats=3)
    trimmed = _trim_repeated_tail_words(collapsed)
    return _normalize_spaces(_trim_fragment_tail(trimmed))


def _filter_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    filtered: list[TranscriptSegment] = []
    for segment in segments:
        cleaned = _sanitize_text(segment.text)
        if not cleaned:
            continue

        current = TranscriptSegment(
            start=segment.start,
            end=segment.end,
            text=cleaned,
            chunk_index=segment.chunk_index,
        )
        if filtered:
            previous = filtered[-1]
            same_text = _canonical_text(previous.text) == _canonical_text(current.text)
            close_gap = (current.start - previous.end) <= 4.0
            if same_text and close_gap:
                filtered[-1] = TranscriptSegment(
                    start=previous.start,
                    end=max(previous.end, current.end),
                    text=previous.text,
                    chunk_index=previous.chunk_index,
                )
                continue

        filtered.append(current)
    return filtered


def _collapse_repeated_spans(text: str, *, max_window: int, min_repeats: int) -> str:
    words = [word for word in text.split(" ") if word]
    if len(words) < (min_repeats * 2):
        return text

    collapsed_words = _collapse_repeated_word_spans(
        words,
        max_window=max_window,
        min_repeats=min_repeats,
    )
    return " ".join(collapsed_words)


def _collapse_repeated_word_spans(
    words: list[str],
    *,
    max_window: int,
    min_repeats: int,
) -> list[str]:
    normalized_words = [_word_key(word) for word in words]
    output: list[str] = []
    index = 0

    while index < len(words):
        collapsed = False
        max_window_here = min(max_window, (len(words) - index) // min_repeats)

        for window in range(1, max_window_here + 1):
            pattern = normalized_words[index : index + window]
            if not pattern or all(not token for token in pattern):
                continue

            repeats = _count_consecutive_span_repeats(
                normalized_words,
                start=index,
                window=window,
            )
            if repeats < min_repeats:
                continue

            output.extend(words[index : index + window])
            index += window * repeats
            collapsed = True
            break

        if collapsed:
            continue
        output.append(words[index])
        index += 1

    return output


def _count_consecutive_span_repeats(words: list[str], *, start: int, window: int) -> int:
    if window <= 0:
        return 1
    if start + window > len(words):
        return 1

    pattern = words[start : start + window]
    repeats = 1
    cursor = start + window
    while cursor + window <= len(words):
        candidate = words[cursor : cursor + window]
        if candidate != pattern:
            break
        repeats += 1
        cursor += window
    return repeats


def _is_low_information_text(text: str) -> bool:
    raw_words = [token for token in text.split(" ") if token]
    if len(raw_words) < 12:
        return False

    words = [_word_key(word) for word in raw_words]
    words = [word for word in words if word]
    if len(words) < 12:
        return False

    unique_ratio = len(set(words)) / float(len(words))
    if unique_ratio < 0.25:
        return True

    most_common = Counter(words).most_common(1)
    if most_common:
        most_common_ratio = most_common[0][1] / float(len(words))
        if len(words) >= 12 and most_common_ratio > 0.34:
            return True

    longest_same_word_run = _longest_consecutive_word_run(words)
    if longest_same_word_run >= 6:
        return True

    max_span_repeats = 1
    max_window = min(8, len(words) // 2)
    for index in range(len(words)):
        for window in range(1, max_window + 1):
            repeats = _count_consecutive_span_repeats(words, start=index, window=window)
            if repeats > max_span_repeats:
                max_span_repeats = repeats
    if max_span_repeats >= 4:
        return True

    return False


def _longest_consecutive_word_run(words: list[str]) -> int:
    if not words:
        return 0

    longest = 1
    current = 1
    for index in range(1, len(words)):
        if words[index] == words[index - 1]:
            current += 1
            if current > longest:
                longest = current
            continue
        current = 1
    return longest


def _word_key(value: str) -> str:
    lowered = value.lower()
    return re.sub(r"[^\w]+", "", lowered, flags=re.UNICODE)


def _trim_fragment_tail(text: str) -> str:
    words = [word for word in text.split(" ") if word]
    if len(words) < 2:
        return text

    last_key = _word_key(words[-1])
    prev_key = _word_key(words[-2])
    if not last_key:
        words = words[:-1]
    elif len(last_key) <= 1 and prev_key.startswith(last_key):
        words = words[:-1]

    if len(words) >= 3:
        last_key = _word_key(words[-1])
        ends_cleanly = text.rstrip().endswith((".", "?", "!", "\"", "'"))
        if last_key and len(last_key) <= 1 and not ends_cleanly:
            words = words[:-1]
            return " ".join(words)
        if last_key and not ends_cleanly:
            keys = [_word_key(word) for word in words]
            key_counts = Counter(key for key in keys if key)
            if key_counts.get(last_key, 0) >= 2 and len(last_key) <= 8:
                words = words[:-1]

    return " ".join(words)


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
