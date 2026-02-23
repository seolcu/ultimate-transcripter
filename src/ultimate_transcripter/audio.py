from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

from ultimate_transcripter.types import AudioInfo, ChunkSpec


def require_binary(name: str) -> None:
    if shutil.which(name):
        return
    raise RuntimeError(
        f"Required binary '{name}' was not found on PATH. Install ffmpeg/ffprobe first."
    )


def probe_audio(input_path: Path) -> AudioInfo:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,bit_rate",
        "-show_entries",
        "stream=codec_type,sample_rate,channels",
        "-of",
        "json",
        str(input_path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"ffprobe failed for {input_path}: {stderr}")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe output was not valid JSON") from exc

    format_info = payload.get("format") or {}
    duration_raw = format_info.get("duration")
    if duration_raw is None:
        raise RuntimeError("Could not read input duration from ffprobe")
    duration_seconds = float(duration_raw)
    if duration_seconds <= 0:
        raise RuntimeError("Input duration is zero or negative")

    bitrate_raw = format_info.get("bit_rate")
    bitrate_kbps = float(bitrate_raw) / 1000.0 if bitrate_raw else None

    sample_rate_hz: int | None = None
    channels: int | None = None
    for stream in payload.get("streams") or []:
        if stream.get("codec_type") != "audio":
            continue
        sr_value = stream.get("sample_rate")
        ch_value = stream.get("channels")
        if sr_value is not None and sample_rate_hz is None:
            sample_rate_hz = int(sr_value)
        if ch_value is not None and channels is None:
            channels = int(ch_value)

    return AudioInfo(
        duration_seconds=duration_seconds,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        bitrate_kbps=bitrate_kbps,
    )


def plan_chunks(
    *, duration_seconds: float, chunk_seconds: int, overlap_seconds: int
) -> list[ChunkSpec]:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be >= 0")
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be smaller than chunk_seconds")

    total_chunks = max(1, math.ceil(duration_seconds / float(chunk_seconds)))
    chunks: list[ChunkSpec] = []

    for index in range(total_chunks):
        logical_start = float(index * chunk_seconds)
        logical_end = min(duration_seconds, float((index + 1) * chunk_seconds))

        extract_start = max(0.0, logical_start - float(overlap_seconds))
        extract_end = min(duration_seconds, logical_end + float(overlap_seconds))

        chunks.append(
            ChunkSpec(
                index=index,
                extract_start=extract_start,
                extract_end=extract_end,
                logical_start=logical_start,
                logical_end=logical_end,
            )
        )
    return chunks


def extract_chunk(
    *,
    input_path: Path,
    output_path: Path,
    start_seconds: float,
    duration_seconds: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(input_path),
        "-t",
        f"{duration_seconds:.3f}",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "aac",
        "-b:a",
        "48k",
        str(output_path),
    ]

    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            f"ffmpeg failed while extracting chunk at {start_seconds:.2f}s: {stderr}"
        )


def format_srt_timestamp(seconds: float) -> str:
    clamped = max(0.0, seconds)
    millis_total = int(round(clamped * 1000.0))

    millis = millis_total % 1000
    total_seconds = millis_total // 1000
    sec = total_seconds % 60
    total_minutes = total_seconds // 60
    minute = total_minutes % 60
    hour = total_minutes // 60

    return f"{hour:02d}:{minute:02d}:{sec:02d},{millis:03d}"
