from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AudioInfo:
    duration_seconds: float
    sample_rate_hz: int | None = None
    channels: int | None = None
    bitrate_kbps: float | None = None


@dataclass(frozen=True)
class ChunkSpec:
    index: int
    extract_start: float
    extract_end: float
    logical_start: float
    logical_end: float

    @property
    def extract_duration(self) -> float:
        return max(0.0, self.extract_end - self.extract_start)


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    chunk_index: int


@dataclass
class PipelineConfig:
    input_path: Path
    output_dir: Path
    provider: str
    model: str
    formats: set[str] = field(default_factory=lambda: {"txt", "srt", "json"})
    language: str | None = None
    prompt: str | None = None
    chunk_seconds: int = 900
    overlap_seconds: int = 8
    keep_temp_chunks: bool = False
    resume: bool = True
    api_base: str | None = None
    timeout_seconds: int = 300
    max_retries: int = 6


@dataclass(frozen=True)
class RunSummary:
    input_path: Path
    output_dir: Path
    text_path: Path | None
    srt_path: Path | None
    json_path: Path | None
    chunk_count: int
    duration_seconds: float
