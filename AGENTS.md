# Ultimate Transcripter SSOT

This file is the single source of truth (SSOT) for architecture, behavior, and operating rules of this repository.
No parallel product docs should be created.

## Product Goal

Build a high quality audio transcription CLI that:
- Uses cloud API models for quality-first transcription.
- Handles very long and very large audio files reliably.
- Uses BYOK (bring your own key) at runtime, without storing keys in `.env` files.

## Explicit Scope

- Input formats: common audio containers such as `.m4a`, `.mp3`, `.wav`, and others supported by ffmpeg.
- Output formats: `txt`, `srt`, and `json`.
- API provider: OpenAI audio transcription endpoint via HTTP.
- Long file support: required (chunking + overlap + resume support).

## Explicit Non-Goals (for now)

- Speaker diarization is out of scope.
- Web UI is out of scope.
- Multi-provider orchestration is out of scope (architecture should still stay provider-extensible).

## Runtime Requirements

- Python 3.10+
- `ffmpeg` and `ffprobe` installed and available on PATH.

Reason:
- `ffprobe` is used for audio duration metadata.
- `ffmpeg` is used for deterministic chunk extraction and re-encoding.

## Security and BYOK Policy

- The tool asks for API key interactively on each run if not provided by CLI argument.
- The key must not be written to disk by this project.
- The key must not be persisted in `.env` files by this project.
- The key is only used in-memory for request authorization.

## CLI Contract

Entry command:
- `ultimate-transcripter transcribe <audio_file> [options]`

Core behavior:
- If key is omitted, prompt user in hidden input mode.
- Validate prerequisites (`ffmpeg`, `ffprobe`, input file).
- Build chunk plan for full file.
- Transcribe each chunk with retries.
- Merge chunk segments into one timeline.
- Emit requested output files.

Recommended defaults:
- model: `gpt-4o-transcribe`
- chunk duration: `900` seconds
- overlap: `8` seconds
- outputs: `txt,srt,json`

## Processing Pipeline

1. Probe input media with `ffprobe` for duration and stream details.
2. Plan chunks over full duration with overlap windows.
3. Extract each chunk using `ffmpeg` to normalized mono/16k/aac for predictable upload size.
4. Send chunk to provider endpoint with `verbose_json` and segment timestamps.
5. Persist each chunk result in state files for resume safety.
6. Merge segments into absolute timeline and remove overlap duplicates.
7. Write final outputs (`.txt`, `.srt`, `.json`).

## Resume and Reliability Rules

- Chunk results are checkpointed in `.state/chunk_XXXXX.json` under output dir.
- On rerun, completed chunk files are reused unless resume is disabled.
- Transcription API calls use bounded retries with exponential backoff.
- Fail fast with actionable errors for missing dependencies or API failures.

## Output Layout

Given input `meeting.m4a`, default output directory is `meeting_transcript/` and contains:
- `meeting.txt`
- `meeting.srt`
- `meeting.json`
- `.state/` checkpoint folder
- `.chunks/` temporary chunk files (optional cleanup depending on runtime flags)

## Code Layout

- `src/ultimate_transcripter/cli.py`: CLI parsing, BYOK prompt, execution entrypoint.
- `src/ultimate_transcripter/audio.py`: ffprobe/ffmpeg integration, chunk planning, SRT timestamp helpers.
- `src/ultimate_transcripter/provider_openai.py`: OpenAI HTTP client, retries, response validation.
- `src/ultimate_transcripter/pipeline.py`: orchestration, resume checkpoints, merge/output logic.
- `src/ultimate_transcripter/types.py`: core dataclasses for chunks, segments, and outputs.

## Engineering Rules

- Keep code ASCII unless format requires otherwise.
- Prefer deterministic behavior over magic heuristics.
- Keep log messages concise and operational.
- Do not silently drop failures.
- Do not persist secrets.

## Verification

Minimum validation after code changes:
- CLI help runs.
- Module import works.
- Python compile check passes.

Optional runtime validation (requires key and sample audio):
- End-to-end transcription on short file.
- Resume behavior by interrupting and rerunning.
