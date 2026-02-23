from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path
from typing import Sequence

from ultimate_transcripter import __version__
from ultimate_transcripter.pipeline import run_transcription
from ultimate_transcripter.types import PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ultimate-transcripter",
        description="Quality-first BYOK audio transcription for very large files.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Transcribe one audio file into txt/srt/json outputs.",
    )
    transcribe_parser.add_argument("audio_file", help="Path to input audio file")
    transcribe_parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory (default: <input_stem>_transcript)",
    )
    transcribe_parser.add_argument(
        "--formats",
        default="txt,srt,json",
        help="Comma-separated output formats: txt,srt,json (default: txt,srt,json)",
    )
    transcribe_parser.add_argument(
        "--model",
        default="gpt-4o-transcribe",
        help="Transcription model (default: gpt-4o-transcribe)",
    )
    transcribe_parser.add_argument(
        "--language",
        help="Optional language hint such as 'ko' or 'en'",
    )
    transcribe_parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=900,
        help="Chunk length in seconds (default: 900)",
    )
    transcribe_parser.add_argument(
        "--overlap-seconds",
        type=int,
        default=8,
        help="Overlap between chunks in seconds (default: 8)",
    )
    transcribe_parser.add_argument(
        "--prompt",
        help="Optional glossary/style hint passed to the model",
    )
    transcribe_parser.add_argument(
        "--api-key",
        help="API key override. If omitted, tool prompts securely every run.",
    )
    transcribe_parser.add_argument(
        "--api-base",
        default="https://api.openai.com/v1",
        help="API base URL (default: https://api.openai.com/v1)",
    )
    transcribe_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="HTTP timeout in seconds (default: 300)",
    )
    transcribe_parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Max API retries for transient failures (default: 6)",
    )
    transcribe_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint resume and re-run all chunks",
    )
    transcribe_parser.add_argument(
        "--keep-temp-chunks",
        action="store_true",
        help="Keep temporary chunk files under .chunks",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "transcribe":
        return _run_transcribe(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


def _run_transcribe(args: argparse.Namespace) -> int:
    input_path = Path(args.audio_file).expanduser()
    if not input_path.exists():
        print(f"Error: input file does not exist: {input_path}", file=sys.stderr)
        return 2
    if not input_path.is_file():
        print(f"Error: input path is not a file: {input_path}", file=sys.stderr)
        return 2

    if args.chunk_seconds <= 0:
        print("Error: --chunk-seconds must be > 0", file=sys.stderr)
        return 2
    if args.overlap_seconds < 0:
        print("Error: --overlap-seconds must be >= 0", file=sys.stderr)
        return 2
    if args.overlap_seconds >= args.chunk_seconds:
        print("Error: --overlap-seconds must be smaller than --chunk-seconds", file=sys.stderr)
        return 2

    output_dir = _resolve_output_dir(input_path=input_path, output_dir_raw=args.output_dir)

    try:
        formats = _parse_formats(args.formats)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    api_key = (args.api_key or "").strip()
    if not api_key:
        api_key = getpass.getpass("OpenAI API key (hidden, BYOK, never stored): ").strip()
    if not api_key:
        print("Error: API key is required.", file=sys.stderr)
        return 2

    config = PipelineConfig(
        input_path=input_path,
        output_dir=output_dir,
        model=args.model,
        formats=formats,
        language=args.language,
        prompt=args.prompt,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        keep_temp_chunks=bool(args.keep_temp_chunks),
        resume=not bool(args.no_resume),
        api_base=args.api_base,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )

    print(f"Input: {config.input_path}")
    print(f"Output dir: {config.output_dir}")
    print(
        f"Model: {config.model} | chunk={config.chunk_seconds}s overlap={config.overlap_seconds}s"
    )

    try:
        summary = run_transcription(config=config, api_key=api_key, logger=print)
    except KeyboardInterrupt:
        print("Interrupted by user. Rerun the same command to resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Done.")
    print(f"Chunks processed: {summary.chunk_count}")
    if summary.text_path is not None:
        print(f"TXT:  {summary.text_path}")
    if summary.srt_path is not None:
        print(f"SRT:  {summary.srt_path}")
    if summary.json_path is not None:
        print(f"JSON: {summary.json_path}")
    return 0


def _parse_formats(raw_value: str) -> set[str]:
    allowed = {"txt", "srt", "json"}
    tokens = {token.strip().lower() for token in raw_value.split(",") if token.strip()}
    if not tokens:
        raise ValueError("At least one format is required.")

    invalid = sorted(token for token in tokens if token not in allowed)
    if invalid:
        joined = ", ".join(invalid)
        raise ValueError(f"Unsupported format(s): {joined}. Allowed: txt,srt,json")
    return tokens


def _resolve_output_dir(*, input_path: Path, output_dir_raw: str | None) -> Path:
    if output_dir_raw:
        return Path(output_dir_raw).expanduser()
    return input_path.with_name(f"{input_path.stem}_transcript")


if __name__ == "__main__":
    raise SystemExit(main())
