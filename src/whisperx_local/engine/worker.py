"""Child-process entry points for the stages that can abort natively.

Run as ``python -m whisperx_local.engine.worker <stage> ...``. Each subcommand writes
exactly one JSON file and exits.

The process boundary is the whole point. WhisperX alignment once aborted with a
``libc++abi`` exception that Python cannot catch, discarding two and a half hours of
work. As a child process that abort is just an exit code, so the caller can retry it,
retry it with fewer threads, or carry on without it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from whisperx_local.chunking import atomic_write_json
from whisperx_local.transcript import Turn, diarize_turns


class PercentReporter:
    """Print whole-number progress, because the parent parses these lines.

    Reporting only on change keeps a long stage from flooding the parent with
    thousands of identical lines that each trigger a repaint.
    """

    def __init__(self) -> None:
        self._last = -1

    def __call__(self, percent: float) -> None:
        value = int(percent)
        if value == self._last:
            return
        self._last = value
        print(f"Progress: {value}%...", flush=True)


def align_segments(args: argparse.Namespace) -> int:
    """Refine segment timings to the word level and write them out."""
    import whisperx

    from whisperx.alignment import load_align_model

    payload = json.loads(Path(args.segments).read_text(encoding="utf-8"))
    segments = payload.get("segments") if isinstance(payload, dict) else payload
    if not segments:
        atomic_write_json(Path(args.output), {"segments": []})
        return 0

    audio = whisperx.load_audio(str(args.audio))
    model, metadata = load_align_model(
        language_code=args.language,
        device=args.device,
        model_dir=str(args.model_dir),
    )
    result = whisperx.align(
        segments,
        model,
        metadata,
        audio,
        args.device,
        return_char_alignments=False,
        print_progress=True,
    )
    atomic_write_json(Path(args.output), {"segments": result.get("segments") or []})
    return 0


def separate_speakers(args: argparse.Namespace) -> int:
    """Run diarization and write cleaned turns."""
    turns = diarize_turns(
        Path(args.audio),
        token=os.environ.get("HF_TOKEN", ""),
        cache_dir=Path(args.model_dir),
        device=args.device,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        progress=PercentReporter(),
    )
    atomic_write_json(
        Path(args.output),
        {
            "turns": [
                {"start": turn.start, "end": turn.end, "speaker": turn.speaker}
                for turn in turns
            ]
        },
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Defined here as well as in the app: this module runs as its own program."""
    parser = argparse.ArgumentParser(prog="whisperx_local.engine.worker")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    align = subparsers.add_parser("align", help="Refine word timings for one chunk")
    align.add_argument("--segments", required=True)
    align.add_argument("--audio", required=True)
    align.add_argument("--language", required=True)
    align.add_argument("--output", required=True)
    align.add_argument("--model-dir", required=True)
    align.add_argument("--device", default="cpu")
    align.add_argument("--cache-only", action="store_true")
    align.set_defaults(handler=align_segments)

    diarize = subparsers.add_parser("diarize", help="Separate speakers for one file")
    diarize.add_argument("--audio", required=True)
    diarize.add_argument("--output", required=True)
    diarize.add_argument("--model-dir", required=True)
    diarize.add_argument("--device", default="cpu")
    diarize.add_argument("--min-speakers", type=int, default=None)
    diarize.add_argument("--max-speakers", type=int, default=None)
    diarize.add_argument("--cache-only", action="store_true")
    diarize.set_defaults(handler=separate_speakers)

    return parser


def main(arguments: list[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    if getattr(args, "cache_only", False):
        # Set before the libraries are imported, because transformers reads these at
        # import time and ignores them afterwards.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
