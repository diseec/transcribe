"""Running each stage, with the retries that make a long run survivable.

Alignment and diarization are child processes, not library calls. A native abort
raises SIGABRT, which cannot be caught in Python, so an in-process stage could never
be retried -- and the one time that happened it discarded two and a half hours of
work. As child processes they are just exit codes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from whisperx_local.chunking import atomic_write_json, read_segments
from whisperx_local.engine import backends
from whisperx_local.engine.commands import managed_worker_command
from whisperx_local.paths import PYTHON
from whisperx_local.transcript import Turn, clean_turns

# A retry waits longer each time, so a transient memory spike has a chance to clear.
RETRY_BACKOFF_SECONDS = 5.0
RETRY_BACKOFF_LIMIT = 30.0


def transcribe_chunk(options, studio, slice_path, chunk_dir, index, network, position, total):
    """Recognise one chunk, retrying with reduced thread pressure on failure.

    Fewer threads lowers peak memory, which is the usual cause of these aborts, so
    each retry is gentler than the last rather than identical.
    """
    output_dir = chunk_dir / f"asr-{index:04d}"
    backend = backends.for_options(options)
    produced = backend.output_path(slice_path, output_dir)
    for attempt in range(1, max(1, options.retries) + 1):
        if attempt > 1:
            studio.set_status(f"Chunk {position}/{total} · retry {attempt - 1}")
            studio.log_line(f"chunk {index}: retry attempt {attempt}\n")
            time.sleep(min(RETRY_BACKOFF_LIMIT, RETRY_BACKOFF_SECONDS * attempt))
        threads = max(1, options.threads // attempt)
        command, env = backend.command(options, slice_path, output_dir, network, threads)
        studio.log_line("$ " + " ".join(command) + "\n")
        code = studio.stream(command, env=env, check=False)
        if code == 0 and produced.is_file():
            return backend.read(produced)
        studio.log_line(f"chunk {index}: recognition exited {code}\n")
    return None


def align_chunk(options, studio, asr_path, slice_path, chunk_dir, index, network, position, total):
    """Align one chunk in a child process so a native abort stays recoverable."""
    predicted = chunk_dir / f"part-{index:04d}.aligned.json"
    command = managed_worker_command(
        "align",
        "--segments", str(asr_path),
        "--audio", str(slice_path),
        "--language", options.language,
        "--output", str(predicted),
        network=network,
    )
    env = _worker_env(network)
    for attempt in range(1, max(1, options.align_retries) + 1):
        studio.set_status(f"Chunk {position}/{total} · aligning this part")
        code = studio.stream(command, env=env, check=False)
        if code == 0 and predicted.is_file():
            return read_segments(predicted)
        studio.log_line(f"chunk {index}: alignment exited {code}\n")
        predicted.unlink(missing_ok=True)
        if attempt < max(1, options.align_retries):
            time.sleep(min(20.0, 4.0 * attempt))
    return None


def diarize_voices(options, studio, source: Path, work_dir: Path, network: str) -> list[Turn]:
    """Detect speakers in a child process, returning cleaned turns."""
    label = options.speakers if options.speakers is not None else "auto"
    turns_path = work_dir / f"turns-{label}.json"
    extra = (
        ["--min-speakers", str(options.speakers), "--max-speakers", str(options.speakers)]
        if options.speakers is not None
        else []
    )
    command = managed_worker_command(
        "diarize",
        "--audio", str(source),
        "--output", str(turns_path),
        *extra,
        network=network,
    )
    env = _worker_env(network)
    for attempt in range(1, max(1, options.diarize_retries) + 1):
        if turns_path.is_file():
            break
        suffix = f" · retry {attempt - 1}" if attempt > 1 else ""
        studio.set_status(f"Separating voices{suffix}")
        code = studio.stream(command, env=env, check=False)
        if code == 0 and turns_path.is_file():
            break
        studio.log_line(f"diarization exited {code}\n")
        if attempt < max(1, options.diarize_retries):
            time.sleep(min(20.0, 4.0 * attempt))

    if not turns_path.is_file():
        # Speaker labels are a bonus; the transcript is not.
        studio.warn("Voice labels were unavailable; the transcript was written without them.")
        return []

    payload = json.loads(turns_path.read_text(encoding="utf-8"))
    raw = [
        Turn(float(row["start"]), float(row["end"]), str(row["speaker"]))
        for row in payload.get("turns") or []
    ]
    return clean_turns(raw, min_duration=options.min_turn)


def _worker_env(network: str) -> dict[str, str]:
    from whisperx_local.engine.commands import offline_environment

    return offline_environment(network)


def write_turns(path: Path, turns: list[Turn]) -> None:
    """Persist cleaned turns so a resume does not repeat the slowest stage."""
    atomic_write_json(
        path,
        {"turns": [{"start": t.start, "end": t.end, "speaker": t.speaker} for t in turns]},
    )


def read_turns(path: Path) -> list[Turn]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        Turn(float(row["start"]), float(row["end"]), str(row["speaker"]))
        for row in payload.get("turns") or []
    ]


__all__ = [
    "PYTHON",
    "align_chunk",
    "diarize_voices",
    "read_turns",
    "transcribe_chunk",
    "write_turns",
]
