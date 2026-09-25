"""Job identity and the completion marker.

A finished run is skipped only when the settings that produced it are unchanged. The
marker is deliberately withheld when part of the work failed, so a rerun finishes the
remainder instead of redoing everything.
"""

from __future__ import annotations

import json
from pathlib import Path

from whisperx_local.engine import recognition_signature
from whisperx_local.paths import MODELS_DIR, STATE_DIR
from whisperx_local.services.artifacts import identity

# Bumped when a change alters what a run produces, so older markers cannot suppress it.
PIPELINE_VERSION = "actions-v1"

FINGERPRINTED_OPTIONS = (
    "language", "speakers", "output_format", "min_turn", "hotwords", "prompt",
    "diarize_audio", "silence_split", "silence_min", "chunk_seconds", "chunk_overlap",
    "profile", "diarize", "output_dir",
)


def fingerprint(source: Path, options) -> dict[str, object]:
    """Everything that changes the output, used to decide whether to skip a run."""
    stat = source.stat()
    recorded = {name: getattr(options, name, None) for name in FINGERPRINTED_OPTIONS}
    return {
        "source": str(source),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "pipeline": PIPELINE_VERSION,
        **recorded,
        **recognition_signature(options),
    }


def marker_path(source: Path) -> Path:
    return STATE_DIR / f"{source.stem}-{identity(source)}.json"


def is_complete(source: Path, options, output: Path) -> bool:
    """Whether an identical run already finished and produced the expected file."""
    if not output.is_file():
        return False
    marker = marker_path(source)
    if not marker.is_file():
        return False
    try:
        return json.loads(marker.read_text(encoding="utf-8")) == fingerprint(source, options)
    except (OSError, json.JSONDecodeError):
        return False


def mark_complete(source: Path, options) -> None:
    marker = marker_path(source)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(fingerprint(source, options), indent=2, sort_keys=True) + "\n"
    marker.write_text(payload, encoding="utf-8")


def mark_diarization_ready() -> None:
    """Record that speaker models are cached, so later runs may work offline."""
    (MODELS_DIR / ".diarization-ready").touch()


def diarization_ready() -> bool:
    return (MODELS_DIR / ".diarization-ready").exists()


def _cache_glob(model: str) -> str:
    """The cache directory a model lives in, as a glob pattern.

    ``--model`` is either a size, cached under an owner directory
    (``large-v3`` -> ``models--Systran--faster-whisper-large-v3``), or a full repository
    id (``owner/name`` -> ``models--owner--name``). Only the first form was understood,
    so a repository id -- which is how a turbo model is named -- did not look cached at
    all.
    """
    if "/" in model:
        owner, _, name = model.partition("/")
        return f"models--{owner}--{name}"
    return f"models--*--faster-whisper-{model.removeprefix('faster-whisper-')}"


def _marker_stem(model: str) -> str:
    """A filesystem-safe name for a cache marker.

    A repository id contains a slash, so ``.core-ready-<model>`` was a *path* whose
    parent directory did not exist and ``touch()`` raised ``FileNotFoundError``. It
    happened after the transcript had been written, so a run that had fully succeeded
    ended in a traceback and never printed what it had produced.
    """
    return model.removeprefix("faster-whisper-").replace("/", "--")


def core_models_cached(model: str, language: str) -> bool:
    """Whether the models a run would need are already on disk.

    Used to choose between cache-only and downloading before anything expensive
    starts, rather than discovering a missing model an hour in.
    """
    if not any(MODELS_DIR.glob(f"{_cache_glob(model)}/snapshots/*")):
        return False
    if language == "fa":
        return any(
            MODELS_DIR.glob("models--jonatasgrosman--wav2vec2-large-xlsr-53-persian/snapshots/*")
        )
    return (MODELS_DIR / f".core-ready-{_marker_stem(model)}-{language}").exists()


def mark_core_ready(model: str, language: str) -> None:
    (MODELS_DIR / f".core-ready-{_marker_stem(model)}-{language}").touch()
