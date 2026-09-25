"""Caching chunk results on disk: fingerprints, atomic writes, and readers.

Two rules keep the cache honest. A finished chunk is never recomputed, and a change
to anything that affects the text invalidates exactly the affected chunks -- which
requires fingerprinting the things that move, not just counting them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from whisperx_local.chunking.plan import Chunk


# What a recogniser leaves behind for one chunk, and therefore what has to be thrown
# away when its settings change. The sliced audio is deliberately absent: it does not
# depend on how recognition was configured.
RECOGNITION_SUFFIXES = (".asr.json", ".aligned.json")


def plan_signature(
    *,
    source: Path,
    duration: float,
    chunk_seconds: float,
    overlap_seconds: float,
    cut_points: Iterable[float] = (),
) -> dict:
    """Fingerprint the slicing inputs so stale cached chunks are discarded.

    The boundaries themselves are fingerprinted, not just how many there were:
    changing the silence threshold used to move every cut while keeping the count
    identical, which silently reused chunks cut on the old boundaries.
    """
    stat = source.stat()
    points = [f"{point:.3f}" for point in cut_points]
    return {
        "source": str(source),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "duration": round(duration, 3),
        "chunk_seconds": chunk_seconds,
        "overlap_seconds": overlap_seconds,
        "cut_points": len(points),
        "cuts": hashlib.sha256(",".join(points).encode()).hexdigest()[:12],
    }


def atomic_write_json(path: Path, payload: object) -> None:
    """Write JSON through a temporary file so a crash cannot corrupt the cache.

    A half-written chunk file would be indistinguishable from a finished one on the
    next run, which turns a crash into permanently wrong output.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def read_segments(path: Path) -> list[dict]:
    """Read either ``{"segments": [...]}`` or a bare list, as WhisperX emits both."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload.get("segments") or []
    return payload or []


def discard_recognition(chunk_dir: Path) -> int:
    """Delete cached recognised and aligned text, keeping the sliced audio.

    Separate from throwing the whole cache away because the two invalidation reasons are
    different. Moving a boundary makes the sliced audio itself wrong, so everything has
    to go. Changing how the recogniser is configured leaves the audio perfectly good and
    only makes the text read from it stale, so re-slicing would be wasted work.

    Returns how many files were removed, so the caller can report it rather than
    quietly redoing minutes of work with no explanation.
    """
    if not chunk_dir.is_dir():
        return 0
    removed = 0
    for path in sorted(chunk_dir.iterdir()):
        if path.is_file() and path.name.endswith(RECOGNITION_SUFFIXES):
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def chunk_paths(chunk_dir: Path, chunk: Chunk) -> dict[str, Path]:
    """The three files belonging to one chunk, in the order they are produced."""
    stem = f"part-{chunk.index:04d}"
    return {
        "audio": chunk_dir / f"{stem}.wav",
        "recognised": chunk_dir / f"{stem}.asr.json",
        "aligned": chunk_dir / f"{stem}.aligned.json",
    }


def completed_indices(chunks: Sequence[Chunk], chunk_dir: Path) -> set[int]:
    """Chunks whose alignment output already exists, so they need no recompute."""
    done: set[int] = set()
    for chunk in chunks:
        if chunk_paths(chunk_dir, chunk)["aligned"].is_file():
            done.add(chunk.index)
    return done
