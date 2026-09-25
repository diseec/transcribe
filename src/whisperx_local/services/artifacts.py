"""Where a recording's intermediate facts live, and which already exist.

Every artifact is derived work that can be rebuilt, so the app is free to keep it on
disk and equally free to discard it. Knowing what is already there is what lets a run
skip stages instead of repeating them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from whisperx_local.actions.base import AUDIO, FILES, PLAN, SEGMENTS, TURNS, WORDS
from whisperx_local.chunking import (
    Chunk,
    atomic_write_json,
    chunk_paths,
    discard_recognition,
    plan_chunks,
    read_segments,
)
from whisperx_local.media import recognition_path
from whisperx_local.paths import OUTPUT_DIR, WORK_DIR


def identity(source: Path) -> str:
    """A stable, short name for a recording, used to keep its work apart from others."""
    return hashlib.sha256(str(source).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class WorkSpace:
    """The per-recording directory holding everything derived from it."""

    source: Path
    output_dir: Path = OUTPUT_DIR

    @property
    def identity(self) -> str:
        return identity(self.source)

    @property
    def root(self) -> Path:
        return WORK_DIR / self.identity

    @property
    def chunks(self) -> Path:
        return self.root / "chunks"

    @property
    def plan_path(self) -> Path:
        return self.root / "plan.json"

    def turns_path(self, speakers: int | None) -> Path:
        """Speaker count is part of the name: a rerun asking for a different count
        must not silently reuse the previous answer."""
        return self.root / f"turns-{speakers if speakers is not None else 'auto'}.json"

    def primary_output(self, stem: str, output_format: str) -> Path:
        extension = "json" if output_format == "all" else output_format
        return self.output_dir / f"{stem}.{extension}"

    def reset(self) -> None:
        """Drop everything derived from this recording, keeping the input."""
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)

    def stored_plan(self) -> dict | None:
        """The slicing recorded by an earlier run, if there is one."""
        import json

        if not self.plan_path.is_file():
            return None
        try:
            return json.loads(self.plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def stored_chunks(self, duration: float, *, chunk_seconds: float, overlap_seconds: float) -> list[Chunk]:
        """Rebuild the chunk list, preferring the boundaries a previous run recorded.

        Falling back to the current options would silently slice differently from the
        cached chunks, so a stage run on its own would look at the wrong files.
        """
        stored = self.stored_plan()
        if not stored:
            return plan_chunks(
                duration, chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds
            )
        return plan_chunks(
            duration,
            chunk_seconds=float(stored.get("chunk_seconds", chunk_seconds)),
            overlap_seconds=float(stored.get("overlap_seconds", overlap_seconds)),
            cut_points=stored.get("boundaries") or [],
        )

    def stale_recognition(self, signature: dict) -> bool:
        """Whether the cached text came from different settings than these.

        Read-only, and separate from clearing it, so a preview can show what a real run
        would do without destroying the cache it is only describing. A dry run that
        mutates state is not a dry run.
        """
        stored = self.stored_plan()
        return stored is not None and stored.get("recognition") != signature

    def discard_stale_recognition(self, signature: dict) -> int:
        """Throw away cached text if it was produced with different settings.

        Called before anything asks what is already on disk, because the answer to that
        question is what lets the planner skip work. Clearing the cache afterwards would
        be too late: a run with a new model would find recognised segments *available*,
        skip transcription entirely, and re-render the previous engine's words.

        Only the text goes. The sliced audio does not depend on how recognition was
        configured, so re-slicing it would be wasted work.
        """
        if not self.stale_recognition(signature):
            return 0
        return discard_recognition(self.chunks)

    def remember_recognition(self, signature: dict) -> None:
        """Record which settings produced the text currently on disk.

        Written whenever text is produced, not only when the slicing plan is built, so a
        run that skipped the slicing stage still leaves an accurate record. Otherwise the
        next run would see a mismatch and throw away perfectly good work.
        """
        stored = self.stored_plan()
        if stored is None:
            return
        stored["recognition"] = signature
        atomic_write_json(self.plan_path, stored)

    def available(self, chunks: list[Chunk], *, speakers: int | None, output_format: str) -> set[str]:
        """Which facts already exist on disk and therefore need no recomputing."""
        present: set[str] = set()
        if recognition_path(self.source).is_file():
            present.add(AUDIO)

        recognised = {chunk.index for chunk in chunks if chunk_paths(self.chunks, chunk)["recognised"].is_file()}
        aligned = {chunk.index for chunk in chunks if chunk_paths(self.chunks, chunk)["aligned"].is_file()}
        indices = {chunk.index for chunk in chunks}
        if indices and recognised == indices:
            present.add(SEGMENTS)
        if indices and aligned == indices:
            # Alignment implies recognition: its input had to exist first.
            present.add(SEGMENTS)
            present.add(WORDS)
        if self.plan_path.is_file():
            present.add(PLAN)
        if self.turns_path(speakers).is_file():
            present.add(TURNS)
        if self.primary_output(self.source.stem, output_format).is_file():
            present.add(FILES)
        return present

    def load_segments(self, chunks: list[Chunk]) -> list[dict]:
        """Rejoin cached chunk results, preferring aligned output where it exists."""
        from whisperx_local.chunking import merge_segments

        payloads: dict[int, list[dict]] = {}
        for chunk in chunks:
            paths = chunk_paths(self.chunks, chunk)
            if paths["aligned"].is_file():
                payloads[chunk.index] = read_segments(paths["aligned"])
            elif paths["recognised"].is_file():
                payloads[chunk.index] = read_segments(paths["recognised"])
        if not payloads:
            return []
        duration = max(chunk.end for chunk in chunks)
        return merge_segments(chunks, payloads, duration=duration)
