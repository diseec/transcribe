"""Speaker turns: the model, how raw diarization is cleaned, and how a span is labelled.

Raw pyannote output on real meetings contains three speakers and many sub-100ms
flicker turns around every genuine change. Those fragments are the main reason single
words end up with the wrong speaker, so they are removed before anything is labelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

# Turns shorter than this are instability rather than a real contribution.
DEFAULT_MIN_TURN = 0.35
# Neighbouring turns of the same speaker closer than this are one turn.
DEFAULT_MERGE_GAP = 0.30
# How far a word may sit from a turn and still inherit its speaker.
DEFAULT_NEAREST_GAP = 2.5


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Line:
    speaker: str | None
    start: float
    end: float
    words: list[dict] = field(default_factory=list)

    @property
    def text(self) -> str:
        parts = [str(word.get("word", "")).strip() for word in self.words]
        return " ".join(part for part in parts if part).strip()


def diarize_turns(
    audio: Path,
    *,
    token: str,
    cache_dir: Path,
    device: str = "cpu",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    progress=None,
) -> list[Turn]:
    """Run pyannote and return cleanly sorted speaker turns.

    ``progress`` is called with a float 0-100. It is forwarded rather than wrapped
    because the caller reports it; this function is also called in a child process,
    where the callback prints a line the parent parses.
    """
    from whisperx.diarize import DiarizationPipeline

    pipeline = DiarizationPipeline(
        model_name=DIARIZATION_MODEL,
        token=token,
        device=device,
        cache_dir=str(cache_dir),
    )
    frame = pipeline(
        str(audio),
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        progress_callback=progress,
    )
    turns = [
        Turn(float(row["start"]), float(row["end"]), str(row["speaker"]))
        for _, row in frame.iterrows()
    ]
    return sorted(turns, key=lambda turn: (turn.start, turn.end))


def drop_micro_turns(
    turns: list[Turn], min_duration: float = DEFAULT_MIN_TURN
) -> list[Turn]:
    """Remove flicker turns that pyannote emits around real speaker changes."""
    return [turn for turn in turns if turn.duration >= min_duration]


def merge_adjacent(turns: list[Turn], gap: float = DEFAULT_MERGE_GAP) -> list[Turn]:
    """Join neighbouring turns that share a speaker and are nearly contiguous."""
    merged: list[Turn] = []
    for turn in turns:
        if merged and merged[-1].speaker == turn.speaker and turn.start - merged[-1].end <= gap:
            previous = merged[-1]
            merged[-1] = Turn(previous.start, max(previous.end, turn.end), previous.speaker)
        else:
            merged.append(turn)
    return merged


def clean_turns(
    turns: list[Turn],
    *,
    min_duration: float = DEFAULT_MIN_TURN,
    gap: float = DEFAULT_MERGE_GAP,
) -> list[Turn]:
    return merge_adjacent(drop_micro_turns(turns, min_duration), gap)


def _overlap(start: float, end: float, turn: Turn) -> float:
    return max(0.0, min(end, turn.end) - max(start, turn.start))


def speaker_for_span(
    start: float,
    end: float,
    turns: list[Turn],
    *,
    nearest_gap: float = DEFAULT_NEAREST_GAP,
    fallback: str | None = None,
) -> str | None:
    """Pick the speaker for a time span.

    Overlap wins. When a span falls in a pause it snaps to the nearest turn, which is
    what WhisperX's ``fill_nearest`` would do, so a short word between two turns is
    still labelled rather than left blank. Without a nearby turn at all it returns
    ``fallback``, which callers pass as the previous speaker.
    """
    if not turns:
        return fallback
    totals: dict[str, float] = {}
    for turn in turns:
        weight = _overlap(start, end, turn)
        if weight > 0.0:
            totals[turn.speaker] = totals.get(turn.speaker, 0.0) + weight
    if totals:
        return max(totals.items(), key=lambda item: item[1])[0]

    midpoint = (start + end) / 2.0
    best: str | None = None
    best_distance: float | None = None
    for turn in turns:
        if turn.start <= midpoint <= turn.end:
            return turn.speaker
        distance = min(abs(midpoint - turn.start), abs(midpoint - turn.end))
        if best_distance is None or distance < best_distance:
            best, best_distance = turn.speaker, distance
    if best_distance is not None and best_distance <= nearest_gap:
        return best
    return fallback
