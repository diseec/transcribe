"""What a run actually produced, in enough detail to tell full from partial.

This exists because of a specific failure. A 52-minute recording once produced a
transcript covering about 42% of the audio, ended with a success message, and looked
complete: 143 tidy lines with speaker labels. The only reason anyone noticed was that
an earlier crashed run had salvaged 9,007 words against its 3,006.

The cause was that "the run ended" and "the work was done" were treated as the same
thing. They are not. A chunk that fails, or succeeds but recognises nothing, leaves a
hole, and a hole is only acceptable if it is named. So every run now states what it
covered, which stages completed, and whether the result is partial -- and a partial
result is never reported as success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

# Below this share of unaccounted-for audio, a result is called partial even when every
# chunk reported success. This is the safety net for the unknown failure: when nothing
# raises and nothing is logged, the only remaining evidence is a recording whose text
# does not reach its end.
COVERAGE_FLOOR = 0.90


def union_seconds(intervals: Iterable[tuple[float, float]]) -> float:
    """Total time covered by these intervals, counting any overlap once.

    Chunks overlap deliberately, so summing their lengths would claim more audio than
    exists. Merging the intervals gives a figure that can be compared with the
    recording itself.
    """
    ordered = sorted((float(start), float(end)) for start, end in intervals if end > start)
    total = 0.0
    open_start: float | None = None
    open_end: float | None = None
    for start, end in ordered:
        if open_end is None or open_start is None:
            open_start, open_end = start, end
        elif start > open_end:
            total += open_end - open_start
            open_start, open_end = start, end
        else:
            open_end = max(open_end, end)
    if open_start is not None and open_end is not None:
        total += open_end - open_start
    return total


@dataclass(frozen=True)
class Gap:
    """A stretch of audio that produced no text, and why."""

    index: int
    start: float
    end: float
    reason: str

    def describe(self) -> str:
        return f"{self.start:.0f}s–{self.end:.0f}s ({self.reason})"


@dataclass
class RunReport:
    """The measured outcome of one run, including its holes."""

    duration: float
    segments: int
    words: int
    chunks: int
    covered: float = 0.0
    silent_seconds: float = 0.0
    speakers: int = 0
    gaps: list[Gap] = field(default_factory=list)
    alignment_requested: bool = False
    alignment_complete: bool = True
    diarization_requested: bool = False
    diarization_complete: bool = True
    canonical_files: list[str] = field(default_factory=list)

    @property
    def coverage_percent(self) -> float:
        if self.duration <= 0:
            return 0.0
        return min(100.0, self.covered / self.duration * 100.0)

    @property
    def unaccounted(self) -> float:
        """Audio that is neither transcribed nor known to be silent.

        This is the figure that matters. Time with no speech in it is explained, so a
        recording that is quiet in places is not a failure; time that simply produced
        nothing is unaccounted for, and that is what a silent shortfall looks like.
        """
        return max(0.0, self.duration - self.covered - self.silent_seconds)

    @property
    def short(self) -> bool:
        """True when a meaningful stretch of audio is unaccounted for."""
        if self.duration <= 0:
            return False
        # A hair of tolerance: ``1 - 0.9`` is not exactly ``0.1`` in binary, and a
        # result sitting precisely on the floor should not become partial by rounding.
        slack = 1e-9 * max(1.0, self.duration)
        return self.unaccounted > (1.0 - COVERAGE_FLOOR) * self.duration + slack

    @property
    def partial(self) -> bool:
        """True when text is missing, a refinement failed, or the text falls short."""
        if self.gaps or self.short:
            return True
        if self.alignment_requested and not self.alignment_complete:
            return True
        if self.diarization_requested and not self.diarization_complete:
            return True
        return False

    @property
    def state(self) -> str:
        return "PARTIAL" if self.partial else "COMPLETE"

    def headline(self) -> str:
        """One line stating how much of the recording the transcript covers."""
        if self.duration <= 0:
            return "transcript written"
        return (
            f"{self.covered:.0f}s of {self.duration:.0f}s "
            f"({self.coverage_percent:.0f}%)"
        )

    def lines(self) -> list[tuple[str, str]]:
        """Label/value rows for display, most diagnostic first."""
        rows: list[tuple[str, str]] = [
            ("Recording", f"{self.duration:.0f}s across {self.chunks} chunk(s)"),
            ("Covered", self.headline()),
            ("Recognised", f"{self.segments} segments, {self.words} words"),
        ]
        if self.silent_seconds >= 1.0:
            rows.append(("No speech", f"{self.silent_seconds:.0f}s of silence"))
        if self.unaccounted >= 1.0:
            rows.append(("Unaccounted", f"{self.unaccounted:.0f}s with no text"))
        if self.diarization_requested:
            rows.append(
                ("Speaker labels", "done" if self.diarization_complete else "NOT completed")
            )
        if self.alignment_requested:
            rows.append(
                ("Word timing", "done" if self.alignment_complete else "NOT completed")
            )
        rows.append(("Result", self.state))
        if self.canonical_files:
            rows.append(("Raw copy", ", ".join(self.canonical_files)))
        return rows

    def gap_lines(self) -> list[str]:
        """The gaps, spelled out so a rerun can target them."""
        return [gap.describe() for gap in self.gaps]

    def advice(self) -> str | None:
        """What to do about a partial result, in one sentence."""
        if not self.partial:
            return None
        if any(gap.reason == "not transcribed" for gap in self.gaps):
            return (
                "Re-run the same command: finished chunks are kept, so only the missing "
                "parts are redone."
            )
        if self.short and not self.gaps:
            return (
                "Run it again with --force to redo the recognition: no text was produced "
                "for part of the recording, and no cause was recorded."
            )
        return "The rest of the text is complete; the missing stage can be run on its own."
