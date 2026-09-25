"""Deciding where one chunk ends and the next begins.

A long recording used to be one monolithic run: it lived entirely in memory and, if
a late stage failed, every hour already spent was lost. A real case reached 100%
transcription after 2h24m, then alignment aborted and nothing had been written.

Splitting the work makes it bounded in memory, persistable as soon as it exists, and
resumable, because a finished chunk is never recomputed.

This module is pure: durations and tuples in, chunks out. No disk, no subprocess, no
clock, which is why the awkward cases below can be tested exhaustively.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# Two boundaries closer than this would produce a chunk too small to be worth the
# process it costs to transcribe.
DEFAULT_MIN_CHUNK_SECONDS = 20.0

# A loudness scan is only informative with at least this much range. Heavily limited
# audio is flat to within a decibel or two, and choosing the "calmest" window there
# is picking noise; overlapping chunks protect a straddling word far better.
MIN_LOUDNESS_SPREAD_DB = 3.0

# How far a boundary may move from its target to land on a candidate. Wide enough to
# find a pause, narrow enough that chunk lengths stay even.
BOUNDARY_TOLERANCE = 0.35


@dataclass(frozen=True)
class Chunk:
    index: int
    start: float
    length: float

    @property
    def end(self) -> float:
        return self.start + self.length


def loudness_spread(samples: Sequence[tuple[float, float]]) -> float | None:
    """Robust loudness range of a scan, in dB, from the 5th to the 95th percentile.

    Percentiles rather than min/max, so one stray window cannot make flat audio look
    dynamic. Returns None when there are too few measurements to judge.
    """
    if len(samples) < 4:
        return None
    levels = sorted(sample[1] for sample in samples)
    return levels[len(levels) * 19 // 20] - levels[len(levels) // 20]


def quiet_cut_points(
    duration: float,
    samples: Sequence[tuple[float, float]],
    *,
    chunk_seconds: float,
    min_chunk_seconds: float = DEFAULT_MIN_CHUNK_SECONDS,
    search_seconds: float | None = None,
    min_spread_db: float = MIN_LOUDNESS_SPREAD_DB,
) -> list[float]:
    """The calmest instant near each target boundary.

    Unlike silence detection this always finds *something*, because every recording
    has loudness minima even when it has no silence at all. Those minima only mean
    something when the recording has dynamic range, so an uninformative scan returns
    nothing and the caller keeps overlapping chunks instead.

    Results are kept apart so no chunk is stunted and the tail is never orphaned.
    """
    if not samples or chunk_seconds <= 0 or duration <= 0:
        return []
    spread = loudness_spread(samples)
    if spread is not None and spread < min_spread_db:
        return []

    search = search_seconds if search_seconds is not None else max(3.0, chunk_seconds * 0.25)
    spacing = max(min_chunk_seconds, chunk_seconds * 0.5)
    ordered = sorted(samples, key=lambda sample: sample[0])
    times = [sample[0] for sample in ordered]
    levels = [sample[1] for sample in ordered]

    cuts: list[float] = []
    target = chunk_seconds
    # Included at equality: a target leaving exactly min_chunk_seconds behind is still
    # a legal last boundary, and dropping it left an oversized tail chunk.
    while target <= duration - min_chunk_seconds:
        low = bisect_left(times, target - search)
        high = bisect_right(times, target + search)
        if high <= low:
            # Nothing close enough, so take the nearest measurement rather than
            # skipping the boundary and letting one chunk grow unbounded.
            nearest = min(range(len(times)), key=lambda index: abs(times[index] - target))
            low, high = nearest, nearest + 1
        best = min(
            range(low, high), key=lambda index: (levels[index], abs(times[index] - target))
        )
        cut = round(times[best], 3)
        # Measured against the previous boundary, with the recording start counting as
        # one: otherwise a lone early measurement yields a stunted first chunk.
        previous = cuts[-1] if cuts else 0.0
        if cut > 0.0 and cut - previous >= spacing and duration - cut >= min_chunk_seconds:
            cuts.append(cut)
        target += chunk_seconds
    return cuts


def _even_chunks(
    duration: float,
    *,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[Chunk]:
    """Fixed intervals with overlap, used when no usable boundary exists.

    The overlap is the protection here: a word straddling a boundary is cut in one
    chunk but whole in the next, so the text survives even without a pause to cut in.
    """
    overlap = max(0.0, min(overlap_seconds, chunk_seconds / 2.0))
    step = chunk_seconds - overlap
    chunks: list[Chunk] = []
    start = 0.0
    index = 0
    while start < duration - 1e-6:
        length = min(chunk_seconds, duration - start)
        chunks.append(Chunk(index, start, length))
        index += 1
        start += step

    if len(chunks) > 1 and chunks[-1].length < overlap:
        # A tail shorter than the overlap carries no unique audio, so fold it in
        # rather than paying a whole process for it.
        chunks.pop()
        previous = chunks[-1]
        chunks[-1] = Chunk(previous.index, previous.start, duration - previous.start)
    return chunks


def _aligned_chunks(
    duration: float,
    *,
    chunk_seconds: float,
    cut_points: Sequence[float],
    min_chunk_seconds: float,
) -> list[Chunk]:
    """Cut at the candidate nearest each target boundary.

    Cutting in a pause means no word is split across two chunks, which is why this
    mode needs no overlap. Returns an empty list unless at least one boundary came
    from a real candidate: a plan made only of evenly spaced targets gains nothing
    here and would give up the overlap that protects a straddling word.
    """
    candidates = sorted({round(point, 3) for point in cut_points if 0.0 < point < duration})
    minimum = max(min_chunk_seconds, chunk_seconds * 0.25)
    tolerance = max(2.0, chunk_seconds * BOUNDARY_TOLERANCE)

    bounds = [0.0]
    used_cut = False
    while duration - bounds[-1] > chunk_seconds:
        target = bounds[-1] + chunk_seconds
        window = [
            candidate
            for candidate in candidates
            if abs(candidate - target) <= tolerance
            and candidate - bounds[-1] >= minimum
            and duration - candidate >= 0.0
        ]
        if window:
            cut = min(window, key=lambda candidate: abs(candidate - target))
            used_cut = True
        else:
            cut = target
        if cut - bounds[-1] < minimum:
            break
        bounds.append(min(cut, duration))

    if not used_cut or len(bounds) < 2:
        return []
    if duration - bounds[-1] < minimum:
        bounds.pop()
        if len(bounds) < 2:
            return []
    bounds.append(duration)
    return [
        Chunk(index, bounds[index], bounds[index + 1] - bounds[index])
        for index in range(len(bounds) - 1)
    ]


def plan_chunks(
    duration: float,
    *,
    chunk_seconds: float,
    overlap_seconds: float,
    cut_points: Iterable[float] = (),
    min_chunk_seconds: float = DEFAULT_MIN_CHUNK_SECONDS,
) -> list[Chunk]:
    """Split a duration into chunks.

    With ``cut_points`` each boundary snaps to the nearest candidate within a
    tolerance, so no word is split and no overlap is required. Candidates come from
    detected silence, or from the loudness scan when a recording has no silence. When
    no candidate is close enough the behaviour falls back to evenly spaced
    overlapping chunks, which is what protects a word that straddles a cut.
    """
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if duration <= chunk_seconds:
        return [Chunk(0, 0.0, duration)]

    points = list(cut_points)
    if points:
        aligned = _aligned_chunks(
            duration,
            chunk_seconds=chunk_seconds,
            cut_points=points,
            min_chunk_seconds=min_chunk_seconds,
        )
        if aligned:
            return aligned
    return _even_chunks(duration, chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds)
