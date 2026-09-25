"""Stitching chunk results back into one timeline.

The subtlety here is ownership. A chunk is only allowed to contribute segments from
the span it owns, and those spans must tile the recording exactly: a gap means words
are silently dropped, and an overlap means they are said twice. Deriving ownership
from the plan rather than from a fixed subtraction is what makes that true for both
plan shapes.
"""

from __future__ import annotations

from collections.abc import Sequence

from whisperx_local.chunking.plan import Chunk


def owned_ranges(chunks: Sequence[Chunk], duration: float) -> dict[int, tuple[float, float]]:
    """The time span each chunk is responsible for in the merged transcript.

    A chunk owns its audio from its own start up to wherever the *next* chunk begins.
    Boundaries snapped into silence are contiguous, so subtracting a fixed overlap
    there opened a gap that no chunk owned and every word inside it was dropped. With
    overlapping fixed intervals the same rule hands the shared audio to the later
    chunk, so nothing is counted twice.
    """
    ownership: dict[int, tuple[float, float]] = {}
    for index, chunk in enumerate(chunks):
        following = chunks[index + 1].start if index + 1 < len(chunks) else duration
        ownership[chunk.index] = (chunk.start, max(chunk.start, following))
    return ownership


def shift_segments(segments: list[dict], offset: float) -> list[dict]:
    """Move chunk-local timings onto the recording's own timeline.

    Words are shifted too, not just the segment, because the speaker assignment and
    the transcript both use word timings.
    """
    shifted: list[dict] = []
    for segment in segments:
        clone = dict(segment)
        clone["start"] = float(segment.get("start", 0.0)) + offset
        clone["end"] = float(segment.get("end", 0.0)) + offset
        if segment.get("words") is not None:
            words = []
            for word in segment["words"] or []:
                copy = dict(word)
                if "start" in copy:
                    copy["start"] = float(copy["start"]) + offset
                if "end" in copy:
                    copy["end"] = float(copy["end"]) + offset
                words.append(copy)
            clone["words"] = words
        shifted.append(clone)
    return shifted


def merge_segments(
    chunk_list: Sequence[Chunk],
    payloads: dict[int, list[dict]],
    *,
    duration: float,
) -> list[dict]:
    """Stitch chunk results into one timeline, dropping duplicated overlap."""
    chunks = list(chunk_list)
    ownership = owned_ranges(chunks, duration)
    last = chunks[-1] if chunks else None
    merged: list[dict] = []
    for chunk in chunks:
        segments = payloads.get(chunk.index)
        if not segments:
            continue
        low, high = ownership[chunk.index]
        if chunk is last:
            # The final chunk owns everything to the end of the recording, so a
            # segment sitting exactly on the end must not be excluded.
            high += 1e-6
        for segment in shift_segments(segments, chunk.start):
            # A segment belongs to whichever chunk contains its midpoint. Using the
            # midpoint rather than the start keeps a segment that begins just before
            # a boundary with the chunk that decoded most of it.
            midpoint = (segment["start"] + segment["end"]) / 2.0
            if low - 1e-6 <= midpoint < high:
                merged.append(segment)
    merged.sort(key=lambda segment: segment["start"])
    return merged
