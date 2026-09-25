"""Grouping chunks so one model load covers several of them.

**Measured neutral on this machine, so it is off by default.** The reasoning that motivated
it was that each recogniser invocation loads the model from scratch, and a first attempt
to price that put it at roughly 25 seconds -- inferred from running the same twenty seconds
of audio as one chunk and as two. That inference did not survive a direct test. Over 240
seconds of audio cut into four chunks, on a warm cache:

    chunks_per_call=1   174.2 s
    chunks_per_call=3   174.7 s

A difference of half a second, in the wrong direction. CTranslate2 maps the model file, so
once the pages are cached, "loading" costs almost nothing, and the two-thousand-second
files still showed no saving from halving the number of invocations. The twenty-second
experiment must have been measuring disk, not model initialisation.

So the mechanism stays, because it is tested, it is text-neutral (verified: 0 errors
against the ungrouped transcript of the same audio), and a machine with slower storage or a
cold cache across many chunks should benefit. It is not switched on, because nothing here
shows a reason to. Enabling it is ``--chunks-per-call N``.

Two things stay true regardless of whether it helps:

* **Only contiguous chunks are grouped.** Where an overlap makes the audio non-contiguous,
  joining would duplicate speech, so those runs keep going one chunk at a time.
* **A group that fails falls back to one chunk per invocation**, so the worst case is the
  behaviour that existed before this did.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from whisperx_local.chunking.plan import Chunk

# Off by default: measured neutral here. Three chunks at the default 600 s target is half
# an hour of audio per invocation, which is where a machine that *does* pay for model
# loading would want it.
DEFAULT_CHUNKS_PER_CALL = 1

# Cut points are stored rounded to three decimals, so contiguity holds to about a
# millisecond. The tolerance is loose enough for that rounding and far tighter than any
# real overlap, which is minutes wide by contrast.
CONTIGUOUS_TOLERANCE = 0.01


def is_contiguous(first: Chunk, second: Chunk) -> bool:
    """Whether these two chunks' audio can be joined without gap or duplication."""
    return abs((first.start + first.length) - second.start) <= CONTIGUOUS_TOLERANCE


def group_chunks(
    chunks: Sequence[Chunk],
    *,
    per_call: int = DEFAULT_CHUNKS_PER_CALL,
    pending: Iterable[int] | None = None,
) -> list[list[Chunk]]:
    """Consecutive chunks still needing work, grouped and never spanning a gap.

    ``pending`` names the indices that still need recognising. A finished chunk is left
    out rather than included and skipped later, because joining its audio in would make
    the model read speech whose result would then be thrown away.
    """
    wanted = None if pending is None else set(pending)
    limit = max(1, int(per_call))
    groups: list[list[Chunk]] = []
    current: list[Chunk] = []
    previous: Chunk | None = None

    for chunk in chunks:
        if wanted is not None and chunk.index not in wanted:
            # A finished chunk is a gap. Nothing may be joined across it.
            if current:
                groups.append(current)
            current, previous = [], None
            continue
        if previous is not None and not is_contiguous(previous, chunk):
            if current:
                groups.append(current)
            current = []
        current.append(chunk)
        if len(current) >= limit:
            groups.append(current)
            current, previous = [], None
            continue
        previous = chunk

    if current:
        groups.append(current)
    return groups


def chunk_spans(group: Sequence[Chunk]) -> list[tuple[Chunk, float, float]]:
    """Each chunk with its span inside the joined audio, from the audio lengths.

    Taken from the chunk lengths rather than from the start times, so this stays correct
    for a group whose members are contiguous but not evenly sized.
    """
    spans: list[tuple[Chunk, float, float]] = []
    cursor = 0.0
    for chunk in group:
        spans.append((chunk, cursor, cursor + chunk.length))
        cursor += chunk.length
    return spans


def distribute_segments(group: Sequence[Chunk], segments: list[dict]) -> dict[int, list[dict]]:
    """Hand each recognised segment to the chunk whose span contains its middle.

    The same ownership rule the merge uses, applied here so the per-chunk cache keeps the
    shape everything downstream expects: timings local to their own chunk, because that is
    what ``merge_segments`` shifts back when it rejoins them.

    A segment is placed by its middle, never split, for the same reason the comparison
    harness does it that way: a phrase cut in half at an arbitrary point is scored, and
    here transcribed, as two pieces that belong to neither chunk.
    """
    spans = chunk_spans(group)
    grouped: dict[int, list[dict]] = {chunk.index: [] for chunk in group}
    if not spans:
        return grouped

    for segment in segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        middle = (start + end) / 2.0
        owner, offset = spans[-1][0], spans[-1][1]
        for chunk, low, high in spans:
            if low <= middle < high:
                owner, offset = chunk, low
                break
        else:
            # Past the end of the last chunk, which the recogniser can report by a
            # fraction of a second. It belongs to the last chunk rather than nowhere.
            owner, offset = spans[-1][0], spans[-1][1]
            if middle < 0:
                owner, offset = spans[0][0], spans[0][1]
        grouped[owner.index].append(_rebase(segment, offset))
    return grouped


def _rebase(segment: dict, offset: float) -> dict:
    """Move a segment from the joined audio's clock onto its own chunk's clock."""
    clone = dict(segment)
    clone["start"] = float(segment.get("start", 0.0)) - offset
    clone["end"] = float(segment.get("end", clone["start"])) - offset
    if segment.get("words"):
        clone["words"] = [
            {
                **word,
                **({"start": float(word["start"]) - offset} if "start" in word else {}),
                **({"end": float(word["end"]) - offset} if "end" in word else {}),
            }
            for word in segment["words"]
        ]
    return clone
