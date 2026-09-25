"""The canonical transcript: recognised text, timestamps, no interpretation.

Written on every run that produced text, before anything else touches it. Two things
make it worth a separate artifact rather than another export format:

- It is **never cleaned**. No speaker smoothing, no turn merging, no filler removal, so
  it is always a faithful record of what was recognised.
- It is **never discarded**. If alignment or speaker separation fails, this file is
  still written, which is what makes those stages safe to attempt.

The readable transcript is rendered from the same segments afterwards, so the two can
be compared and any difference attributed to a specific choice rather than to chance.
"""

from __future__ import annotations

import json
from pathlib import Path

from whisperx_local.transcript.render import timestamp

RAW_TEXT_SUFFIX = "raw.txt"
RAW_JSON_SUFFIX = "raw.json"


def usable_segments(segments: list[dict]) -> list[dict]:
    """Segments carrying text and a timing, in order.

    Segments with no text are dropped because they add nothing, and a segment missing
    its timings cannot be placed in a timestamped transcript.
    """
    usable = [
        segment
        for segment in segments
        if str(segment.get("text", "")).strip() and "start" in segment and "end" in segment
    ]
    return sorted(usable, key=lambda segment: float(segment["start"]))


def coverage_seconds(segments: list[dict]) -> float:
    """How far into the recording recognised text reaches.

    The maximum end time rather than a sum, because the question being answered is
    "how much of the audio was covered", and overlapping or adjacent segments must not
    be counted twice.
    """
    usable = usable_segments(segments)
    if not usable:
        return 0.0
    return max(float(segment["end"]) for segment in usable)


def render_raw_text(segments: list[dict]) -> str:
    """Timestamped recognition text, one segment per line, exactly as recognised."""
    rows = [
        f"[{timestamp(float(segment['start']))} --> {timestamp(float(segment['end']))}]  "
        f"{str(segment.get('text', '')).strip()}"
        for segment in usable_segments(segments)
    ]
    return "\n".join(rows) + ("\n" if rows else "")


def render_raw_json(segments: list[dict], language: str) -> str:
    """The same content as structured data, keeping whatever timings exist.

    Word timings are passed through when alignment produced them and simply absent
    when it did not, so the file always describes the same recognition. The
    recogniser's own ``avg_logprob`` is kept for the same reason: it is part of what
    was produced, and it is the only confidence signal available when there is no
    ground truth to measure accuracy against. Dropping it meant a comparison of two
    engines could report that they disagree without any indication of which one was
    struggling.
    """
    payload = {
        "language": language,
        "canonical": True,
        "segments": [
            {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": str(segment.get("text", "")).strip(),
                **(
                    {"words": segment["words"]}
                    if segment.get("words")
                    else {}
                ),
                **(
                    {"avg_logprob": float(segment["avg_logprob"])}
                    if isinstance(segment.get("avg_logprob"), (int, float))
                    else {}
                ),
            }
            for segment in usable_segments(segments)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def write_canonical(output_dir: Path, stem: str, segments: list[dict], language: str) -> list[Path]:
    """Write both canonical forms, returning the paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix, body in (
        (RAW_TEXT_SUFFIX, render_raw_text(segments)),
        (RAW_JSON_SUFFIX, render_raw_json(segments, language)),
    ):
        destination = output_dir / f"{stem}.{suffix}"
        destination.write_text(body, encoding="utf-8")
        written.append(destination)
    return written
