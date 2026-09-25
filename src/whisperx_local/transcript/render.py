"""Writing a transcript out in each supported format.

Renderers take prepared lines and return text, so they are pure and can be checked
against exact expected strings without touching the disk.
"""

from __future__ import annotations

import json
from pathlib import Path

from whisperx_local.transcript.assemble import prepare_lines

# The formats a run may ask for. Order matters only for the listing shown to a user.
FORMAT_ORDER = ("txt", "srt", "vtt", "tsv", "json", "aud")


def timestamp(seconds: float, *, comma: bool = False) -> str:
    """HH:MM:SS.mmm, or with a comma for SubRip, which requires it."""
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole, milliseconds = divmod(milliseconds, 1000)
    marker = "," if comma else "."
    return f"{hours:02d}:{minutes:02d}:{whole:02d}{marker}{milliseconds:03d}"


def render_txt(lines: list, *, timestamps: bool = False) -> str:
    out: list[str] = []
    for line in lines:
        text = line.text
        if not text:
            continue
        if line.speaker:
            label = (
                f"{line.speaker}: "
                if not timestamps
                else f"[{timestamp(line.start)}] {line.speaker}: "
            )
            out.append(f"{label}{text}")
        else:
            out.append(text)
    return "\n".join(out) + ("\n" if out else "")


def render_srt(lines: list) -> str:
    blocks = []
    index = 0
    for line in lines:
        if not line.text:
            continue
        index += 1
        prefix = f"{line.speaker}: " if line.speaker else ""
        blocks.append(
            f"{index}\n"
            f"{timestamp(line.start, comma=True)} --> {timestamp(line.end, comma=True)}\n"
            f"{prefix}{line.text}\n"
        )
    return "\n".join(blocks)


def render_vtt(lines: list) -> str:
    blocks = ["WEBVTT\n"]
    for line in lines:
        if not line.text:
            continue
        prefix = f"{line.speaker}: " if line.speaker else ""
        blocks.append(
            f"{timestamp(line.start)} --> {timestamp(line.end)}\n{prefix}{line.text}\n"
        )
    return "\n".join(blocks)


def render_tsv(lines: list) -> str:
    rows = ["start\tend\tspeaker\ttext"]
    for line in lines:
        if line.text:
            rows.append(f"{line.start:.3f}\t{line.end:.3f}\t{line.speaker or ''}\t{line.text}")
    return "\n".join(rows) + "\n"


def render_aud(lines: list) -> str:
    """Audacity label format: tab separated, seconds with six decimals."""
    rows = []
    for line in lines:
        if line.text:
            rows.append(f"{line.start:.6f}\t{line.end:.6f}\t{line.text}")
    return "\n".join(rows) + ("\n" if rows else "")


def render_json(segments: list[dict], turns: list, language: str) -> str:
    """The verbose form: every word carries its own timing and speaker."""
    lines = prepare_lines(segments, turns)
    payload = {
        "language": language,
        "segments": [
            {
                "start": line.start,
                "end": line.end,
                "speaker": line.speaker,
                "text": line.text,
                "words": [
                    {
                        "word": str(word.get("word", "")),
                        "start": float(word.get("start", line.start)),
                        "end": float(word.get("end", line.end)),
                        "speaker": line.speaker,
                    }
                    for word in line.words
                ],
            }
            for line in lines
            if line.text
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _from_lines(renderer):
    """Adapt a line renderer into the ``(segments, turns, language)`` signature."""

    def render(segments: list[dict], turns: list, language: str) -> str:
        return renderer(prepare_lines(segments, turns))

    return render


RENDERERS = {
    "txt": _from_lines(render_txt),
    "srt": _from_lines(render_srt),
    "vtt": _from_lines(render_vtt),
    "tsv": _from_lines(render_tsv),
    "aud": _from_lines(render_aud),
    "json": render_json,
}


def write_formats(
    output_dir: Path,
    stem: str,
    segments: list[dict],
    turns: list,
    *,
    language: str,
    formats: list[str],
) -> list[Path]:
    """Write each requested format, returning the files written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in formats:
        renderer = RENDERERS[name]
        destination = output_dir / f"{stem}.{name}"
        destination.write_text(renderer(segments, turns, language), encoding="utf-8")
        written.append(destination)
    return written
