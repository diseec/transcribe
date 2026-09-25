"""Finding where one chunk may end and the next begin.

Two strategies, tried in order:

1. Silence, which is the safe boundary because no word can straddle it.
2. Loudness minima, for recordings that contain no silence at all.

The second exists because the first silently failed on real material: a measured
52-minute meeting had no pause below -80 dB, so every boundary degraded to an
arbitrary point. Loudness minima always exist, but they are only trusted when the
recording has enough dynamic range for a minimum to mean anything -- heavily limited
audio is flat to within a decibel or two, and picking the "calmest" window there is
noise rather than signal.
"""

from __future__ import annotations

import re
from pathlib import Path

from whisperx_local.media.filters import ENERGY_WINDOW, SILENT_DB, energy_filter
from whisperx_local.media.probe import mean_volume_db
from whisperx_local.media.runner import run_ffmpeg_progress

_SILENCE_START = re.compile(r"silence_start:\s*(-?[0-9.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?[0-9.]+)")
_PTS_TIME = re.compile(r"pts_time:\s*(-?[0-9.]+)")
_RMS_LEVEL = re.compile(r"RMS_level=\s*(-?[0-9.]+|-?inf|nan)")

# How far below the recording's own average a pause has to be. Derived rather than
# fixed because a fixed value fails both ways: on a quiet recording speech reads as
# silence, and on a loud one silence is never detected at all.
SILENCE_MARGIN_DB = 22.0
FALLBACK_THRESHOLD_DB = -50.0


def detect_silences(
    ffmpeg: str,
    path: Path,
    *,
    env: dict[str, str],
    noise_db: float | None = None,
    min_silence: float = 0.5,
    duration: float | None = None,
    on_progress=None,
    log=None,
) -> list[tuple[float, float]]:
    """Return silence spans as ``(start, end)`` pairs."""
    if noise_db is None:
        mean = mean_volume_db(ffmpeg, path, env=env)
        noise_db = round(mean - SILENCE_MARGIN_DB, 1) if mean is not None else FALLBACK_THRESHOLD_DB

    text = run_ffmpeg_progress(
        ffmpeg,
        [
            "-i", str(path),
            "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
            "-f", "null", "-",
        ],
        env=env,
        duration=duration,
        on_progress=on_progress,
        log=log,
    )
    starts = [float(value) for value in _SILENCE_START.findall(text)]
    ends = [float(value) for value in _SILENCE_END.findall(text)]

    spans: list[tuple[float, float]] = []
    for index, start in enumerate(starts):
        if index < len(ends):
            end = ends[index]
        elif duration is not None:
            # A trailing pause has no end line; the file ends when it ends.
            end = duration
        else:
            continue
        if end > start:
            spans.append((max(0.0, start), end))
    return spans


def silence_cut_points(spans: list[tuple[float, float]]) -> list[float]:
    """Midpoints of silence spans: the safest places to cut."""
    return [round((start + end) / 2.0, 3) for start, end in spans if end > start]


def parse_energy_lines(lines) -> list[tuple[float, float]]:
    """Pair each ``pts_time`` with the ``RMS_level`` that follows it.

    The filter prints a timestamp line and then a value line, so the pair has to be
    carried across two lines. ``pts_time`` marks the start of the window.
    """
    samples: list[tuple[float, float]] = []
    pending: float | None = None
    for line in lines:
        stamp = _PTS_TIME.search(line)
        if stamp is not None:
            pending = float(stamp.group(1))
            continue
        level = _RMS_LEVEL.search(line)
        if level is None or pending is None:
            continue
        try:
            value = float(level.group(1))
        except ValueError:
            value = SILENT_DB
        if value != value:  # nan compares false against everything, including itself
            value = 0.0
        elif value == float("-inf"):
            value = SILENT_DB
        samples.append((pending, value))
        pending = None
    return samples


def energy_windows(
    ffmpeg: str,
    path: Path,
    *,
    env: dict[str, str],
    window: float = ENERGY_WINDOW,
    duration: float | None = None,
    on_progress=None,
    log=None,
) -> list[tuple[float, float]]:
    """Per-window loudness of a file as ``(start, dBFS)`` pairs."""
    collected: list[str] = []
    text = run_ffmpeg_progress(
        ffmpeg,
        [
            "-loglevel", "error",
            "-i", str(path),
            "-af", energy_filter(window),
            "-f", "null", "-",
        ],
        env=env,
        duration=duration,
        on_progress=on_progress,
        on_line=collected.append,
        log=log,
    )
    samples = parse_energy_lines(collected)
    if not samples:
        # Some builds route filter metadata to stderr instead of stdout.
        samples = parse_energy_lines(text.splitlines())

    unique: list[tuple[float, float]] = []
    seen: set[float] = set()
    for stamp, value in samples:
        key = round(stamp, 4)
        if key in seen:
            continue
        seen.add(key)
        unique.append((stamp, value))
    return unique
