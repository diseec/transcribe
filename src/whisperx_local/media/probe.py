"""Measuring an input: does it have audio, how long is it, how loud is it."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_MEAN_VOLUME = re.compile(r"mean_volume:\s*(-?[0-9.]+) dB")


def has_audio_stream(ffprobe: str, path: Path, *, env: dict[str, str]) -> bool:
    """Whether the file has an audio track at all.

    Checked before anything expensive runs, so a silent screen recording is refused
    with an explanation rather than failing deep inside the recogniser.
    """
    probe = subprocess.run(
        [
            ffprobe, "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    return "audio" in probe.stdout


def probe_duration(ffprobe: str, path: Path, *, env: dict[str, str]) -> float:
    """Length of the file in seconds.

    Required for any honest progress reporting, so a failure here is fatal rather
    than silently disabling the progress bar.
    """
    result = subprocess.run(
        [
            ffprobe, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    value = result.stdout.strip()
    if not value:
        raise ValueError(f"Could not read the duration of {path}")
    return float(value)


def mean_volume_db(ffmpeg: str, path: Path, *, env: dict[str, str]) -> float | None:
    """Average loudness, used to derive a silence threshold for this recording."""
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True,
        text=True,
        env=env,
    )
    match = _MEAN_VOLUME.search(probe.stderr)
    return float(match.group(1)) if match else None
