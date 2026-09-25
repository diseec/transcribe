"""Derived audio tracks and slicing.

Two derived tracks are built from one recording, because recognition and speaker
separation want opposite things from the same audio: recognition wants every quiet
word pushed up until it is audible, while speaker separation needs the level to stay
faithful or the voice cues it matches on are smeared away. Caching them separately
means that conflict is resolved once, on disk, rather than in the middle of a run.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from whisperx_local.media.filters import RECOGNITION_FILTER, SAMPLE_RATE, SPEAKER_FILTER
from whisperx_local.media.runner import run_ffmpeg_progress
from whisperx_local.paths import NORMALIZED_DIR, SPEAKER_DIR, managed_env

# Bumped when a filter chain changes, so cached audio cannot outlive the settings
# that produced it.
RECOGNITION_PREFIX = "rec-v5"
SPEAKER_PREFIX = "spk-v1"


def _identity(prefix: str, path: Path) -> str:
    stat = path.stat()
    return hashlib.sha256(
        f"{prefix}:{path}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    ).hexdigest()[:12]


def derived_path(directory: Path, prefix: str, source: Path) -> Path:
    """Where a derived track is cached, whether or not it exists yet.

    Exposed separately from building it so the rest of the app can ask what is
    already on disk without invoking ffmpeg to find out.
    """
    return directory / _identity(prefix, source) / f"{source.stem}.wav"


def recognition_path(source: Path) -> Path:
    """The cached recognition track for a recording."""
    return derived_path(NORMALIZED_DIR, RECOGNITION_PREFIX, source)


def speaker_path(source: Path) -> Path:
    """The cached speaker-separation track for a recording."""
    return derived_path(SPEAKER_DIR, SPEAKER_PREFIX, source)


def slice_audio(
    ffmpeg: str,
    source: Path,
    start: float,
    length: float,
    destination: Path,
    *,
    env: dict[str, str],
) -> None:
    """Cut one chunk to a mono WAV at the recognition sample rate."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.3f}",
            "-i", str(source),
            "-t", f"{length:.3f}",
            "-vn",
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            str(destination),
        ],
        check=True,
        env=env,
    )


def _derive(
    source: Path,
    *,
    ffmpeg: str,
    directory: Path,
    prefix: str,
    audio_filter: str,
    duration: float | None = None,
    on_progress=None,
    log=None,
) -> Path:
    destination = derived_path(directory, prefix, source)
    if destination.is_file():
        return destination
    destination_dir = destination.parent
    destination_dir.mkdir(parents=True, exist_ok=True)
    run_ffmpeg_progress(
        ffmpeg,
        [
            "-loglevel", "error", "-y",
            "-i", str(source),
            "-vn",
            "-af", audio_filter,
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            str(destination),
        ],
        env=managed_env(),
        duration=duration,
        on_progress=on_progress,
        log=log,
    )
    return destination


def recognition_audio(
    source: Path,
    ffmpeg: str,
    enabled: bool = True,
    *,
    duration: float | None = None,
    on_progress=None,
    log=None,
) -> Path:
    """Speech-levelled mono WAV used for recognition."""
    if not enabled:
        return source
    return _derive(
        source,
        ffmpeg=ffmpeg,
        directory=NORMALIZED_DIR,
        prefix=RECOGNITION_PREFIX,
        audio_filter=RECOGNITION_FILTER,
        duration=duration,
        on_progress=on_progress,
        log=log,
    )


def speaker_audio(
    source: Path,
    ffmpeg: str,
    *,
    duration: float | None = None,
    on_progress=None,
    log=None,
) -> Path:
    """Gently levelled mono WAV used for speaker separation."""
    return _derive(
        source,
        ffmpeg=ffmpeg,
        directory=SPEAKER_DIR,
        prefix=SPEAKER_PREFIX,
        audio_filter=SPEAKER_FILTER,
        duration=duration,
        on_progress=on_progress,
        log=log,
    )
