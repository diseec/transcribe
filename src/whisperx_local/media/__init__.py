"""Media inspection and transformation: the only place ffmpeg is invoked.

Split by concern rather than by convenience:

- ``filters``  the filter chains, which encode measurement results
- ``runner``   running ffmpeg safely, with progress and without deadlocks
- ``probe``    measuring an input
- ``scan``     finding boundaries inside it
- ``tracks``   building the derived audio and cutting chunks

Nothing here imports from ``actions`` or ``services``; the dependency only ever
points this way.
"""

from whisperx_local.media.filters import (
    ENERGY_WINDOW,
    RECOGNITION_FILTER,
    SAMPLE_RATE,
    SILENT_DB,
    SPEAKER_FILTER,
    energy_filter,
)
from whisperx_local.media.probe import has_audio_stream, mean_volume_db, probe_duration
from whisperx_local.media.runner import parse_progress_line, run_ffmpeg_progress
from whisperx_local.media.scan import (
    FALLBACK_THRESHOLD_DB,
    SILENCE_MARGIN_DB,
    detect_silences,
    energy_windows,
    parse_energy_lines,
    silence_cut_points,
)
from whisperx_local.media.tracks import (
    derived_path,
    recognition_audio,
    recognition_path,
    slice_audio,
    speaker_audio,
    speaker_path,
)

__all__ = [
    "ENERGY_WINDOW",
    "FALLBACK_THRESHOLD_DB",
    "RECOGNITION_FILTER",
    "SAMPLE_RATE",
    "SILENCE_MARGIN_DB",
    "SILENT_DB",
    "SPEAKER_FILTER",
    "derived_path",
    "detect_silences",
    "energy_filter",
    "energy_windows",
    "has_audio_stream",
    "mean_volume_db",
    "parse_energy_lines",
    "parse_progress_line",
    "probe_duration",
    "recognition_audio",
    "recognition_path",
    "run_ffmpeg_progress",
    "silence_cut_points",
    "slice_audio",
    "speaker_audio",
    "speaker_path",
]
