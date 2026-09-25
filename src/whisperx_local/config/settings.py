"""The settings a user may persist, declared once.

Each entry is the single source of truth for a setting's type, accepted values and
default. Validation lives here so a typo in a settings file cannot quietly become a
default, and so any interface can offer the accepted values without restating them.
"""

from __future__ import annotations

import platform

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Setting:
    name: str
    kind: type
    default: Any
    choices: tuple[str, ...] | None = None
    help: str = ""


# ``default`` documents the built-in behaviour and is used to describe a setting.
# Only values a user has actually chosen are ever pushed into a parser, so these
# defaults cannot drift out of step with the code that owns the real decision.
SETTINGS: tuple[Setting, ...] = (
    Setting(
        "profile", str, "balanced", ("fast", "balanced", "accurate", "maximum"),
        "Speed and accuracy trade-off",
    ),
    Setting("language", str, "fa", ("fa", "en"), "Spoken language"),
    Setting(
        "output_format", str, "txt",
        ("all", "txt", "srt", "vtt", "tsv", "json", "aud"),
        "Transcript format",
    ),
    Setting("output_dir", str, None, None, "Where transcripts are written"),
    Setting(
        "copy_beside_input", bool, True, None,
        "Also save a copy of the transcript beside the recording",
    ),
    Setting("model", str, None, None, "Recognition model"),
    Setting(
        "engine", str, "whispercpp" if platform.system() == "Darwin" else "whisperx",
        ("whispercpp", "whisperx"),
        "Which recogniser runs",
    ),
    Setting(
        "compute_type", str, None, ("float32", "int8"),
        "Numeric precision",
    ),
    Setting("beam_size", int, None, None, "Beam width for recognition"),
    Setting("batch_size", int, None, None, "Batched inference size"),
    Setting("threads", int, None, None, "CPU threads"),
    Setting(
        "vad_method", str, None, ("pyannote", "silero"),
        "Voice activity detector",
    ),
    Setting("normalize", bool, True, None, "Level the audio before recognition"),
    Setting(
        "silence_split", bool, True, None,
        "Snap chunk boundaries into speech gaps",
    ),
    Setting("silence_min", float, 0.5, None, "Shortest pause to cut on"),
    Setting("chunk_seconds", float, 600.0, None, "Target audio per chunk"),
    Setting(
        "chunks_per_call", int, 1, None,
        "Consecutive chunks covered by one recogniser call; measured neutral, off by default",
    ),
    Setting("chunk_overlap", float, 2.0, None, "Overlap between chunks"),
    Setting("min_turn", float, 0.35, None, "Shortest speaker turn to keep"),
    Setting(
        "diarize", bool, None, None,
        "Label speakers; defaults on when a token is configured",
    ),
    Setting("speakers", int, None, None, "Exact speaker count"),
    Setting(
        "diarize_audio", str, "gentle", ("gentle", "original"),
        "Audio used for speaker separation",
    ),
    Setting(
        "network", str, "auto", ("auto", "offline", "online"),
        "Model cache policy",
    ),
    Setting("retries", int, None, None, "Recognition retries per chunk"),
    Setting("align_retries", int, None, None, "Alignment retries per chunk"),
    Setting("diarize_retries", int, None, None, "Speaker retries"),
    Setting("hotwords", str, None, None, "Comma-separated terms to bias toward"),
    Setting("prompt", str, None, None, "Context sentence that steers vocabulary"),
)

BY_NAME: dict[str, Setting] = {setting.name: setting for setting in SETTINGS}

_TRUE = {"1", "true", "yes", "on", "y", "t"}
_FALSE = {"0", "false", "no", "off", "n", "f"}


class UnknownSetting(KeyError):
    """Raised for a setting name that does not exist."""


class InvalidSetting(ValueError):
    """Raised for a value of the wrong type or outside the accepted choices."""


def coerce(name: str, value: Any) -> Any:
    """Validate and convert one value, with a message the user can act on."""
    setting = BY_NAME.get(name)
    if setting is None:
        known = ", ".join(sorted(BY_NAME))
        raise UnknownSetting(f"unknown setting {name!r}. Known settings: {known}")
    if setting.kind is bool:
        return _as_bool(name, value)
    try:
        converted = setting.kind(value)
    except (TypeError, ValueError) as error:
        raise InvalidSetting(
            f"{name} expects {setting.kind.__name__}, got {value!r}"
        ) from error
    if setting.choices is not None and converted not in setting.choices:
        allowed = ", ".join(setting.choices)
        raise InvalidSetting(f"{name} must be one of: {allowed}")
    return converted


def _as_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
    raise InvalidSetting(f"{name} expects true or false, got {value!r}")


def describe() -> list[tuple[str, str, str]]:
    """``(name, current default, help)`` rows, for a settings listing."""
    rows = []
    for setting in SETTINGS:
        if setting.choices:
            shown = "|".join(setting.choices)
        elif setting.default is None:
            shown = "automatic"
        else:
            shown = str(setting.default)
        rows.append((setting.name, shown, setting.help))
    return rows
