"""Configuration: what a user may change, and where it is remembered.

``settings`` declares each setting once; ``preferences`` stores the chosen values and
resolves the layers. Nothing here imports the pipeline or the terminal, so it can be
tested and reasoned about on its own.
"""

from whisperx_local.config.preferences import Preferences
from whisperx_local.config.settings import (
    BY_NAME,
    SETTINGS,
    InvalidSetting,
    Setting,
    UnknownSetting,
    coerce,
    describe,
)

__all__ = [
    "BY_NAME",
    "SETTINGS",
    "InvalidSetting",
    "Preferences",
    "Setting",
    "UnknownSetting",
    "coerce",
    "describe",
]
