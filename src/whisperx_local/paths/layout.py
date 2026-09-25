"""Where everything lives.

Paths only, no behaviour, so importing this never touches the disk and a test can
reason about the layout without a filesystem.
"""

from __future__ import annotations

import os
from pathlib import Path

# This file is <app>/src/whisperx_local/paths/layout.py, so the app root is parents[3].
APP_DIR = Path(__file__).resolve().parents[3]

VENV_DIR = APP_DIR / ".venv"
PYTHON = VENV_DIR / "bin" / "python"
WHISPERX = VENV_DIR / "bin" / "whisperx"

MODELS_DIR = APP_DIR / "models"
HUGGINGFACE_DIR = MODELS_DIR / "huggingface"
TORCH_DIR = MODELS_DIR / "torch"
NLTK_DATA_DIR = MODELS_DIR / "nltk_data"
# Derived audio is cached beside the models: it is expensive to rebuild and is
# worthless without the model that consumed it.
NORMALIZED_DIR = MODELS_DIR / "normalized"
SPEAKER_DIR = MODELS_DIR / "speaker"
DIARIZATION_READY = MODELS_DIR / ".diarization-ready"

INPUT_DIR = APP_DIR / "input"
OUTPUT_DIR = APP_DIR / "output"
STATE_DIR = OUTPUT_DIR / ".state"
WORK_DIR = OUTPUT_DIR / ".work"
LOG_DIR = OUTPUT_DIR / ".logs"

DOWNLOAD_DIR = Path.home() / "Downloads"

# Preferences live outside the app so they survive a reinstall and are shared by
# every checkout. XDG_CONFIG_HOME is honoured because that is what the rest of the
# user's tooling respects on this platform.
CONFIG_DIR = (
    Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    / "whisperx-local"
)
# An override exists so tests can point at a temporary store instead of the user's.
_SETTINGS_OVERRIDE = os.environ.get("WHISPERX_SETTINGS")
SETTINGS_FILE = (
    Path(_SETTINGS_OVERRIDE) if _SETTINGS_OVERRIDE else CONFIG_DIR / "settings.json"
)

GIB = 1024**3
