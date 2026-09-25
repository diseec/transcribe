"""Filesystem layout, process environment, and external tool discovery.

Split three ways because they change for different reasons: ``layout`` when the
directory structure moves, ``environment`` when tuning or child-process settings
change, and ``tools`` when something outside the app changes.

Everything is re-exported here so callers can write ``from whisperx_local.paths
import WORK_DIR`` without knowing which part owns it.
"""

from whisperx_local.paths.environment import (
    ALLOWED_ENV_NAMES,
    SUPPORTED_LINUX_ARCHITECTURES,
    SUPPORTED_PYTHON,
    available_memory_bytes,
    host_architecture,
    load_local_env,
    managed_env,
    performance_cores,
    require_install,
    supported_host,
)
from whisperx_local.paths.layout import (
    APP_DIR,
    CONFIG_DIR,
    DIARIZATION_READY,
    DOWNLOAD_DIR,
    GIB,
    HUGGINGFACE_DIR,
    INPUT_DIR,
    LOG_DIR,
    MODELS_DIR,
    NLTK_DATA_DIR,
    NORMALIZED_DIR,
    OUTPUT_DIR,
    PYTHON,
    SETTINGS_FILE,
    SPEAKER_DIR,
    STATE_DIR,
    TORCH_DIR,
    VENV_DIR,
    WHISPERX,
    WORK_DIR,
)
from whisperx_local.paths.tools import (
    nltk_resource_available,
    working_ffmpeg,
    working_ffprobe,
)

__all__ = [
    "ALLOWED_ENV_NAMES",
    "APP_DIR",
    "CONFIG_DIR",
    "DIARIZATION_READY",
    "DOWNLOAD_DIR",
    "GIB",
    "HUGGINGFACE_DIR",
    "INPUT_DIR",
    "LOG_DIR",
    "MODELS_DIR",
    "NLTK_DATA_DIR",
    "NORMALIZED_DIR",
    "OUTPUT_DIR",
    "PYTHON",
    "SETTINGS_FILE",
    "SPEAKER_DIR",
    "STATE_DIR",
    "SUPPORTED_PYTHON",
    "SUPPORTED_LINUX_ARCHITECTURES",
    "TORCH_DIR",
    "VENV_DIR",
    "WHISPERX",
    "WORK_DIR",
    "available_memory_bytes",
    "host_architecture",
    "load_local_env",
    "managed_env",
    "nltk_resource_available",
    "performance_cores",
    "require_install",
    "supported_host",
    "working_ffmpeg",
    "working_ffprobe",
]
