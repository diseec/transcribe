"""Running the recogniser and its stages.

- ``backends``  which recogniser runs, and how its command and output are spoken to
- ``commands``  building the exact command lines, and the cache signature
- ``stages``    one function per stage, each responsible for its own retries
- ``worker``    the child-process entry points, importable as well as executable
- ``profiles``  speed/quality presets and automatic memory tuning

Importing this package does not import whisperx: the heavy library is loaded inside
the functions that need it, so the app can start, print its help, and run its tests
without the models present.
"""

from whisperx_local.engine import backends
from whisperx_local.engine.commands import (
    SIGNATURE_FIELDS,
    WORKER_MODULE,
    managed_worker_command,
    offline_environment,
    recognition_signature,
    whisperx_command,
)
from whisperx_local.engine.profiles import (
    DEFAULT_PROFILE,
    PROFILES,
    Profile,
    apply,
    describe,
    resolve,
)
from whisperx_local.engine.stages import (
    align_chunk,
    diarize_voices,
    read_turns,
    transcribe_chunk,
    write_turns,
)

__all__ = [
    "DEFAULT_PROFILE",
    "PROFILES",
    "SIGNATURE_FIELDS",
    "WORKER_MODULE",
    "Profile",
    "align_chunk",
    "apply",
    "backends",
    "describe",
    "diarize_voices",
    "managed_worker_command",
    "offline_environment",
    "read_turns",
    "recognition_signature",
    "resolve",
    "transcribe_chunk",
    "whisperx_command",
    "write_turns",
]
