"""Progress tracking: weighted, monotonic, and predictable.

``model`` owns the maths, ``timing`` owns what the machine has taught us. Both are
free of Rich and of subprocesses, so they are testable without a terminal.
"""

from whisperx_local.progress.model import DEFAULT_STAGES, TIME_CEILING, ProgressModel, Stage
from whisperx_local.progress.timing import TimingStore, context_key

__all__ = [
    "DEFAULT_STAGES",
    "TIME_CEILING",
    "ProgressModel",
    "Stage",
    "TimingStore",
    "context_key",
]
