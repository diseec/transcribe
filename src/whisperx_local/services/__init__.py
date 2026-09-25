"""Orchestration and the facts a run leaves behind.

- ``artifacts``  where derived work lives, and which of it already exists
- ``jobs``       job identity, the completion marker, and cache-readiness checks
- ``runner``     executing a plan, one function per action

The runner is the only place that knows the order in which actions are useful, and it
takes that order from the plan rather than deciding it, so a partial run stays honest
about what it actually did.
"""

from whisperx_local.services.artifacts import WorkSpace, identity
from whisperx_local.services.jobs import (
    PIPELINE_VERSION,
    core_models_cached,
    diarization_ready,
    fingerprint,
    is_complete,
    mark_complete,
    mark_core_ready,
    mark_diarization_ready,
    marker_path,
)
from whisperx_local.services.runner import Outcome, describe_plan, execute

__all__ = [
    "PIPELINE_VERSION",
    "Outcome",
    "WorkSpace",
    "core_models_cached",
    "describe_plan",
    "diarization_ready",
    "execute",
    "fingerprint",
    "identity",
    "is_complete",
    "mark_complete",
    "mark_core_ready",
    "mark_diarization_ready",
    "marker_path",
]
