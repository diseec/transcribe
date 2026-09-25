"""Chunked, resumable execution.

The package is pure: planning and merging take numbers and return chunks or merged
segments, with no disk, no subprocess and no clock. That is what makes the awkward
boundary cases testable without a real recording.

- ``plan``   where boundaries go, including the loudness fallback
- ``seams``  how results from separate chunks are rejoined without loss or duplication
- ``store``  how finished chunks are cached and invalidated
"""

from whisperx_local.chunking.batching import (
    DEFAULT_CHUNKS_PER_CALL,
    chunk_spans,
    distribute_segments,
    group_chunks,
    is_contiguous,
)
from whisperx_local.chunking.plan import (
    BOUNDARY_TOLERANCE,
    DEFAULT_MIN_CHUNK_SECONDS,
    MIN_LOUDNESS_SPREAD_DB,
    Chunk,
    loudness_spread,
    plan_chunks,
    quiet_cut_points,
)
from whisperx_local.chunking.seams import merge_segments, owned_ranges, shift_segments
from whisperx_local.chunking.store import (
    atomic_write_json,
    chunk_paths,
    completed_indices,
    discard_recognition,
    plan_signature,
    read_segments,
)

__all__ = [
    "BOUNDARY_TOLERANCE",
    "DEFAULT_MIN_CHUNK_SECONDS",
    "MIN_LOUDNESS_SPREAD_DB",
    "Chunk",
    "DEFAULT_CHUNKS_PER_CALL",
    "atomic_write_json",
    "chunk_paths",
    "chunk_spans",
    "completed_indices",
    "discard_recognition",
    "distribute_segments",
    "group_chunks",
    "is_contiguous",
    "loudness_spread",
    "merge_segments",
    "owned_ranges",
    "plan_chunks",
    "plan_signature",
    "quiet_cut_points",
    "read_segments",
    "shift_segments",
]
