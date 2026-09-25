"""Speed and quality profiles, plus automatic memory tuning.

WhisperX defaults to ``--batch_size 8`` and uses a batched dataloader whenever
``batch_size > 1``. The previous wrapper forced ``1``, which silently selected the
slow sequential path, and pinned ``--threads`` to 4 while this machine has 8
performance cores. Both are pure throughput wins that cost nothing in accuracy, so
they are corrected here regardless of profile.
"""

from __future__ import annotations

from dataclasses import dataclass

from whisperx_local.paths import GIB, available_memory_bytes, performance_cores


@dataclass(frozen=True)
class Profile:
    name: str
    model: str
    compute_type: str
    beam_size: int
    best_of: int
    patience: float
    batch_size: int
    vad_method: str
    summary: str
    threads: int = 0  # 0 means "use every performance core"

    def defaults(self) -> dict[str, object]:
        """The options this profile supplies when the user has not chosen one."""
        return {
            "model": self.model,
            "compute_type": self.compute_type,
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "vad_method": self.vad_method,
            "threads": self.threads or performance_cores(),
        }


PROFILES: dict[str, Profile] = {
    "fast": Profile(
        name="fast",
        model="large-v3",
        compute_type="int8",
        beam_size=1,
        best_of=1,
        patience=1.0,
        batch_size=16,
        vad_method="silero",
        summary="Greedy decoding, int8, big batches. Several times faster, a little less exact.",
    ),
    "balanced": Profile(
        name="balanced",
        model="large-v3",
        compute_type="int8",
        beam_size=5,
        best_of=5,
        patience=1.0,
        batch_size=8,
        vad_method="pyannote",
        summary="Beam search with int8 batching. Good speed without giving up much.",
    ),
    "accurate": Profile(
        name="accurate",
        model="large-v3",
        # int8 rather than float32. Measured on this machine, float32 roughly doubles the
        # time for the same audio while the wide beam search -- which is what actually
        # buys accuracy here -- is unaffected. Full precision is still available, as
        # ``maximum`` below, for when the last fraction is worth the wait.
        compute_type="int8",
        beam_size=10,
        best_of=10,
        patience=2.0,
        batch_size=4,
        vad_method="pyannote",
        summary="Widest search at int8 speed. Slow, closest to the best result per minute.",
    ),
    "maximum": Profile(
        name="maximum",
        model="large-v3",
        compute_type="float32",
        beam_size=10,
        best_of=10,
        patience=2.0,
        batch_size=4,
        vad_method="pyannote",
        summary="Full precision and the widest search. Much slower; the last fraction of accuracy.",
    ),
}

DEFAULT_PROFILE = "balanced"

# (available memory ceiling, max batch, max threads, max chunk seconds). Ordered from
# the tightest constraint, and applied to the first tier that matches.
_MEMORY_TIERS = (
    (0.75 * GIB, 1, 2, 240.0),
    (1.50 * GIB, 2, 4, 300.0),
    (3.00 * GIB, 8, 0, 600.0),
)

# Above this, nothing is adjusted.
COMFORTABLE_MEMORY = 3.0 * GIB


def resolve(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise SystemExit(
            f"Unknown profile {name!r}. Choose one of: {', '.join(PROFILES)}"
        ) from None


def describe(name: str) -> str:
    return resolve(name).summary


def apply(options, *, available: int | None = None) -> tuple[Profile, list[str]]:
    """Fill unset options from the profile, then adapt to free memory.

    An explicitly supplied value always wins over the profile. Memory tuning can
    still override the batch size, because ignoring it is what caused a native abort
    two hours into a run: the abort consumed the whole transcript that had just been
    produced. Returns the profile and the notes to show the user.
    """
    profile = resolve(options.profile)
    notes: list[str] = []

    for name, value in profile.defaults().items():
        if getattr(options, name, None) is None:
            setattr(options, name, value)

    if available is None:
        available = available_memory_bytes()
    if available is None or available >= COMFORTABLE_MEMORY:
        return profile, notes

    for ceiling, max_batch, max_threads, max_chunk in _MEMORY_TIERS:
        if available < ceiling:
            if options.batch_size > max_batch:
                notes.append(
                    f"Free memory is {available / GIB:.1f} GB, so batch size was "
                    f"reduced from {options.batch_size} to {max_batch}."
                )
                options.batch_size = max_batch
            if max_threads and options.threads > max_threads:
                notes.append(f"Threads reduced to {max_threads} to limit peak memory.")
                options.threads = max_threads
            if options.chunk_seconds > max_chunk:
                notes.append(
                    f"Chunks shortened to {int(max_chunk)}s to bound memory per step."
                )
                options.chunk_seconds = max_chunk
            break

    return profile, notes
