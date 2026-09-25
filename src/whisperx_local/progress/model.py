"""Monotonic, weighted progress for a whole run.

The previous dashboard gave every stage its own 0-100% sweep, so the bar filled
and reset six times, and no stage's pace was comparable to another's. This model
gives each stage a slice of one global bar, sized by its expected cost, keeps the
result monotonic, and advances on elapsed time when the work itself reports
nothing, so silent phases (model loading, native alignment) still move.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class Stage:
    key: str
    label: str
    weight: float


# Relative expected cost. These only matter for the first run, before
# ``timing.TimingStore`` has measured the machine; afterwards the bands are sized
# from real durations. Transcription dominates on CPU, which is why it holds the
# largest share.
DEFAULT_STAGES: tuple[Stage, ...] = (
    Stage("prepare", "Preparing audio", 0.02),
    Stage("speech", "Finding speech", 0.03),
    Stage("transcribe", "Transcribing", 0.70),
    Stage("align", "Aligning words", 0.15),
    Stage("speakers", "Identifying speakers", 0.09),
    Stage("write", "Writing transcript", 0.01),
)

# A stage must never *look* finished while it is still running, or the bar parks at
# 99% for minutes. Elapsed time alone may therefore claim at most this much of a
# stage's band.
TIME_CEILING = 0.97

# An estimate taken from one instant swings wildly, because chunked work advances in
# bursts: the bar sits still through model loading and then moves quickly. The rate is
# therefore measured across a window at least this wide, and samples are taken no
# more often than the second constant, which bounds the history at any refresh rate.
ETA_WINDOW_SECONDS = 30.0
ETA_SAMPLE_SECONDS = 2.0
SAMPLE_LIMIT = 128


class ProgressModel:
    """Weighted, monotonic progress across ordered stages.

    ``expected`` maps stage keys to predicted durations in seconds, usually from
    :class:`whisperx_local.progress.timing.TimingStore`. It is optional: without it
    the bands still give each stage a fair share, and progress simply waits for real
    reports instead of interpolating.
    """

    def __init__(
        self,
        stages: Sequence[Stage] = DEFAULT_STAGES,
        *,
        expected: Mapping[str, float] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not stages:
            raise ValueError("at least one stage is required")
        self.stages = tuple(stages)
        self._clock = clock
        self._expected = {
            key: float(value) for key, value in (expected or {}).items() if value > 0
        }
        self._bands = self._assign_bands(self.stages, self._expected)
        self._order = {stage.key: index for index, stage in enumerate(self.stages)}
        self._index = 0
        self._reported = 0.0
        self._units = (1, 1)
        self._done: set[str] = set()
        self._seconds: dict[str, float] = {}
        self._samples: deque[tuple[float, float]] = deque(maxlen=SAMPLE_LIMIT)
        self._started_at = self._clock()
        self._stage_started = self._started_at

    @staticmethod
    def _assign_bands(
        stages: Sequence[Stage],
        expected: Mapping[str, float] | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Lay the stages end to end across 0-100, proportional to expected cost.

        Predicted durations are used when they exist, and they are what make the bar
        advance at a steady rate. Static weights cannot: measured on this machine,
        speaker separation really costs about 36% of a run while transcription costs
        about 51%, so the built-in 9% band for speakers filled almost at once and then
        parked while work continued -- the exact "jumps to 60% then crawls" complaint.

        A stage with no history keeps its relative standing by taking the cost per unit
        of static weight implied by the stages that do, so a first run is still sensible.
        """
        expected = expected or {}
        weights = {stage.key: max(0.0, stage.weight) for stage in stages}
        known = {
            key: value
            for key, value in expected.items()
            if key in weights and value > 0
        }
        known_seconds = sum(known.values())
        known_weight = sum(weights[key] for key in known)
        if known_seconds > 0 and known_weight > 0:
            per_weight = known_seconds / known_weight
            weights = {
                key: known.get(key) or value * per_weight
                for key, value in weights.items()
            }

        total = sum(weights.values())
        bands: dict[str, tuple[float, float]] = {}
        edge = 0.0
        for stage in stages:
            span = (weights[stage.key] / total * 100.0) if total > 0 else 100.0 / len(stages)
            bands[stage.key] = (edge, edge + span)
            edge += span
        return bands

    # -- introspection ----------------------------------------------------
    @property
    def current(self) -> Stage:
        return self.stages[self._index]

    def elapsed(self) -> float:
        return self._clock() - self._started_at

    def band(self, key: str) -> tuple[float, float]:
        return self._bands[key]

    def is_done(self, key: str) -> bool:
        """True once a stage has finished or been skipped as not part of this run."""
        return key in self._done

    def is_running(self, key: str) -> bool:
        """True for the stage currently accepting reports."""
        return key == self.current.key and key not in self._done

    def measured(self) -> dict[str, float]:
        """Observed seconds per completed stage, for next-run prediction.

        Skipped stages are deliberately absent rather than recorded as zero, which
        would otherwise teach the predictor that a stage costs nothing.
        """
        return dict(self._seconds)

    # -- driving ----------------------------------------------------------
    def begin(self, key: str) -> None:
        """Make ``key`` the running stage and reset its own progress.

        Moving forward is the only direction allowed, and the new stage starts at
        the end of the previous stage's band, so the bar never rewinds.
        """
        if key not in self._order:
            raise KeyError(f"unknown stage: {key}")
        self._index = self._order[key]
        self._reported = 0.0
        self._units = (1, 1)
        self._stage_started = self._clock()

    def set_units(self, position: int, total: int) -> None:
        """Declare that the running stage has ``total`` units, such as chunks.

        Percentages then describe the current unit, and the stage band is filled
        (position - 1 + unit_fraction) / total of the way, so per-chunk restarts do
        not reset the bar.
        """
        self._units = (max(1, int(position)), max(1, int(total)))
        self._reported = 0.0

    def report(self, percent: float) -> None:
        """Record progress within the running stage, or the running unit of it."""
        self._reported = max(self._reported, max(0.0, min(float(percent), 100.0)))

    def complete(self) -> None:
        """Mark the running stage finished and move on to the next one."""
        key = self.current.key
        self._seconds[key] = max(0.0, self._clock() - self._stage_started)
        self._done.add(key)
        if self._index + 1 < len(self.stages):
            self._index += 1
            self._reported = 0.0
            self._units = (1, 1)
            self._stage_started = self._clock()

    def skip(self, *keys: str) -> None:
        """Mark stages as not part of this run, so the bar never waits on them."""
        for key in keys:
            if key not in self._order:
                raise KeyError(f"unknown stage: {key}")
            self._done.add(key)

    # -- derived values ---------------------------------------------------
    def _reported_fraction(self) -> float:
        position, total = self._units
        if total <= 1:
            return self._reported / 100.0
        return ((position - 1) + self._reported / 100.0) / total

    def _time_fraction(self) -> float | None:
        expected = self._expected.get(self.current.key)
        if not expected:
            return None
        elapsed = self._clock() - self._stage_started
        return min(elapsed / expected, TIME_CEILING)

    def fraction(self) -> float:
        """How far through the running stage we are, by reports or by time.

        Capped below 1.0 either way: a sub-source reporting 100% does not mean the
        stage has finished, because only :meth:`complete` decides that. Reserving the
        top of a band is what keeps the bar moving instead of parking at its end while
        a slow tail of work continues.
        """
        reported = self._reported_fraction()
        predicted = self._time_fraction()
        leading = reported if predicted is None else max(reported, predicted)
        return min(leading, TIME_CEILING)

    def percent(self) -> float:
        value = self._value()
        self._remember(value)
        return value

    def _value(self) -> float:
        if len(self._done) >= len(self.stages):
            return 100.0
        highest = 0.0
        for stage in self.stages:
            if stage.key in self._done:
                highest = max(highest, self._bands[stage.key][1])
        if self.current.key not in self._done:
            low, high = self._bands[self.current.key]
            highest = max(highest, low + (high - low) * self.fraction())
        return max(0.0, min(100.0, highest))

    def _remember(self, value: float) -> None:
        """Record a (time, percent) sample for the rate estimate.

        Sampling happens here rather than in a separate tick call so a caller cannot
        forget it. The interval guard keeps it cheap however often this is read.
        """
        now = self._clock()
        if self._samples and now - self._samples[-1][0] < ETA_SAMPLE_SECONDS:
            return
        self._samples.append((now, value))

    def rate(self) -> float | None:
        """Percent gained per second across the recent window, or None if unmeasurable.

        A window in which nothing moved carries no information about the remaining
        work, so it returns None rather than a misleading zero or infinity.
        """
        if len(self._samples) < 2:
            return None
        cutoff = self._clock() - ETA_WINDOW_SECONDS
        recent = [sample for sample in self._samples if sample[0] >= cutoff]
        if len(recent) < 2:
            return None
        (first_time, first_value), (last_time, last_value) = recent[0], recent[-1]
        elapsed = last_time - first_time
        gained = last_value - first_value
        if elapsed <= 0 or gained <= 0.0:
            return None
        return gained / elapsed

    def _predicted_remaining(self) -> float | None:
        remaining = 0.0
        known = False
        for index, stage in enumerate(self.stages):
            if stage.key in self._done:
                continue
            expected = self._expected.get(stage.key)
            if not expected:
                continue
            known = True
            if index == self._index:
                remaining += expected * (1.0 - self.fraction())
            else:
                remaining += expected
        return remaining if known else None

    def eta(self) -> float | None:
        """Seconds remaining, or None while there is nothing to base it on.

        The measured rate is preferred because it reflects the machine as it is now;
        predictions from previous runs cover the opening stretch, where a rate cannot
        be measured yet.
        """
        measured = self.rate()
        if measured is not None:
            return max(0.0, (100.0 - self.percent()) / measured)
        return self._predicted_remaining()

    def predicted_total(self) -> float | None:
        """Total predicted seconds, when every remaining stage has a prediction."""
        known = {key: value for key, value in self._expected.items() if value > 0}
        if not known:
            return None
        return sum(known.get(stage.key, 0.0) for stage in self.stages) or None
