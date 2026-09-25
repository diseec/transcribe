"""Learned per-stage durations, so progress and ETAs are predictable from run two.

Weights baked into the code can only ever be a guess. This records how long each
stage actually took, as a rate per second of audio, so a prediction follows the
machine and the chosen profile instead of a constant. Rates (not raw seconds)
mean a measurement from a 10-minute file still predicts a 2-hour one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from whisperx_local.paths import STATE_DIR

TIMING_FILE = STATE_DIR / "timings.json"

# Weight given to the newest measurement. 0.5 forgets the past quickly enough to
# adapt to a changed machine, slowly enough that one odd run cannot dominate.
DEFAULT_ALPHA = 0.5

# Bumped when the meaning of a stored rate changes, so old files are ignored
# instead of silently mispredicting.
SCHEMA = "timing-v1"

# Below this, a measurement is noise rather than a duration worth learning from.
MIN_AUDIO_SECONDS = 1.0


@dataclass
class TimingStore:
    """Persistent per-context stage rates, updated by exponential moving average."""

    path: Path = TIMING_FILE
    alpha: float = DEFAULT_ALPHA

    def load(self) -> dict[str, dict[str, float]]:
        """Read stored rates, tolerating a missing or damaged file."""
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
            return {}
        contexts = payload.get("contexts")
        if not isinstance(contexts, dict):
            return {}
        clean: dict[str, dict[str, float]] = {}
        for context, rates in contexts.items():
            if isinstance(rates, dict):
                clean[str(context)] = {
                    str(stage): float(rate)
                    for stage, rate in rates.items()
                    if isinstance(rate, (int, float)) and rate > 0
                }
        return clean

    def predict(self, context: str, audio_seconds: float) -> dict[str, float]:
        """Predicted seconds per stage for a recording of this length."""
        if audio_seconds <= 0:
            return {}
        rates = self.load().get(context, {})
        return {stage: rate * audio_seconds for stage, rate in rates.items()}

    def record(
        self,
        context: str,
        *,
        audio_seconds: float,
        measured: dict[str, float],
    ) -> dict[str, float]:
        """Fold one run's measurements in and persist the result.

        Returns the updated rates for this context.
        """
        if audio_seconds < MIN_AUDIO_SECONDS:
            return self.load().get(context, {})
        contexts = self.load()
        rates = dict(contexts.get(context, {}))
        for stage, seconds in measured.items():
            if seconds <= 0:
                continue
            observed = seconds / audio_seconds
            previous = rates.get(stage)
            rates[stage] = observed if previous is None else (
                (1.0 - self.alpha) * previous + self.alpha * observed
            )
        contexts[context] = rates
        self._write(contexts)
        return rates

    def _write(self, contexts: dict[str, dict[str, float]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": SCHEMA, "contexts": contexts}
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def forget(self) -> None:
        """Drop all learned timings, e.g. after a machine or settings change."""
        self.path.unlink(missing_ok=True)


def context_key(*, profile: str, model: str, language: str, device: str) -> str:
    """Identify the settings that change how long the work takes.

    Deliberately excludes per-file options (hotwords, chunk length and so on):
    those affect text, not speed in any way worth splitting the history over.
    """
    return f"{profile}|{model}|{language}|{device}"
