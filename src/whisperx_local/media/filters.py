"""Audio levels and the ffmpeg filter chains.

These constants are the result of measurement, not taste, so they are separated from
the code that applies them: a change here changes what the model hears, and the
comments record what was compared.
"""

from __future__ import annotations

SAMPLE_RATE = 16000

# Recognition track. Chosen by measurement on real meetings: this chain recovered
# 107 words where the previous one recovered 83, and beat two more aggressive
# variants (93 and 94 words). speechnorm lifts quiet speech before dynaudnorm
# evens out the level, and the highpass removes rumble that the model cannot use.
RECOGNITION_FILTER = (
    "highpass=f=55,"
    "lowpass=f=7800,"
    "speechnorm=e=12.5:r=0.0001:l=1,"
    "dynaudnorm=f=300:g=20:p=0.9:m=15,"
    "loudnorm=I=-15:LRA=6:TP=-1.5"
)

# Speaker track. Loudness only, and deliberately without dynaudnorm: heavy
# short-term gain changes blur the voice cues pyannote matches on, so levelling the
# speaker track the way the recognition track is levelled costs speaker accuracy.
SPEAKER_FILTER = "highpass=f=55,loudnorm=I=-20:LRA=11:TP=-1.5"

# Length of one loudness measurement for the boundary scan. Short enough to find a
# calm instant inside a sentence, long enough not to be dominated by a single
# glottal stop. 0.25s at 16 kHz is 4000 samples per window.
ENERGY_WINDOW = 0.25

# Stands in for a digitally silent window, because -inf does not survive arithmetic.
SILENT_DB = -200.0


def energy_filter(window_seconds: float = ENERGY_WINDOW) -> str:
    """Emit one RMS reading per fixed-length window, as filter metadata.

    ``asetnsamples`` groups the stream into frames and ``astats`` with ``reset=1``
    reports each group separately, which turns a continuous stream into a series of
    measurements a planner can reason about.
    """
    samples = max(1, int(round(SAMPLE_RATE * window_seconds)))
    return (
        f"aresample={SAMPLE_RATE},"
        f"asetnsamples=n={samples},"
        "astats=metadata=1:reset=1,"
        "ametadata=mode=print:file=-:key=lavfi.astats.Overall.RMS_level"
    )
