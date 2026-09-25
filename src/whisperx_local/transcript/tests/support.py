"""Shared fixtures for the transcript tests.

The raw turns are real pyannote output from a recording, kept because the awkward
parts -- sub-100ms flickers, overlapping turns, out-of-order starts -- are what the
cleaning rules exist for, and synthetic data does not reproduce them.
"""

from __future__ import annotations

from whisperx_local.transcript.turns import Turn

# Note the flickers: 4.35-4.37 is 20ms, 24.13-24.15 is 20ms, and several turns
# overlap or start before the previous one ended.
RAW_TURNS = [
    (1.55, 2.83, "SPEAKER_02"),
    (3.10, 4.35, "SPEAKER_01"),
    (4.35, 4.37, "SPEAKER_00"),
    (6.66, 7.66, "SPEAKER_02"),
    (7.15, 8.27, "SPEAKER_01"),
    (20.01, 27.12, "SPEAKER_01"),
    (24.13, 24.15, "SPEAKER_02"),
    (24.15, 25.61, "SPEAKER_00"),
    (55.68, 60.97, "SPEAKER_02"),
    (59.62, 60.12, "SPEAKER_01"),
]


def turns_from(raw) -> list[Turn]:
    return [Turn(start, end, speaker) for start, end, speaker in raw]


def cleaned_raw() -> list[Turn]:
    from whisperx_local.transcript.turns import clean_turns

    return clean_turns(turns_from(RAW_TURNS))


# One segment holding four words that fall in four different turns, which is the case
# WhisperX's own writer gets wrong: it would label all four with the majority speaker.
SPLIT_SEGMENTS = [
    {
        "start": 1.5,
        "end": 12.0,
        "text": "a b c",
        "words": [
            {"word": "یک", "start": 1.6, "end": 2.0},
            {"word": "دو", "start": 2.0, "end": 2.5},
            {"word": "سه", "start": 6.7, "end": 7.1},
            {"word": "چهار", "start": 25.0, "end": 25.4},
        ],
    }
]
