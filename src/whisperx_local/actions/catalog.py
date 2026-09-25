"""The action set: one entry per thing this tool can do to a recording.

Declaring them as data rather than as branches in a pipeline is what makes "any
subset" expressible. The list is ordered by dependency, so the planner can rely on
position to mean "after".

Each action carries the options that belong to it, which is how the command line and
the interactive composer can both offer the right choices at the right moment without
a second list to keep in step.
"""

from __future__ import annotations

from whisperx_local.actions.base import (
    AUDIO,
    FILES,
    PLAN,
    REPORT,
    SEGMENTS,
    SOURCE,
    TURNS,
    WORDS,
    Action,
    Option,
)

PREPARE = Action(
    key="prepare",
    label="Preparing audio",
    summary="Build the levelled mono track the recogniser reads.",
    provides=(AUDIO,),
    requires=(SOURCE,),
    options=(
        Option("normalize", bool, "Level quiet speech before recognition", examples=("true", "false")),
    ),
)

BOUNDARIES = Action(
    key="boundaries",
    label="Finding speech",
    summary="Decide where chunks may be cut without splitting a word.",
    provides=(PLAN,),
    requires=(AUDIO,),
    options=(
        Option("silence_split", bool, "Snap boundaries into pauses", examples=("true", "false")),
        Option("silence_min", float, "Shortest pause to cut on, seconds"),
        Option("chunk_seconds", float, "Target audio per chunk, seconds"),
        Option("chunk_overlap", float, "Overlap between fixed chunks, seconds"),
    ),
)

TRANSCRIBE = Action(
    key="transcribe",
    label="Transcribing",
    summary="Recognise speech, chunk by chunk, saving each part as it finishes.",
    provides=(SEGMENTS,),
    requires=(AUDIO, PLAN),
    options=(
        Option("language", str, "Spoken language", choices=("fa", "en")),
        Option("model", str, "Recognition model", advanced=True),
        Option(
            "engine", str, "Which recogniser runs",
            choices=("whispercpp", "whisperx"),
        ),
        Option("compute_type", str, "Numeric precision", choices=("int8", "float32"), advanced=True),
        Option("beam_size", int, "Beam width", advanced=True),
        Option("batch_size", int, "Batched inference size", advanced=True),
        Option("threads", int, "CPU threads", advanced=True),
        Option("vad_method", str, "Voice activity detector", choices=("pyannote", "silero"), advanced=True),
        Option("hotwords", str, "Terms to bias recognition toward, comma separated"),
        Option("prompt", str, "Context sentence that steers vocabulary"),
        Option("retries", int, "Recognition attempts per chunk", advanced=True),
        Option(
            "chunks_per_call", int,
            "Consecutive chunks covered by one model load; measured neutral, 1 disables",
            advanced=True,
        ),
        Option("vad_onset", float, "Speech probability to start a region", advanced=True),
        Option("vad_offset", float, "Speech probability to end a region", advanced=True),
        Option("chunk_size", int, "Voice-detection window merge size, seconds", advanced=True),
        Option(
            "compression_ratio_threshold", float,
            "Reject a segment above this repetition ratio", advanced=True,
        ),
        Option("logprob_threshold", float, "Reject a segment below this confidence", advanced=True),
        Option("no_speech_threshold", float, "Treat a region as silence above this", advanced=True),
    ),
)

ALIGN = Action(
    key="align",
    label="Aligning words",
    summary="Refine segment timings to the word level.",
    provides=(WORDS,),
    requires=(SEGMENTS,),
    options=(Option("align_retries", int, "Alignment attempts per chunk", advanced=True),),
)

DIARIZE = Action(
    key="diarize",
    label="Identifying speakers",
    summary="Label who spoke when, and attach those labels to the words.",
    provides=(TURNS,),
    requires=(AUDIO, SEGMENTS),
    options=(
        Option("speakers", int, "Exact speaker count, when known"),
        Option("diarize_audio", str, "Audio used for separation", choices=("gentle", "original")),
        Option("min_turn", float, "Shortest speaker turn to keep, seconds", advanced=True),
        Option("diarize_retries", int, "Separation attempts", advanced=True),
    ),
)

EXPORT = Action(
    key="export",
    label="Writing transcript",
    summary="Render the transcript in each requested format.",
    provides=(FILES,),
    requires=(SEGMENTS,),
    options=(
        Option("output_format", str, "Transcript format", choices=("all", "txt", "srt", "vtt", "tsv", "json", "aud")),
        Option("output_dir", str, "Where transcripts are written"),
        Option(
            "copy_beside_input", bool,
            "Also save a copy next to the recording; not from input/",
        ),
    ),
)

ANALYZE = Action(
    key="analyze",
    label="Measuring quality",
    summary="Report loudness, pause structure, coverage and speaking time.",
    provides=(REPORT,),
    requires=(SEGMENTS,),
    always_available=True,
)

# Dependency order. The planner treats position as "runs after", so this tuple is the
# single place that decides stage order.
ACTIONS: tuple[Action, ...] = (
    PREPARE,
    BOUNDARIES,
    TRANSCRIBE,
    ALIGN,
    DIARIZE,
    EXPORT,
    ANALYZE,
)

BY_KEY: dict[str, Action] = {action.key: action for action in ACTIONS}

# What a plain `run` does.
#
# Recognition, timing, and the file on disk -- nothing that can only take away. Two
# stages a plain run does not do, for different reasons:
#
# * Analysis is a question, not a step. It is reported when asked for.
# * Word alignment and speaker separation are refinements. Both are worth having and
#   both can fail on their own, and neither adds a single recognised word. Asking for
#   them explicitly keeps the default path short, fast, and impossible to make worse.
#
# Nothing here cleans the text. The transcript is what was recognised, cut where
# recognition cut it; see ``transcript.assemble.prepare_lines``.
DEFAULT_ACTIONS: tuple[str, ...] = (
    "prepare",
    "boundaries",
    "transcribe",
    "export",
)


def keys() -> tuple[str, ...]:
    return tuple(action.key for action in ACTIONS)


def get(key: str) -> Action:
    try:
        return BY_KEY[key]
    except KeyError:
        raise KeyError(
            f"unknown action {key!r}. Known actions: {', '.join(keys())}"
        ) from None


def ordered(selection) -> tuple[Action, ...]:
    """The selected actions, in dependency order and without duplicates."""
    wanted = set(selection)
    unknown = wanted - set(BY_KEY)
    if unknown:
        raise KeyError(
            f"unknown action(s): {', '.join(sorted(unknown))}. "
            f"Known actions: {', '.join(keys())}"
        )
    return tuple(action for action in ACTIONS if action.key in wanted)


def options_for(selection) -> tuple[Option, ...]:
    """Every option belonging to the selected actions, deduplicated by name."""
    seen: dict[str, Option] = {}
    for action in ordered(selection):
        for option in action.options:
            seen.setdefault(option.name, option)
    return tuple(seen.values())
