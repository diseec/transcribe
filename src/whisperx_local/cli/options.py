"""The command line, built from the same declarations the composer uses.

The flags are written out rather than generated from the action catalog: an explicit
parser gives better help text and better errors, and generator bugs are harder to see
than a missing line. What keeps the two in step is a test asserting that every option
the catalog offers is reachable as a flag, so a new option cannot be added to the
composer and forgotten here.

Options a profile may fill default to None, which is how "the user did not choose this"
is expressed. They are filled from the profile before anything reads them.
"""

from __future__ import annotations

import argparse

from whisperx_local.actions import catalog
from whisperx_local.actions.plan import parse_only
from whisperx_local.chunking.batching import DEFAULT_CHUNKS_PER_CALL
from whisperx_local.compare.transcripts import DEFAULT_WINDOW
from whisperx_local.config import Preferences
from whisperx_local.engine import DEFAULT_PROFILE, PROFILES
from whisperx_local.engine import backends

PROG = "whisperx"

# The one-shot command's only variable. Persian because that is what this installation
# exists to transcribe; everything else about that command is fixed in its handler.
SIMPLE_LANGUAGE = "fa"


def action_list(text: str) -> str:
    """Validate an action selection while parsing, so a typo fails immediately.

    Deferring this to the planner would push the complaint into a run that has
    already started, which is a much worse place to discover a spelling mistake.
    """
    names = parse_only(text)
    known = set(catalog.keys())
    unknown = [name for name in names if name not in known]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown action(s): {', '.join(unknown)}. Known: {', '.join(catalog.keys())}"
        )
    return text


def build_parser() -> argparse.ArgumentParser:
    """The parser, without any user preferences applied.

    Kept preference-free so tests are unaffected by whatever the user has saved;
    :func:`parser_with_preferences` is what the app itself uses.
    """
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Local transcription for Persian and English recordings. "
            "Run with no command to be shown a menu."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    menu = subparsers.add_parser("menu", help="Choose a recording and settings (the default)")
    menu.set_defaults(handler="menu")

    composer = subparsers.add_parser(
        "composer", help="Set up a request in any order, without a wizard"
    )
    composer.set_defaults(handler="composer")

    compare = subparsers.add_parser(
        "compare", help="Measure one transcript against another"
    )
    compare.add_argument("reference", help="The transcript to measure against")
    compare.add_argument("other", help="The transcript to measure")
    compare.add_argument(
        "--window", type=float, default=DEFAULT_WINDOW,
        help="Seconds per comparison window",
    )
    compare.add_argument("--worst", type=int, default=5, help="How many worst windows to show")
    compare.add_argument(
        "--term", action="append", metavar="WORD",
        help="A term to track; repeat to replace the default vocabulary",
    )
    compare.add_argument(
        "--keep-zwnj", action="store_true",
        help="Treat the zero-width non-joiner as significant",
    )
    compare.add_argument(
        "--distinct-letters", action="store_true",
        help="Also fold آ to ا and the hamza carriers, at the risk of hiding a real mistake",
    )
    compare.add_argument(
        "--keep-punctuation", action="store_true", help="Do not fold punctuation away"
    )
    compare.set_defaults(handler="compare")

    run = subparsers.add_parser("run", help="Process one recording")
    _add_run_arguments(run)
    run.set_defaults(handler="run")

    # A path in, text out. Given no settings at all on purpose: the handler pins every one
    # of them, so there is nothing to learn and nothing saved can change the result.
    simple = subparsers.add_parser(
        "transcribe",
        help="Transcribe recordings with fixed settings -- for scripts and agents",
    )
    simple.add_argument("files", nargs="+", help="One or more recordings")
    simple.add_argument(
        "--language", default=SIMPLE_LANGUAGE,
        help=f"Spoken language (fixed default: {SIMPLE_LANGUAGE})",
    )
    simple.set_defaults(handler="transcribe")

    status = subparsers.add_parser("status", help="Prerequisites, paths and caches")
    status.set_defaults(handler="status")

    install = subparsers.add_parser("install", help="Create .venv and install WhisperX")
    install.set_defaults(handler="install")

    resources = subparsers.add_parser("resources", help="Download the alignment tokenizer")
    resources.set_defaults(handler="resources")

    tests = subparsers.add_parser("selftest", help="Run the test suite")
    tests.set_defaults(handler="selftest")

    config = subparsers.add_parser("config", help="Show or change saved defaults")
    config.add_argument(
        "action", nargs="?", choices=("show", "set", "unset", "reset"), default="show"
    )
    config.add_argument("name", nargs="?")
    config.add_argument("value", nargs="?")
    config.set_defaults(handler="config")

    preset = subparsers.add_parser("preset", help="Save, list or delete bundles of settings")
    preset.add_argument(
        "action", nargs="?", choices=("list", "save", "delete", "show"), default="list"
    )
    preset.add_argument("name", nargs="?")
    preset.set_defaults(handler="preset")

    return parser


def _add_run_arguments(run: argparse.ArgumentParser) -> None:
    run.add_argument("audio", help="The recording to process")

    selection = run.add_argument_group("what to do")
    selection.add_argument(
        "--only",
        type=action_list,
        help=f"Run exactly these, comma separated. Known: {', '.join(catalog.keys())}",
    )
    selection.add_argument("--without", type=action_list, help="Skip these, comma separated")
    selection.add_argument("--analyze", action="store_true", help="Also report statistics")
    selection.add_argument("--dry-run", action="store_true", help="Show the plan and stop")
    selection.add_argument("--force", action="store_true", help="Discard cached work first")
    selection.add_argument("--preset", help="Apply a saved bundle; flags still win")

    quality = run.add_argument_group("quality and speed")
    quality.add_argument("--profile", choices=tuple(PROFILES), default=DEFAULT_PROFILE)
    quality.add_argument("--language", choices=("fa", "en"), default="fa")
    quality.add_argument("--model", default=None)
    quality.add_argument(
        # Defaulted rather than left as None, unlike the profile-filled options. ``engine``
        # is part of the recognition signature, and None versus the name of the default
        # engine would be two spellings of one behaviour -- so choosing the default
        # explicitly would throw away cached text for no reason.
        "--engine", choices=tuple(backends.BACKENDS), default=backends.DEFAULT_BACKEND,
        help="Which recogniser runs; whispercpp needs its binary and a GGML model",
    )
    quality.add_argument("--compute-type", choices=("int8", "float32"), default=None)
    quality.add_argument("--beam-size", type=int, default=None)
    quality.add_argument("--best-of", type=int, default=None)
    quality.add_argument("--patience", type=float, default=None)
    quality.add_argument("--batch-size", type=int, default=None, help="Batched inference; >1 is much faster")
    quality.add_argument("--threads", type=int, default=None, help="Defaults to the performance-core count")
    quality.add_argument("--vad-method", choices=("pyannote", "silero"), default=None)
    quality.add_argument(
        "--chunks-per-call", type=int, default=DEFAULT_CHUNKS_PER_CALL,
        help="Consecutive chunks covered by one model load; 1 disables batching",
    )
    quality.add_argument("--hotwords", help="Comma-separated terms to bias recognition toward")
    quality.add_argument("--prompt", help="Context sentence that steers vocabulary")

    decoding = run.add_argument_group("decoding and voice detection")
    decoding.add_argument("--vad-onset", type=float, default=0.10, help="Speech probability to start a region")
    decoding.add_argument("--vad-offset", type=float, default=0.05, help="Speech probability to end a region")
    decoding.add_argument("--chunk-size", type=int, default=8, help="VAD window merge size in seconds")
    decoding.add_argument("--compression-ratio-threshold", type=float, default=3.0)
    decoding.add_argument("--logprob-threshold", type=float, default=-2.0)
    decoding.add_argument("--no-speech-threshold", type=float, default=0.95)

    audio = run.add_argument_group("audio and chunking")
    audio.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    audio.add_argument("--silence-split", action=argparse.BooleanOptionalAction, default=True,
                       help="Snap boundaries into pauses so no word is cut")
    audio.add_argument("--silence-min", type=float, default=0.5, help="Shortest pause to cut on")
    audio.add_argument("--chunk-seconds", type=float, default=600.0, help="Target audio per chunk")
    audio.add_argument("--chunk-overlap", type=float, default=2.0)

    voices = run.add_argument_group("speakers")
    voices.add_argument("--diarize", action=argparse.BooleanOptionalAction, default=None,
                        help="Label speakers; on by default when a token is configured")
    voices.add_argument("--speakers", type=int, help="Exact speaker count when you know it")
    voices.add_argument("--min-turn", type=float, default=0.35)
    voices.add_argument("--diarize-audio", choices=("gentle", "original"), default="gentle")

    output = run.add_argument_group("output")
    output.add_argument(
        "--output-format",
        choices=("all", "txt", "srt", "vtt", "tsv", "json", "aud"),
        default="txt",
    )
    output.add_argument("--output-dir")
    output.add_argument(
        "--copy-beside-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also save a copy of the transcript beside the recording, except when the "
            "recording is already in input/"
        ),
    )
    output.add_argument("--network", choices=("auto", "offline", "online"), default="auto")

    retries = run.add_argument_group("retries")
    retries.add_argument("--retries", type=int, default=3)
    retries.add_argument("--align-retries", type=int, default=3)
    retries.add_argument("--diarize-retries", type=int, default=2)


def parser_with_preferences(preset: str | None = None, overrides: dict | None = None):
    """The parser with saved defaults already applied.

    Preferences become parser defaults, so a flag beats a preference and a preference
    beats the built-in value. A preset is layered on top of the preferences without
    being written to disk, and explicit overrides (from the composer) on top of that.
    """
    parser = build_parser()
    preferences = Preferences.load()
    values = dict(preferences.values)
    if preset:
        values.update(preferences.preset(preset))
    if overrides:
        values.update(overrides)
    preferences.apply_to(parser, values)
    return parser


# A name that cannot be a real recording, only there to satisfy the positional
# argument while the parser's defaults are read.
PLACEHOLDER = "unused.m4a"


def default_values() -> dict[str, object]:
    """Every option's built-in value, read from the parser that declares it.

    From the parser rather than from the settings table, deliberately. The table holds
    only the settings a user may *save*; options such as ``vad_onset`` are real and
    adjustable without being savable, and looking one of those up in the table is what
    made ``show`` raise ``KeyError: 'vad_onset'`` instead of printing a value.
    """
    return vars(build_parser().parse_args(["run", PLACEHOLDER]))


def effective_values(
    preferences: Preferences, *, preset: str | None = None, overrides: dict | None = None
) -> dict[str, object]:
    """What every option is set to right now, across the layers that decide it.

    Follows the same order a real run does -- built-in, then saved, then preset, then
    this session's own choices -- so a screen showing these values is showing what
    would actually be used rather than a plausible reconstruction of it.
    """
    values = default_values()
    values.update(preferences.values)
    if preset:
        values.update(preferences.preset(preset))
    if overrides:
        values.update(overrides)
    return values
