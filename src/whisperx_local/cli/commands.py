"""Handlers for every command, including the one that runs a plan.

The run handler is the only place that assembles a full request: resolve a recording,
decide what already exists, work out the smallest plan, then hand it to the runner. It
deliberately knows nothing about stages beyond that, which is what lets the command
line accept `--only export` and have it mean exactly that.
"""

from __future__ import annotations

import argparse
import getpass
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from whisperx_local.actions import catalog, plan as planning
from whisperx_local.actions.plan import Selection
from whisperx_local.config import Preferences, describe
from whisperx_local.engine import apply as apply_profile
from whisperx_local.engine import describe as describe_profile
from whisperx_local.engine import PROFILES
from whisperx_local.media import has_audio_stream, probe_duration
from whisperx_local.paths import (
    APP_DIR,
    DIARIZATION_READY,
    NLTK_DATA_DIR,
    PYTHON,
    SUPPORTED_PYTHON,
    VENV_DIR,
    WHISPERX,
    host_architecture,
    load_local_env,
    managed_env,
    nltk_resource_available,
    performance_cores,
    available_memory_bytes,
    require_install,
    supported_host,
    working_ffmpeg,
    working_ffprobe,
)
from whisperx_local.services import (
    PIPELINE_VERSION,
    WorkSpace,
    core_models_cached,
    describe_plan,
    diarization_ready,
    execute,
    is_complete,
    marker_path,
)
from whisperx_local.ui import AccuracyStudio, clock, print_notice, print_plan, print_report

FORMATS = ("txt", "srt", "vtt", "tsv", "json", "aud")

# Exit status for a transcript that was produced but does not cover the recording.
# Distinct from 1, which is a run that could not start or crashed outright.
INCOMPLETE_STATUS = 2

# Exit status for a run that finished and wrote no transcript at all. Distinct from
# INCOMPLETE, which at least leaves something to read.
NOTHING_WRITTEN_STATUS = 3

# Exit status for a comparison that had nothing to compare -- one side held no
# recognised text -- so no conclusion may be drawn from it.
NOT_COMPARABLE_STATUS = 4


# -- helpers --------------------------------------------------------------


def heading(message: str) -> None:
    print(f"\n◆ {message}", flush=True)


def detail(label: object, value: object) -> None:
    print(f"  {str(label):<20} {value}", flush=True)


def _effective_diarization(options) -> bool:
    """Speaker separation is on unless the user said otherwise or no token exists."""
    if options.diarize is not None:
        return bool(options.diarize)
    return bool(os.environ.get("HF_TOKEN"))


def _network_policy(options, diarize: bool) -> str:
    """Cache-only once everything needed is already downloaded."""
    if options.network != "auto":
        return options.network
    if not core_models_cached(options.model, options.language):
        return "online"
    if diarize and not diarization_ready():
        return "online"
    return "offline"


def _require_ready(options, network: str) -> None:
    """Refuse to begin work that cannot finish, before any of it is done.

    Two failures are worth catching here rather than an hour later, both of which have
    happened for real:

    * A recogniser that is not installed at all, so the run dies on its first chunk. An
      ``--only transcribe`` request that never writes a transcript is exactly what the
      end-of-run reporting exists to prevent, and this stops it earlier still.
    * A model that has to be downloaded first. That is legitimate, but it can be an hour
      of network on a slow connection, and discovering it by watching a stalled progress
      bar is not a reasonable way to find out. It is stated up front, and refused outright
      when the network is pinned to cache-only, because then the download cannot happen at
      all and the run would sit there rather than fail.
    """
    from whisperx_local.engine import backends

    backend = backends.for_options(options)
    if not backend.available():
        raise SystemExit(f"{backend.label} cannot run: {backend.missing_reason()}")

    # A setting the engine cannot honour is reported here rather than dropped quietly: a
    # comparison taken from a run that ignored a setting measures two things at once.
    for note in backend.notes(options):
        print(f"  ! {note}", flush=True)

    if backend.key != backends.HF_CACHED_BACKEND:
        # Whichever engine is chosen owns its own model; the cache check below is about
        # the Hugging Face one only.
        if backend.model_path(options) is None:
            raise SystemExit(f"{backend.label} has no model: {backend.model_reason()}")
        return

    if core_models_cached(options.model, options.language):
        return
    if network == "offline":
        raise SystemExit(
            f"{options.model} is not in the model cache, and the network is set to "
            "offline so it cannot be fetched. Run once with --network online, or name a "
            "model that is already cached."
        )
    print(
        f"  ! {options.model} is not cached: this will download it first. On a slow or "
        "interrupted connection that can take a long time, and it is fetched again from "
        "wherever it stopped.",
        flush=True,
    )


def _token(options) -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        return getpass.getpass("Hugging Face read token (input hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def selection_from(options) -> Selection:
    """The requested action set, including the ones settings have ruled out.

    Speaker separation is excluded here rather than inside the runner, because a stage
    that was switched off must not appear in the plan at all: leaving it in meant
    ``--no-diarize`` still ran it, and the run reported speakers it was told to skip.
    """
    only = planning.parse_only(options.only) if options.only else ()
    without = list(planning.parse_only(options.without)) if options.without else []
    if not getattr(options, "diarize", None) and "diarize" not in without:
        without.append("diarize")
    extra = ("analyze",) if getattr(options, "analyze", False) else ()
    return Selection(only=only, without=tuple(without), extra=extra)


# -- the run --------------------------------------------------------------


# The one-shot command's fixed settings. Pinned here rather than resolved from the
# settings file, because the value of that command is that it behaves the same for every
# user and on every machine. An agent should not have to know any of this, and should not
# be able to inherit a human's last experiment either.
SIMPLE_BEAM_SIZE = 5
SIMPLE_COMPUTE_TYPE = "int8"
SIMPLE_OUTPUT_FORMAT = "txt"
SIMPLE_VAD = "silero"
SIMPLE_SILENCE_MIN = 0.5
SIMPLE_CHUNK_SECONDS = 600.0
SIMPLE_CHUNK_OVERLAP = 2.0
SIMPLE_RETRIES = 3


def transcribe_command(args) -> None:
    """A path in, text out -- with nothing read from the user's saved settings.

    ``./cli /path/to/file`` lands here. Every setting that can change the output is pinned
    in :func:`_simple_options`, so the result does not depend on which profile someone
    last chose, and the only flag that exists is the language.
    """
    require_install()
    load_local_env()
    for name in args.files:
        source = Path(name).expanduser().resolve()
        if not source.is_file():
            raise SystemExit(f"No such recording: {source}")
        _run_recording(
            source, _simple_options(source, language=args.language), dry_run=False
        )


def _simple_options(source: Path, *, language: str):
    """The fixed settings, built from the preference-free parser.

    ``whisper.cpp`` when it can run, because it measured several times faster on the same
    audio; WhisperX otherwise, announced rather than assumed, so the command works on a
    machine that has never installed anything else.
    """
    from whisperx_local.cli.options import build_parser
    from whisperx_local.engine import backends
    from whisperx_local.paths import OUTPUT_DIR

    options = build_parser().parse_args(["run", str(source)])
    whispercpp = backends.get("whispercpp")
    ready = whispercpp.available() and bool(whispercpp.installed())
    options.engine = "whispercpp" if ready else "whisperx"
    if not ready:
        why = whispercpp.missing_reason() if not whispercpp.available() else whispercpp.model_reason()
        print(f"  ! whisper.cpp is not ready, so WhisperX runs instead: {why}", flush=True)
    options.model = "automatic"
    options.language = language
    options.preset = None
    options.compute_type = SIMPLE_COMPUTE_TYPE
    options.beam_size = SIMPLE_BEAM_SIZE
    options.vad_method = SIMPLE_VAD
    options.normalize = True
    options.silence_split = True
    options.silence_min = SIMPLE_SILENCE_MIN
    options.chunk_seconds = SIMPLE_CHUNK_SECONDS
    options.chunk_overlap = SIMPLE_CHUNK_OVERLAP
    options.retries = SIMPLE_RETRIES
    options.hotwords = None
    options.prompt = None
    options.output_format = SIMPLE_OUTPUT_FORMAT
    options.output_dir = str(OUTPUT_DIR)
    # A copy beside the recording as well, because an agent that has just handed over a
    # path should be able to find the text where it put the recording.
    options.copy_beside_input = True
    # Export as well as recognise: an agent asking for text should get text, not a raw
    # copy it has to convert first.
    options.only = "transcribe,export"
    options.without = None
    options.diarize = False
    options.analyze = False
    options.dry_run = False
    options.argv = None
    return options


def run_command(args: argparse.Namespace) -> None:
    """Resolve options, then process one recording.

    The options are resolved by re-parsing the original command line with preferences
    as defaults, rather than by copying values around. Re-parsing the real tokens keeps
    nuances such as ``--no-normalize`` intact, which a hand-built argument list loses.
    """
    from whisperx_local.cli.options import parser_with_preferences

    if not args.dry_run:
        require_install()
    load_local_env()

    tokens = getattr(args, "argv", None)
    if tokens:
        options = parser_with_preferences(
            getattr(args, "preset", None), getattr(args, "overrides", None)
        ).parse_args(tokens)
    else:
        options = args

    source = Path(options.audio).expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"No such recording: {source}")
    _run_recording(source, options, dry_run=args.dry_run)


def _options_for_composer(source: Path, selection: Selection, overrides: dict, preset: str | None, dry_run: bool):
    """A complete options namespace for a recording chosen in the composer.

    Composer overrides are explicit choices, so they are applied directly instead of
    being treated as defaults.
    """
    from whisperx_local.cli.options import build_parser

    options = build_parser().parse_args(["run", str(source)])
    resolved = Preferences.load().resolve(preset=preset, cli=overrides)
    for name in overrides:
        # Only settings exist in ``resolved``; an action is not a setting, so an unknown
        # key here means the caller sent an action by mistake rather than a typo.
        if name in resolved and hasattr(options, name):
            setattr(options, name, resolved[name])
    options.only = ",".join(selection.only) or None
    options.without = ",".join(selection.without) or None
    options.analyze = "analyze" in set(selection.extra) | set(selection.only)
    options.dry_run = dry_run
    options.preset = preset
    return options


def _run_recording(source: Path, options, *, dry_run: bool):
    """The shared core: resolve the profile, then execute."""
    profile, notes = apply_profile(options)
    workspace = WorkSpace(source=source, output_dir=_output_dir(options))
    return _execute_recording(options, workspace, profile, notes, dry_run=dry_run)


def _output_dir(options) -> Path:
    from whisperx_local.paths import OUTPUT_DIR

    if getattr(options, "output_dir", None):
        return Path(options.output_dir).expanduser().resolve()
    return OUTPUT_DIR


def _execute_recording(options, workspace, profile, notes, *, dry_run: bool):
    """Build the plan, show it, and run it."""
    ffmpeg = working_ffmpeg()
    ffprobe = working_ffprobe(ffmpeg)
    env = managed_env()

    if not nltk_resource_available():
        raise SystemExit(f"Missing the alignment tokenizer. Run: {APP_NAME} resources")

    if not has_audio_stream(ffprobe, workspace.source, env=env):
        raise SystemExit(
            f"{workspace.source.name} has no audio track, so there is nothing to transcribe."
        )

    diarize = _effective_diarization(options)
    if diarize:
        token = _token(options)
        if token:
            os.environ["HF_TOKEN"] = token
        else:
            diarize = False
    options.diarize = diarize
    network = _network_policy(options, diarize)
    _require_ready(options, network)

    duration = probe_duration(ffprobe, workspace.source, env=env)
    if options.force:
        # The marker lives outside the work directory, so clearing the workspace is not
        # enough to actually force a rerun.
        workspace.reset()
        marker_path(workspace.source).unlink(missing_ok=True)

    # Before asking what is already on disk, because that answer decides what runs: cached
    # text from a different model or precision would otherwise count as finished work and
    # be rendered verbatim by an export-only run.
    from whisperx_local.actions.base import SEGMENTS, WORDS
    from whisperx_local.engine.commands import recognition_signature

    signature = recognition_signature(options)
    stale = workspace.stale_recognition(signature)
    if stale:
        if dry_run:
            # Report it, change nothing, and plan as though it were already gone, so the
            # preview matches the run that would follow it.
            print(
                "  ! recognition settings changed: cached text is stale and will be discarded",
                flush=True,
            )
        else:
            removed = workspace.discard_stale_recognition(signature)
            print(
                f"  ! recognition settings changed: {removed} cached result(s) discarded",
                flush=True,
            )

    chunks = workspace.stored_chunks(
        duration,
        chunk_seconds=options.chunk_seconds,
        overlap_seconds=options.chunk_overlap,
    )
    present = workspace.available(
        chunks, speakers=options.speakers, output_format=options.output_format
    )
    if stale:
        present -= {SEGMENTS, WORDS}
    plan = planning.resolve(selection_from(options), available=present)

    primary = workspace.primary_output(workspace.source.stem, options.output_format)
    print_plan(
        [
            ("Recording", f"{workspace.source.name}  ({clock(duration)})"),
            ("Will run", describe_plan(plan)),
            ("Transcript", str(primary)),
            ("Profile", f"{profile.name} — {describe_profile(profile.name)}"),
            ("Language", "Persian" if options.language == "fa" else "English"),
            ("Speakers", str(options.speakers) if options.speakers else ("automatic" if diarize else "skipped")),
            ("Models", f"{network} cache policy"),
        ],
        dry_run=dry_run,
    )
    for note in notes:
        print(f"  ! {note}")
    for blocked in plan.blocked:
        print(f"  ! {blocked} cannot run: something it needs was excluded")

    if dry_run:
        return None
    if not plan:
        print_notice([("Reason", "everything is already up to date")], title="[bold]Nothing to do[/]")
        return None
    if is_complete(workspace.source, options, primary):
        print_notice(
            [("Output", str(primary)), ("Hint", "use --force to run again")],
            title="[bold]Already complete[/]",
        )
        return None

    log_path = _log_path(workspace)
    started = time.monotonic()
    try:
        with AccuracyStudio(output=primary, log_path=log_path) as studio:
            for note in notes:
                studio.warn(note)
            for blocked in plan.blocked:
                studio.warn(f"{blocked} was skipped: something it needs was excluded.")
            outcome = execute(
                plan, options, workspace, studio,
                ffmpeg=ffmpeg, env=env, duration=duration, network=network,
            )
            studio.finish()
            _report_outcome(studio, outcome, workspace)
    except subprocess.CalledProcessError:
        raise SystemExit(1)

    elapsed = time.monotonic() - started
    if outcome.report:
        print_report("Recording report", outcome.report)
    outcome.elapsed = elapsed
    beside = _copy_beside_input(
        options, workspace.source, primary, exported="export" in set(plan.keys())
    )
    _announce_outcome(outcome, workspace, primary, log_path, elapsed, beside=beside)
    return outcome


def _copy_beside_input(options, source: Path, primary: Path, *, exported: bool) -> Path | None:
    """Leave a ready-to-use copy of the transcript next to the recording.

    A transcript is easier to find beside the recording it came from than in a directory
    named after that recording, so one is left there too. The app's own ``input/`` is
    excepted: recordings are already collected there to be transcribed, so a copy amounts
    to clutter rather than convenience.

    Deliberately never fatal. The transcript is written and safe in the output directory
    by this point, and a read-only folder must not turn a finished run into a failure.
    """
    from whisperx_local.paths import INPUT_DIR

    if not exported or not getattr(options, "copy_beside_input", False):
        return None
    if not primary.is_file():
        return None
    destination = source.with_suffix(primary.suffix)
    if destination == source or INPUT_DIR in destination.parents:
        return None
    try:
        shutil.copyfile(primary, destination)
    except OSError as failure:
        print(f"  ! could not leave a copy beside the recording: {failure}")
        return None
    return destination


def _announce_outcome(
    outcome,
    workspace,
    primary: Path,
    log_path: Path,
    elapsed: float,
    *,
    beside: Path | None = None,
) -> None:
    """Say what was produced, and never call a run complete that wrote nothing.

    The worst result this program can produce is a transcript that looks finished and is
    not. The second worst is a run that reports success and wrote no transcript at all,
    which happened: ``--only transcribe`` plans the recognition stages and no export, so
    nothing was written -- and this panel still announced "Transcript complete" over a
    path that did not exist, with a zero exit status.

    So the file on disk decides, not the shape of the request. Four states, and only one
    of them is called complete.
    """
    summary = outcome.summary
    written = primary.is_file()

    if not written:
        _announce_unwritten(outcome, workspace, primary, log_path, elapsed)
        return

    if summary is None:
        # Something else wrote this file, or an earlier run did. Say that rather than
        # taking credit for output this run did not produce.
        print_notice(
            [
                ("Transcript", str(primary)),
                ("Note", "left as it was: this request did not include Writing transcript"),
                ("Elapsed", f"{elapsed:.1f}s"),
                ("Diagnostics", str(log_path)),
            ],
            title="[bold cyan]Existing transcript kept[/]",
        )
        return

    detail: list[tuple[str, str]] = [
        ("Transcript", str(primary)),
        ("Recovered", f"{outcome.words} words"),
        ("Covered", summary.headline()),
        ("Speakers", str(outcome.speakers) if outcome.speakers else "not labelled"),
        ("Elapsed", f"{elapsed:.1f}s"),
        ("Diagnostics", str(log_path)),
    ]
    if beside is not None:
        detail.insert(1, ("Also saved", str(beside)))
    if not summary.partial:
        print_notice(detail, title="[bold green]✓ Transcript complete[/]")
        return

    gaps = summary.gap_lines()
    shown = gaps[:4]
    loud: list[tuple[str, str]] = [("Covered", summary.headline())]
    if beside is not None:
        loud.append(("Also saved", str(beside)))
    loud += [(f"Missing {index}" if index else "", gap) for index, gap in enumerate(shown, 1)]
    if len(gaps) > len(shown):
        loud.append(("", f"and {len(gaps) - len(shown)} more"))
    if summary.unaccounted >= 1.0:
        loud.append(("Unaccounted", f"{summary.unaccounted:.0f}s with no text and no cause"))
    if summary.canonical_files:
        loud.append(("Raw copy", ", ".join(summary.canonical_files)))
    advice = summary.advice()
    if advice:
        loud.append(("Next", advice))
    loud.append(("Diagnostics", str(log_path)))
    print_notice(loud, title="[bold red]✗ Transcript INCOMPLETE[/]", border="red")
    raise SystemExit(INCOMPLETE_STATUS)


def _announce_unwritten(outcome, workspace, primary: Path, log_path: Path, elapsed: float) -> None:
    """The requested transcript does not exist. Say what does, and fail if nothing does.

    Two different situations wear the same face here. Asking for recognition only leaves
    the raw transcript on disk and simply did not render a readable one -- the request
    was honoured, so that is a warning. Asking for a transcript and getting no file at
    all is a failure, however it came about, and it does not exit zero.
    """
    rows: list[tuple[str, str]] = []
    if outcome.canonical:
        rows.append(("Wrote", ", ".join(path.name for path in outcome.canonical)))
        rows.append(("Not written", f"{primary.name}: this request had no Writing transcript"))
        rows.append(("Recognised", f"{outcome.words} words"))
        rows.append(("To write it", f'{APP_NAME} run "{workspace.source}" --only export'))
        rows.append(("Elapsed", f"{elapsed:.1f}s"))
        rows.append(("Diagnostics", str(log_path)))
        print_notice(
            rows,
            title="[bold yellow]✓ Recognised · no readable transcript[/]",
            border="yellow",
        )
        return

    rows = [
        ("Wrote", "nothing at all"),
        ("Asked for", f"{primary.name}"),
        ("Recognised", "no speech was found in the recording"),
        ("Diagnostics", str(log_path)),
    ]
    print_notice(rows, title="[bold red]✗ Nothing was written[/]", border="red")
    raise SystemExit(NOTHING_WRITTEN_STATUS)


def _log_path(workspace: WorkSpace) -> Path:
    from whisperx_local.paths import LOG_DIR

    return LOG_DIR / f"{workspace.source.stem}-{workspace.identity}.log"


def _report_outcome(studio, outcome, workspace: WorkSpace) -> None:
    """Say plainly what was lost, rather than leaving gaps to be discovered later."""
    for chunk in outcome.missing:
        studio.warn(
            f"Part {chunk.index + 1} ({clock(chunk.start)}–{clock(chunk.end)}) could not be transcribed."
        )
    if outcome.silent:
        studio.warn(
            f"No speech was found in {len(outcome.silent)} part(s); "
            "the transcript is shorter than the recording."
        )
    if outcome.unaligned:
        studio.warn(
            f"Word timing could not be refined for {len(outcome.unaligned)} part(s); the text was kept."
        )
    if outcome.incomplete:
        studio.warn("Run the same command again to retry only the failed parts.")


APP_NAME = os.environ.get("WHISPERX_LAUNCHER") or "whisperx"


# -- environment commands -------------------------------------------------


def status_command(_: argparse.Namespace) -> None:
    heading("System status")
    detail("App directory", APP_DIR)
    detail("Platform", f"{platform.system()} {host_architecture()} ({'supported' if supported_host() else 'NOT supported'})")
    detail("Python", f"{platform.python_version()} ({'supported' if SUPPORTED_PYTHON else 'NOT supported, need 3.10-3.13'})")
    detail("Performance cores", performance_cores())
    memory = available_memory_bytes()
    detail("Free memory", f"{memory / 1024**3:.2f} GB" if memory else "unknown")
    detail("ffmpeg", shutil.which("ffmpeg", path=managed_env().get("PATH")) or "not installed")
    detail("Environment", "installed" if WHISPERX.exists() else "not installed; run install")
    detail("HF token", "configured" if os.environ.get("HF_TOKEN") else "not configured")
    detail("Alignment tokenizer", "ready" if nltk_resource_available() else "missing; run resources")
    detail("Persian core cache", "ready" if core_models_cached("large-v3", "fa") else "incomplete")
    detail("Speaker cache", "ready" if DIARIZATION_READY.exists() else "pending a first labelled run")
    heading("Profiles")
    for name in PROFILES:
        detail(name, describe_profile(name))
    heading("Settings")
    for name, built_in, _ in describe():
        detail(name, built_in)
    heading("Actions")
    for action in catalog.ACTIONS:
        detail(action.key, action.summary)


def install_command(_: argparse.Namespace) -> None:
    if not supported_host():
        raise SystemExit(
            f"Unsupported platform: {platform.system()} {host_architecture()}. "
            "Supported hosts are macOS and Ubuntu/Linux on x86_64 or aarch64."
        )
    if not SUPPORTED_PYTHON:
        raise SystemExit("Python 3.10-3.13 is required. Install Python 3.12 first.")
    working_ffmpeg()
    heading("Installing pinned WhisperX")
    print("Downloads Python packages, but not speech models.", flush=True)
    if not VENV_DIR.exists():
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
    for command in (
        [str(PYTHON), "-m", "pip", "install", "--upgrade", "pip"],
        [str(PYTHON), "-m", "pip", "install", "-e", str(APP_DIR)],
    ):
        print("$ " + " ".join(command), flush=True)
        subprocess.run(command, check=True)
    heading("Installed")
    subprocess.run([str(WHISPERX), "--version"], env=managed_env(), check=False)


def resources_command(_: argparse.Namespace) -> None:
    require_install()
    NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    heading("Downloading the alignment tokenizer")
    print(f"Destination: {NLTK_DATA_DIR}", flush=True)
    subprocess.run(
        [str(PYTHON), "-m", "nltk.downloader", "-d", str(NLTK_DATA_DIR), "punkt_tab"],
        check=True,
        env=managed_env(),
    )
    if not nltk_resource_available():
        raise SystemExit("NLTK downloaded without a usable punkt_tab resource.")
    print("Alignment tokenizer: ready", flush=True)


def selftest_command(_: argparse.Namespace) -> None:
    require_install()
    heading("Running the test suite")
    # Two roots on purpose: colocated tests live beside each module, and the suites
    # that span modules live in tests/. Separate processes because the module names
    # would otherwise collide.
    roots = (("tests", "."), ("src/whisperx_local", "src/whisperx_local"))
    for start, top_level in roots:
        print(f"\n› {start}", flush=True)
        completed = subprocess.run(
            [str(PYTHON), "-m", "unittest", "discover",
             "-s", start, "-t", top_level, "-p", "test_*.py", "-v"],
            cwd=str(APP_DIR),
            env={**managed_env(), "PYTHONPATH": str(APP_DIR / "src")},
        )
        if completed.returncode:
            raise SystemExit(completed.returncode)
    heading("All tests passed")


def menu_command(args: argparse.Namespace) -> None:
    """Walk through a request in a menu, then run it.

    The menu is the default way in, so this handler is deliberately thin: the screens
    live in ``cli.guided`` and the running lives here, which is the same path a command
    line takes. Nothing about the guided route is a second implementation.
    """
    from whisperx_local.cli.guided import guided

    require_install()
    load_local_env()

    requests = guided(Preferences.load())
    if not requests:
        # Cancelling is a decision, not an error and not an empty result, so it says
        # nothing rather than reporting that there was nothing to do.
        return
    for request in requests:
        options = _options_for_composer(
            request.source,
            request.session.selection(),
            request.session.overrides,
            request.session.preset,
            request.dry_run,
        )
        _run_recording(request.source, options, dry_run=request.dry_run)


def composer_command(_: argparse.Namespace) -> None:
    """Open the free-form console, for setting a request up in any order."""
    from whisperx_local.ui import Composer

    require_install()
    load_local_env()

    def run_recording(*, files, selection, overrides, preset, dry_run):
        for path in files:
            options = _options_for_composer(path, selection, overrides, preset, dry_run)
            try:
                _run_recording(path, options, dry_run=dry_run)
            except SystemExit as error:
                print(f"  ! {path.name}: {error}", flush=True)

    Composer(run=run_recording).loop()


def compare_command(args: argparse.Namespace) -> None:
    """Measure one transcript against another, and say where they differ.

    Reads two files and touches nothing else: no workspace, no cache, no recording. The
    question "is this engine worse" should be answerable about two files that already
    exist, without running anything again.
    """
    from whisperx_local.compare import compare
    from whisperx_local.compare.transcripts import DEFAULT_TERMS, DEFAULT_WINDOW

    reference = Path(args.reference).expanduser()
    other = Path(args.other).expanduser()
    for path in (reference, other):
        if not path.is_file():
            raise SystemExit(f"No transcript at {path}")

    # Folding is on by default because without it a spelling difference reads as a
    # recognition error. Each flag turns one fold *off*, for when the spelling is the
    # thing being asked about.
    fold_options: dict[str, object] = {}
    if getattr(args, "keep_zwnj", False):
        fold_options["zwnj"] = False
    if getattr(args, "distinct_letters", False):
        fold_options["distinct_letters"] = True
    if getattr(args, "keep_punctuation", False):
        fold_options["punctuation"] = False

    result = compare(
        reference,
        other,
        window=getattr(args, "window", None) or DEFAULT_WINDOW,
        terms=tuple(getattr(args, "term", None) or DEFAULT_TERMS),
        **fold_options,
    )
    print_notice(result.rows(), title="[bold]Transcript comparison[/]")
    worst = result.worst_lines(getattr(args, "worst", 5) or 5)
    if worst:
        print_report("Worst windows", "\n".join(worst))
    if not result.compared:
        # A comparison that could not compare anything is not a result. Saying so with a
        # status means a script cannot take it for a clean bill of health.
        raise SystemExit(NOT_COMPARABLE_STATUS)


def config_command(args: argparse.Namespace) -> None:
    preferences = Preferences.load()
    try:
        if args.action == "set":
            if not args.name or args.value is None:
                raise SystemExit("usage: config set <name> <value>")
            heading(f"{args.name} = {preferences.set(args.name, args.value)}")
            print("  Applies to every later run unless a flag overrides it.")
        elif args.action == "unset":
            if not args.name:
                raise SystemExit("usage: config unset <name>")
            removed = preferences.unset(args.name)
            heading(f"Cleared {args.name}" if removed else f"{args.name} was not set")
        elif args.action == "reset":
            preferences.clear(keep_presets=True)
            heading("Preferences cleared (presets kept)")
        else:
            heading("Saved defaults")
            detail("file", preferences.path)
            for name, built_in, _ in describe():
                stored = preferences.get(name)
                detail(name, stored if stored is not None else f"{built_in}  (built-in)")
            names = preferences.preset_names()
            detail("presets", ", ".join(names) if names else "none saved")
    except (KeyError, ValueError) as error:
        raise SystemExit(str(error))


def preset_command(args: argparse.Namespace) -> None:
    preferences = Preferences.load()
    if args.action == "list":
        heading("Presets")
        names = preferences.preset_names()
        if not names:
            print("  none saved; save one with: preset save <name>", flush=True)
        for name in names:
            body = preferences.preset(name)
            detail(name, ", ".join(f"{key}={body[key]}" for key in sorted(body)))
        return
    if not args.name:
        raise SystemExit(f"usage: preset {args.action} <name>")
    if args.action == "save":
        try:
            saved = preferences.save_preset(args.name)
        except ValueError as error:
            raise SystemExit(str(error))
        heading(f"Saved preset {args.name}")
        print("  " + ", ".join(saved), flush=True)
    elif args.action == "delete":
        if not preferences.delete_preset(args.name):
            raise SystemExit(f"No preset named {args.name}")
        heading(f"Deleted preset {args.name}")
    else:
        body = preferences.preset(args.name)
        if not body:
            raise SystemExit(f"No preset named {args.name}")
        heading(f"Preset {args.name}")
        for key in sorted(body):
            detail(key, body[key])


HANDLERS = {
    "run": run_command,
    "transcribe": transcribe_command,
    "menu": menu_command,
    "composer": composer_command,
    "compare": compare_command,
    "status": status_command,
    "install": install_command,
    "resources": resources_command,
    "selftest": selftest_command,
    "config": config_command,
    "preset": preset_command,
}
