"""The guided path: which recording, how much work, and how careful.

This is the way in by default. The alternative -- knowing which of twenty-five settings
to combine into a command line -- is not a reasonable thing to ask of the person who
owns the recordings, and it is the reason a fully capable program can still be unusable.

Four ideas shape the screens:

* **Offer the answers rather than ask for them.** Every screen lists what is available
  with the current value marked, so pressing Enter is a valid answer throughout.
* **Ask few questions.** One for the recording, one for how much work, one for how
  careful, then a confirmation. Everything else is reachable and none of it is required.
* **Show the consequence.** The last screen states what will run and what will be
  written, which is also the point at which changing your mind is still free.
* **Keep the expert path.** Per-setting control, with suggestions and free text, stays
  one keystroke away for the times a specific value is the whole point.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from whisperx_local.actions import catalog
from whisperx_local.config import Preferences
from whisperx_local.engine import PROFILES, describe as describe_profile
from whisperx_local.ui.composer import Session
from whisperx_local.ui.menu import Choice, Menu

# The stages every request includes. Recognition and a file on disk are the product;
# the others are refinements that can fail without costing a single recognised word.
CORE_STAGES = ("prepare", "boundaries", "transcribe", "export")


@dataclass
class Request:
    """One recording, the choices made for it, and whether to only describe them."""

    session: Session
    dry_run: bool = False

    @property
    def source(self) -> Path:
        return self.session.files[0]


@dataclass(frozen=True)
class Work:
    """One answer to "how much work", as named actions rather than a sentence."""

    label: str
    detail: str
    align: bool = False
    diarize: bool = False
    analyze: bool = False

    @property
    def stages(self) -> tuple[str, ...]:
        chosen = {"align": self.align, "diarize": self.diarize, "analyze": self.analyze}
        keep = set(CORE_STAGES) | {key for key, on in chosen.items() if on}
        return tuple(key for key in catalog.keys() if key in keep)


WORK: tuple[Work, ...] = (
    Work(
        "Transcription only",
        "recognised text with timestamps, and nothing else",
    ),
    Work(
        "Add word timings",
        "word-level times, useful for subtitles; slower, no extra words",
        align=True,
    ),
    Work(
        "Add speakers",
        "who spoke when, labelled per word; the slowest stage by some way",
        diarize=True,
    ),
    Work(
        "Everything",
        "word timings, speakers, and a quality report",
        align=True,
        diarize=True,
        analyze=True,
    ),
)


def apply_work(session: Session, work: Work) -> None:
    """Record one answer as the set of stages to run.

    Speaker separation is written as an explicit ``False`` rather than left unset.
    Left unset it falls back to "on if a token is configured", which is how a request
    that excluded it still ended up running it.
    """
    session.only = work.stages
    session.without = ()
    session.overrides["diarize"] = bool(work.diarize)
    # The quality report is not a stored setting -- it is an action, and it is carried
    # by the selected stages -- so it is deliberately not written here.


def profile_choices() -> list[Choice]:
    """The profiles, most exact first, because that is the direction the choice is about."""
    return [
        Choice(name, describe_profile(name), value=name)
        for name in ("maximum", "accurate", "balanced", "fast")
        if name in PROFILES
    ]


def _size(path: Path) -> str:
    try:
        return f"{path.stat().st_size / 1024**2:.0f} MB"
    except OSError:
        return ""


def _already_done(path: Path) -> str:
    """Whether this recording looks transcribed already, so a rerun is a plain choice."""
    from whisperx_local.paths import OUTPUT_DIR

    for suffix in (".raw.txt", ".txt", ".srt", ".json"):
        if (OUTPUT_DIR / f"{path.stem}{suffix}").is_file():
            return "already transcribed"
    return "new"


def recording_choices(paths: list[Path]) -> list[Choice]:
    return [
        Choice(path.name, f"{_size(path)}  ·  {_already_done(path)}", value=path)
        for path in paths
    ]


def _ask_recording(menu: Menu, preferences: Preferences) -> list[Path] | None:
    """Choose one or more recordings, or name a path that was not found automatically."""
    from whisperx_local.cli.library import candidates

    while True:
        found = candidates()
        choices = recording_choices(found)
        choices.append(Choice("A file somewhere else…", "type a path", value=None))
        answer = menu.choose(
            "Which recording?",
            choices,
            multi=True,
            hint="↑↓ move  ·  space to mark  ·  Enter continue  ·  q quit",
        )
        if answer is None:
            return None
        if not answer:
            continue
        if len(choices) - 1 in answer:
            typed = menu.ask_text("Path to the recording")
            if typed:
                path = Path(typed).expanduser()
                if path.is_file():
                    return [path.resolve()]
                menu.banner("Not found", f"{path} does not exist")
            continue
        return [choices[index].value for index in answer if choices[index].value]


def _ask_work(menu: Menu, session: Session) -> bool:
    """How much work, with the expert path as the last row."""
    choices = [Choice(work.label, work.detail, value=work) for work in WORK]
    choices.append(Choice("Choose each step myself…", "pick from every stage", value=None))
    answer = menu.choose("How much work?", choices, multi=False)
    if answer is None:
        return False
    if choices[answer].value is not None:
        apply_work(session, choices[answer].value)
        return True

    stages = menu.choose(
        "Which steps?",
        [Choice(action.label, action.summary, value=action.key) for action in catalog.ACTIONS],
        multi=True,
        chosen=tuple(
            index
            for index, action in enumerate(catalog.ACTIONS)
            if action.key in session.only or not session.only
        ),
    )
    if stages is None:
        return False
    picked = tuple(catalog.ACTIONS[index].key for index in stages)
    session.only = tuple(key for key in catalog.keys() if key in set(picked) | set(CORE_STAGES))
    session.without = ()
    session.overrides["diarize"] = "diarize" in picked
    return True


def _ask_care(menu: Menu, session: Session) -> bool:
    """How careful, which is the one real trade-off in the program."""
    choices = profile_choices()
    choices.append(Choice("Fine-tune settings…", "model, precision, batching, vocabulary", value=None))
    current = choices.index(
        next(
            (choice for choice in choices if choice.value == session.preferences.resolve()["profile"]),
            choices[1] if len(choices) > 1 else choices[0],
        )
    )
    answer = menu.choose("How careful?", choices, index=current)
    if answer is None:
        return False
    if choices[answer].value is not None:
        session.preset = None
        session.overrides["profile"] = choices[answer].value
        return True
    return _fine_tune(menu, session)


def _value_text(value: object) -> str:
    """How a current value reads, with nothing pretending to be a value.

    ``None`` means "decided for you": the profile supplies the model and batch size,
    ``output_dir`` falls back to the default folder, and an unset ``hotwords`` means none
    were given. All three read as ``automatic``, which is the wording the ``config``
    command already uses for the same situation.
    """
    if value is None:
        return "automatic"
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value)


def _fine_tune(menu: Menu, session: Session) -> bool:
    """Per-setting control, with each setting's suggestions and free text.

    The options come from the action catalog, not the settings table. Both carry the
    same names, but only the catalog declares the options that belong to the work being
    done, and some of those -- ``vad_onset``, ``chunk_size``, the decoding thresholds
    -- are adjustable without being savable. Handing one of those to the prompt as a
    saved-setting record is what raised ``AttributeError: 'Setting' object has no
    attribute 'examples'`` and made this screen unusable.
    """
    from whisperx_local.cli.options import effective_values
    from whisperx_local.ui.prompts import PromptCancelled, ask

    options = catalog.options_for(session.only or catalog.DEFAULT_ACTIONS)
    while True:
        resolved = effective_values(
            session.preferences, preset=session.preset, overrides=session.overrides
        )
        choices = [
            Choice(option.name, _value_text(resolved.get(option.name)), value=option)
            for option in options
        ]
        choices.append(Choice("Done", "back to the previous screen", value=None))
        answer = menu.choose("Settings", choices)
        if answer is None:
            return True
        chosen = choices[answer].value
        if chosen is None:
            return True
        try:
            value = ask(chosen, resolved.get(chosen.name))
        except PromptCancelled:
            continue
        if value is not None:
            session.overrides[chosen.name] = value


def _confirm_rows(session: Session) -> list[tuple[str, str]]:
    """What the last screen says, in plain text, including what will *not* run.

    Built here rather than reusing the composer's summary. That one carries rich markup
    for a rich panel; this banner is written straight out, so reusing it showed literal
    ``[green]`` tags -- which destroyed the only signal about which stages are off.
    """
    from whisperx_local.cli.options import effective_values

    resolved = effective_values(
        session.preferences, preset=session.preset, overrides=session.overrides
    )
    enabled = set(session.action_keys())
    will_run = [action.label for action in catalog.ACTIONS if action.key in enabled]
    skipped = [action.label for action in catalog.ACTIONS if action.key not in enabled]
    profile = resolved.get("profile")

    rows: list[tuple[str, str]] = [
        ("Recording", ", ".join(path.name for path in session.files) or "none chosen"),
        ("Will run", " → ".join(will_run) or "nothing"),
    ]
    if skipped:
        # Named explicitly: a stage left out on purpose and a stage forgotten look
        # identical on a list that only shows what is enabled.
        rows.append(("Not running", ", ".join(skipped)))
    rows += [
        ("Quality", f"{profile} — {describe_profile(profile)}" if profile else "default"),
        ("Language", "Persian" if resolved.get("language") == "fa" else "English"),
        ("Cut on pauses", _value_text(resolved.get("silence_split"))),
        ("Quiet voices", "levelling on" if resolved.get("normalize") else "levelling off"),
        ("Format", _value_text(resolved.get("output_format"))),
    ]
    return rows


def _confirm(menu: Menu, session: Session) -> str | None:
    """The last look before anything runs. ``None`` means cancelled."""
    rows = _confirm_rows(session)
    menu.banner("Ready to run", "  ·  ".join(f"{label}: {value}" for label, value in rows))
    answer = menu.choose(
        "Start?",
        [
            Choice("Run now", "start transcribing", value="run"),
            Choice("Preview only", "show what would run, change nothing", value="preview"),
            Choice("Change something", "back to the settings", value="again"),
            Choice("Cancel", "nothing happens", value="cancel"),
        ],
    )
    return None if answer is None else ["run", "preview", "again", "cancel"][answer]


def guided(preferences: Preferences | None = None, *, menu: Menu | None = None) -> list[Request] | None:
    """Walk through a request. Returns what to run, or ``None`` if it was cancelled."""
    preferences = preferences if preferences is not None else Preferences.load()
    menu = menu if menu is not None else Menu()
    menu.banner("WhisperX · Accuracy Studio", "Transcribe recordings on this machine")

    files = _ask_recording(menu, preferences)
    if not files:
        return None

    session = Session(preferences=preferences, files=files)
    apply_work(session, WORK[0])
    if not _ask_work(menu, session) or not _ask_care(menu, session):
        return None

    while True:
        answer = _confirm(menu, session)
        if answer in (None, "cancel"):
            return None
        if answer != "again":
            # One request per recording, so each gets its own progress display and its
            # own coverage report rather than one summary covering a batch.
            return [
                Request(session=_for_one(session, path), dry_run=answer == "preview")
                for path in files
            ]
        if not _ask_work(menu, session) or not _ask_care(menu, session):
            return None


def _for_one(session: Session, path: Path) -> Session:
    """The same choices, applied to a single recording."""
    return Session(
        preferences=session.preferences,
        files=[path],
        only=session.only,
        without=session.without,
        overrides=dict(session.overrides),
        preset=session.preset,
    )
