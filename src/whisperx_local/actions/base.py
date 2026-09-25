"""What an action is, and what an option is.

These are declarations, not behaviour. One ``Option`` record drives the command-line
flag, the interactive prompt, the saved preference and the validation, so the
interface cannot offer a choice the engine does not support and no setting is defined
twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The facts a recording accumulates as it is processed. Actions declare which of these
# they need and which they produce, and the planner uses that to work out what
# actually has to run -- which is what lets a user ask for one stage on its own.
SOURCE = "source"        # the input file, as given
AUDIO = "audio"          # levelled mono audio for recognition
PLAN = "plan"            # the chunk boundaries
SEGMENTS = "segments"    # recognised text with segment timings
WORDS = "words"          # per-word timings, from alignment
TURNS = "turns"          # speaker turns, from diarization
FILES = "files"          # written transcripts
REPORT = "report"        # measured statistics about the recording


@dataclass(frozen=True)
class Option:
    """One adjustable setting, declared once and reused everywhere."""

    name: str
    kind: type
    help: str
    choices: tuple[str, ...] | None = None
    advanced: bool = False
    examples: tuple[str, ...] = ()

    def describe(self) -> str:
        """The value hint shown beside the name, e.g. ``fa|en``."""
        if self.choices:
            return "|".join(self.choices)
        return self.kind.__name__


@dataclass(frozen=True)
class Action:
    """One unit of work, with what it needs and what it leaves behind."""

    key: str
    label: str
    summary: str
    provides: tuple[str, ...]
    requires: tuple[str, ...] = ()
    options: tuple[Option, ...] = ()
    always_available: bool = False

    def needs(self) -> frozenset[str]:
        return frozenset(self.requires)


@dataclass
class RunRequest:
    """What the user asked for, before it is turned into a plan."""

    source: str | None = None
    actions: tuple[str, ...] = ()
    preset: str | None = None
    overrides: dict[str, object] = field(default_factory=dict)

    def describe(self) -> str:
        return ", ".join(self.actions) if self.actions else "the default set"
