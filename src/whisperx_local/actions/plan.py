"""Turning a request into the smallest set of actions that satisfies it.

This is the piece that makes subsets real. Asking for one stage does not mean running
everything before it: if the artifacts it needs are already on disk from an earlier
run, only that stage runs. Asking for a stage whose inputs are missing pulls in just
the producers needed to fill the gap.

The logic is pure, so the awkward combinations are tested without a recording.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from whisperx_local.actions.base import SOURCE, Action
from whisperx_local.actions.catalog import ACTIONS, BY_KEY, DEFAULT_ACTIONS, ordered


@dataclass(frozen=True)
class Plan:
    """What will actually run, and what had to be added to make it possible."""

    actions: tuple[Action, ...]
    available: frozenset[str] = frozenset()
    added: tuple[str, ...] = ()
    blocked: tuple[str, ...] = ()

    def keys(self) -> tuple[str, ...]:
        return tuple(action.key for action in self.actions)

    def labels(self) -> tuple[str, ...]:
        return tuple(action.label for action in self.actions)

    def __bool__(self) -> bool:
        return bool(self.actions)


@dataclass
class Selection:
    """A request in the form the command line and composer both produce."""

    only: tuple[str, ...] = ()
    without: tuple[str, ...] = ()
    extra: tuple[str, ...] = field(default_factory=tuple)

    def is_empty(self) -> bool:
        return not (self.only or self.without or self.extra)


def canonical(selection: Selection) -> tuple[str, ...]:
    """The requested action keys: an explicit list, or the default set minus removals."""
    base = list(selection.only) if selection.only else list(DEFAULT_ACTIONS)
    for key in selection.extra:
        if key not in base:
            base.append(key)
    return tuple(key for key in base if key not in set(selection.without))


def resolve(selection: Selection, available=()) -> Plan:
    """The smallest dependency-closed plan that satisfies ``selection``.

    ``available`` names the artifacts already on disk. Only what is genuinely missing
    is added, which is why re-rendering subtitles or adding speaker labels does not
    re-transcribe anything.

    An action whose inputs cannot be produced -- because the very action that would
    produce them was excluded -- is dropped and reported in ``blocked`` rather than
    run and failing later.
    """
    present = set(available)
    present.discard(SOURCE)  # always obtainable from the recording itself
    wanted = canonical(selection)
    if not wanted:
        # Removing everything is a legitimate request, but it is not a run.
        return Plan(actions=(), available=frozenset(present))

    forbidden = set(selection.without)
    chosen = list(ordered(wanted))
    added: list[str] = []
    blocked: list[str] = []

    # One pass can reveal a new dependency (a producer inserted for a later action has
    # needs of its own), so this repeats until nothing changes.
    changed = True
    while changed:
        changed = False
        for action in list(chosen):
            produced_earlier: set[str] = set()
            for earlier in chosen:
                if earlier is action:
                    break
                produced_earlier |= set(earlier.provides)
            missing = (action.needs() - {SOURCE}) - present - produced_earlier
            if not missing:
                continue

            producers = [
                producer
                for producer in _producers_for(missing, before=action)
                if producer.key not in forbidden
            ]
            underway = {entry.key for entry in chosen}
            fresh = [producer for producer in producers if producer.key not in underway]
            if not producers and not produced_earlier.intersection(missing):
                chosen.remove(action)
                blocked.append(action.key)
                changed = True
                continue
            position = chosen.index(action)
            for producer in fresh:
                chosen.insert(position, producer)
                added.append(producer.key)
                changed = True

    final = tuple(action for action in ACTIONS if action.key in {entry.key for entry in chosen})
    produced = set(present)
    for action in final:
        produced |= set(action.provides)
    return Plan(
        actions=final,
        available=frozenset(produced),
        added=tuple(added),
        blocked=tuple(blocked),
    )


def _producers_for(artifacts: set[str], *, before: Action) -> tuple[Action, ...]:
    """Actions that produce the needed artifacts and run before ``before``."""
    limit = ACTIONS.index(before)
    return tuple(
        action for action in ACTIONS[:limit] if set(action.provides) & artifacts
    )


def parse_only(text: str) -> tuple[str, ...]:
    """Parse a comma or plus separated action list, as the command line accepts."""
    parts = [part.strip() for part in text.replace("+", ",").split(",")]
    return tuple(part for part in parts if part)


def known_action_names() -> tuple[str, ...]:
    return tuple(BY_KEY)
