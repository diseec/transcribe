"""Asking for one setting, with the accepted values offered and free text allowed.

The prompt shows what the option accepts and what it is currently set to, then accepts
anything valid. That combination is the point: suggestions remove the need to remember
values, and free text means a suggestion list is never a cage.
"""

from __future__ import annotations

from rich.console import Console

from whisperx_local.actions.base import Option

TRUE_WORDS = frozenset({"1", "true", "yes", "on", "y"})
FALSE_WORDS = frozenset({"0", "false", "no", "off", "n"})


class PromptCancelled(Exception):
    """The user asked to stop rather than answer."""


def hint_for(option: Option) -> str:
    """The short value hint shown beside an option's name."""
    if option.choices:
        return "|".join(option.choices)
    return option.kind.__name__


def convert(option: Option, raw: str):
    """Turn typed text into a value, or explain why it cannot be."""
    text = raw.strip()
    if text in {"", "-"}:
        raise ValueError(f"{option.name} needs a value")
    if option.choices is not None and text not in option.choices:
        allowed = ", ".join(option.choices)
        raise ValueError(f"{option.name} must be one of: {allowed}")
    if option.kind is bool:
        lowered = text.lower()
        if lowered in TRUE_WORDS:
            return True
        if lowered in FALSE_WORDS:
            return False
        raise ValueError(f"{option.name} expects true or false, got {text!r}")
    try:
        return option.kind(text)
    except ValueError as error:
        raise ValueError(
            f"{option.name} expects {option.kind.__name__}, got {text!r}"
        ) from error


def ask(option: Option, current=None, *, console: Console | None = None):
    """Prompt for one option. An empty answer keeps the current value."""
    console = console or Console()
    console.print(
        f"  [bold]{option.name}[/] [dim]({hint_for(option)})[/]  {option.help}"
    )
    if option.examples:
        console.print(f"    [dim]examples: {', '.join(option.examples)}[/]")
    shown = "" if current is None else f" [dim]currently {current}[/]"
    try:
        raw = console.input(f"    value{shown} [dim]· enter to keep[/] › ")
    except (EOFError, KeyboardInterrupt) as error:
        raise PromptCancelled() from error
    if not raw.strip():
        return current
    return convert(option, raw)


def ask_many(options, values: dict, *, console: Console | None = None) -> dict:
    """Prompt through a group of options, skipping any answered ``c`` for cancel.

    Returns the updated values. An answer is applied as soon as it is accepted, so
    cancelling midway keeps what was already decided rather than discarding it.
    """
    console = console or Console()
    for option in options:
        try:
            values[option.name] = ask(option, values.get(option.name), console=console)
        except ValueError as error:
            console.print(f"  [yellow]![/] {error}")
    return values
