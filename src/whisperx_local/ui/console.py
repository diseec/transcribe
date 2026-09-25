"""Plain console output: the plan panel, notices, and duration formatting.

These are the pieces that print once and scroll away, as opposed to the dashboard,
which owns the screen while a run is in progress. Kept apart because that difference
is what decides which one a caller wants.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

TITLE = "[bold black on cyan] WHISPERX [/][bold]  Accuracy Studio[/]"


def clock(seconds: float) -> str:
    """Format a duration as m:ss, or h:mm:ss for longer spans."""
    minutes, remainder = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{remainder:02d}"
    return f"{minutes}:{remainder:02d}"


def rows_table(rows: list[tuple[str, str]]) -> Table:
    """A label/value grid, which is the shape almost every panel here wants.

    Expanding to the panel width matters: without it a long value such as the list of
    actions about to run is silently truncated at the border, hiding exactly the
    information the panel exists to show.
    """
    table = Table.grid(padding=(0, 2), expand=True)
    table.add_column(style="dim", min_width=16)
    table.add_column(style="bold", overflow="fold")
    for label, value in rows:
        table.add_row(label, str(value))
    return table


def print_plan(rows: list[tuple[str, str]], *, dry_run: bool = False) -> None:
    """Show what is about to happen, before it happens."""
    subtitle = "Preview · nothing will run" if dry_run else "Everything runs on this machine"
    Console().print()
    Console().print(
        Panel(
            rows_table(rows),
            title=TITLE,
            subtitle=f"[dim]{subtitle}[/]",
            border_style="cyan",
            padding=(1, 2),
        )
    )


def print_notice(rows: list[tuple[str, str]], *, title: str, border: str = "cyan") -> None:
    """A short informational panel, such as when a finished job is skipped.

    ``border`` carries meaning rather than decoration: red is reserved for a result
    that is missing something, so a panel that did not fully succeed is recognisable
    at a glance even when the title is skimmed.
    """
    Console().print()
    Console().print(
        Panel(rows_table(rows), title=title, border_style=border, padding=(1, 2))
    )


def print_report(title: str, body: str) -> None:
    """Show a block of pre-formatted text, used by the report action."""
    Console().print()
    Console().print(
        Panel(body.rstrip(), title=title, border_style="green", padding=(1, 2))
    )
