"""A menu you navigate, so using this tool does not mean remembering its options.

Every screen offers what is available and marks the current answer, so the shortest path
through the program is pressing Enter a few times. Free text is still possible where a
value genuinely cannot be listed -- a path, a list of terms -- but nothing requires
typing, and no command or flag name has to be known or spelled correctly.

Two input styles, chosen automatically rather than configured:

* On a terminal, arrow keys move and space toggles, with the list redrawn in place.
* Otherwise the list is printed once and answered with a number, which keeps a piped or
  scripted run working without pretending to be interactive.

Drawing is plain text with cursor movement rather than a live rich display. The
dashboard already owns that machinery during a run, and two things fighting over the
screen is worse than one of them being plain.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from whisperx_local.ui.keys import DOWN, ENTER, ESCAPE, INTERRUPT, LEFT, RIGHT, SPACE, UP, Keys

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
RESET = "\033[0m"

MOVE_HINT = "↑↓ move  ·  Enter choose  ·  q quit"
TOGGLE_HINT = "↑↓ move  ·  space toggle  ·  Enter continue  ·  q quit"


def _is_tty(stream) -> bool:
    """Whether this looks like a terminal, treating anything unusual as "not one"."""
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False


@dataclass(frozen=True)
class Choice:
    """One row: what it says, what it is, and any explanation worth showing."""

    label: str
    detail: str = ""
    value: object = None


class Menu:
    """Draws a list, reads a key, redraws. Falls back to numbers without a terminal."""

    def __init__(self, keys: Keys | None = None, out=None):
        self.keys = keys if keys is not None else Keys()
        self.out = out if out is not None else sys.stdout
        self._height = 0

    @property
    def interactive(self) -> bool:
        """Whether a navigable menu is possible, which needs both ends to be a terminal.

        Output matters as much as input: the menu redraws itself in place, and a redraw
        written into a pipe is a pile of escape sequences. Redirecting the output is
        therefore enough on its own to switch to numbered answers -- which also means
        piping a menu to a file cannot hang waiting for a keypress that will never be
        visible.
        """
        return self.keys.available and _is_tty(self.out)

    # -- drawing ----------------------------------------------------------

    def banner(self, title: str, subtitle: str = "") -> None:
        """A screen heading, printed above the part that gets redrawn."""
        self._finish()
        self._write(f"\n{BOLD}{title}{RESET}\n")
        if subtitle:
            self._write(f"{DIM}{subtitle}{RESET}\n")
        self._write("\n")

    def _write(self, text: str) -> None:
        self.out.write(text)
        self.out.flush()

    def _paint(self, rows: list[str]) -> None:
        """Redraw the block in place, padding so the cursor maths stays exact."""
        width = max(40, min(160, self._terminal_width() - 1))
        rows = [row[:width] for row in rows]
        height = max(self._height, len(rows))
        rows = rows + [""] * (height - len(rows))
        lead = f"\033[{self._height}A" if self._height else ""
        self._write(lead + "".join(f"\033[2K{row}\n" for row in rows))
        self._height = len(rows)

    def _terminal_size(self) -> tuple[int, int]:
        try:
            import shutil

            size = shutil.get_terminal_size((100, 24))
            return size.columns, size.lines
        except (OSError, ValueError):
            return 100, 24

    def _terminal_width(self) -> int:
        return self._terminal_size()[0]

    def _viewport_limit(self) -> int:
        """How many rows fit, leaving room for the heading, hints and a spare line."""
        return max(4, self._terminal_size()[1] - 6)

    def _window(self, count: int, cursor: int) -> tuple[int, int]:
        """The slice of rows worth drawing around the cursor.

        A list taller than the terminal has to be windowed. Drawing past the bottom
        makes the terminal scroll, and moving the cursor up by the number of rows
        written then lands somewhere else entirely, leaving a trail of half-drawn
        frames down the screen.
        """
        limit = self._viewport_limit()
        if count <= limit:
            return 0, count
        half = max(1, limit // 2)
        start = max(0, min(cursor - half, count - limit))
        return start, start + limit

    def _finish(self) -> None:
        """Forget the drawn block, leaving what is on screen where it is."""
        self._height = 0

    # -- rows -------------------------------------------------------------

    @staticmethod
    def rows(
        choices: list[Choice],
        cursor: int,
        *,
        chosen: tuple[int, ...] = (),
        multi: bool = False,
        offset: int = 0,
    ) -> list[str]:
        """The list as text, numbered from ``offset`` so numbering stays global.

        Pure and static, so the layout can be checked without a terminal.
        """
        rows: list[str] = []
        for index, choice in enumerate(choices):
            marker = f"{CYAN}❯{RESET}" if index == cursor else " "
            box = f"[{'x' if index in chosen else ' '}] " if multi else ""
            number = f"{index + offset + 1:>2}."
            label = choice.label if index != cursor else f"{BOLD}{choice.label}{RESET}"
            row = f"  {marker} {box}{number} {label}"
            if choice.detail:
                row += f"  {DIM}{choice.detail}{RESET}"
            rows.append(row)
        return rows

    def frame(
        self,
        choices: list[Choice],
        cursor: int,
        *,
        chosen: tuple[int, ...] = (),
        multi: bool = False,
    ) -> list[str]:
        """The visible rows, with a count of what is scrolled off either end."""
        start, end = self._window(len(choices), cursor)
        visible = choices[start:end]
        local = tuple(index - start for index in chosen if start <= index < end)
        rows: list[str] = []
        if start:
            rows.append(f"  {DIM}↑ {start} above{RESET}")
        rows.extend(
            self.rows(visible, cursor - start, chosen=local, multi=multi, offset=start)
        )
        if end < len(choices):
            rows.append(f"  {DIM}↓ {len(choices) - end} below{RESET}")
        return rows

    # -- interaction ------------------------------------------------------

    def choose(
        self,
        title: str,
        choices: list[Choice],
        *,
        index: int = 0,
        multi: bool = False,
        chosen: tuple[int, ...] = (),
        hint: str | None = None,
    ) -> int | tuple[int, ...] | None:
        """Pick one row, or a set of rows in multi mode. ``None`` means cancelled."""
        if not choices:
            return () if multi else None
        self.banner(title)
        if not self.interactive:
            return self._choose_by_number(choices, multi=multi, chosen=chosen)

        cursor = max(0, min(index, len(choices) - 1))
        picked = set(chosen)
        footer = [f"{DIM}{hint or (TOGGLE_HINT if multi else MOVE_HINT)}{RESET}"]
        with self.keys as keys:
            while True:
                self._paint(
                    self.frame(
                        choices, cursor, chosen=tuple(sorted(picked)), multi=multi
                    )
                    + footer
                )
                key = keys.read()
                if key in (ESCAPE, "q", INTERRUPT):
                    self._finish()
                    return None
                if key == UP:
                    cursor = (cursor - 1) % len(choices)
                elif key == DOWN:
                    cursor = (cursor + 1) % len(choices)
                elif key == LEFT and cursor > 0:
                    cursor -= 1
                elif key == RIGHT and cursor < len(choices) - 1:
                    cursor += 1
                elif key == SPACE and multi:
                    picked.symmetric_difference_update({cursor})
                elif key == ENTER:
                    self._finish()
                    return tuple(sorted(picked)) if multi else cursor
                elif key.isdigit() and key != "0":
                    position = int(key) - 1
                    if position < len(choices):
                        cursor = position
                        if multi:
                            picked.symmetric_difference_update({cursor})
                        else:
                            self._finish()
                            return cursor

    def _choose_by_number(
        self, choices: list[Choice], *, multi: bool, chosen: tuple[int, ...]
    ) -> int | tuple[int, ...] | None:
        """The non-terminal path: list once, then read a plain line."""
        for row in self.rows(choices, -1, chosen=chosen, multi=multi):
            self._write(row + "\n")
        prompt = "Numbers, space separated: " if multi else "Number: "
        while True:
            try:
                answer = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                return None
            if not answer:
                return tuple(sorted(chosen)) if multi else 0
            numbers = [int(part) - 1 for part in answer.split() if part.isdigit()]
            valid = [number for number in numbers if 0 <= number < len(choices)]
            if valid:
                return tuple(sorted(valid)) if multi else valid[-1]
            self._write("Please enter a number from the list.\n")

    def ask_text(self, label: str, *, default: str = "") -> str | None:
        """Read a line, for the few values that cannot be offered as a list."""
        self._finish()
        suffix = f" [{default}]" if default else ""
        self._write(f"\n{BOLD}{label}{RESET}{suffix}\n")
        self.keys.pause()
        try:
            answer = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        finally:
            self.keys.resume()
        return answer or default or None
