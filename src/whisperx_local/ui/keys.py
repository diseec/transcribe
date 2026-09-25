"""Reading single keypresses, so a menu can be navigated instead of typed at.

Deliberately small and optional. If no terminal is attached -- output piped to a file,
run from another program -- nothing here is used and the caller falls back to reading a
plain line. That way the menu works the same way in a script without pretending to be
interactive.
"""

from __future__ import annotations

import sys

UP = "up"
DOWN = "down"
LEFT = "left"
RIGHT = "right"
ENTER = "enter"
SPACE = "space"
ESCAPE = "escape"
BACKSPACE = "backspace"
INTERRUPT = "interrupt"
EOF = "eof"

_ARROWS = {"A": UP, "B": DOWN, "C": LEFT, "D": RIGHT}
_NAMED = {
    "\r": ENTER,
    "\n": ENTER,
    " ": SPACE,
    "\x7f": BACKSPACE,
    "\x08": BACKSPACE,
    "\x03": INTERRUPT,
    "\x04": EOF,
}


class Keys:
    """Single-key reads while the terminal is in cbreak mode.

    cbreak rather than raw: the normal line disciplines stay off, but a partial escape
    sequence is still delivered character by character, which is what arrow keys need.
    The previous settings are always restored, including on Ctrl-C, because a terminal
    left in cbreak mode stops echoing anything the user types afterwards.
    """

    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stdin
        self._saved: list | None = None

    @property
    def available(self) -> bool:
        try:
            return bool(self.stream.isatty())
        except (AttributeError, ValueError):
            return False

    def _enter(self) -> None:
        if not self.available or self._saved is not None:
            return
        import termios
        import tty

        self._saved = termios.tcgetattr(self.stream.fileno())
        tty.setcbreak(self.stream.fileno())

    def _leave(self) -> None:
        if self._saved is None:
            return
        import termios

        termios.tcsetattr(self.stream.fileno(), termios.TCSADRAIN, self._saved)
        self._saved = None

    def pause(self) -> None:
        """Hand the terminal back, for the duration of a line-edited prompt."""
        self._leave()

    def resume(self) -> None:
        self._enter()

    def __enter__(self) -> "Keys":
        self._enter()
        return self

    def __exit__(self, *exception) -> bool:
        self._leave()
        return False

    def read(self) -> str:
        """One keystroke, as a name from this module or as the character itself."""
        try:
            char = self.stream.read(1)
        except (KeyboardInterrupt, EOFError):
            return INTERRUPT
        if not char:
            return EOF
        if char == "\x1b":
            # An arrow key arrives as an escape sequence. A bare Escape is also
            # possible, and the difference is whether anything follows it.
            second = self.stream.read(1)
            if second != "[":
                return ESCAPE
            third = self.stream.read(1)
            return _ARROWS.get(third, ESCAPE)
        return _NAMED.get(char, char)
