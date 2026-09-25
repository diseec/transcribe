"""Everything the user sees.

- ``console``    panels that print once and scroll away
- ``dashboard``  the live view that owns the screen during a run
- ``menu``       the navigable menu, which is how the program is normally driven
- ``prompts``    asking for one setting, with suggestions and free text
- ``composer``   the flexible console where work is chosen in any order

Rich is confined to this package. Everything else deals in percentages, paths and
text, which is why the logic underneath is testable without a terminal.
"""

from whisperx_local.ui.composer import Composer, Session
from whisperx_local.ui.console import (
    clock,
    print_notice,
    print_plan,
    print_report,
)
from whisperx_local.ui.dashboard import AccuracyStudio
from whisperx_local.ui.menu import Choice, Menu
from whisperx_local.ui.prompts import PromptCancelled, ask, ask_many, convert, hint_for

__all__ = [
    "AccuracyStudio",
    "Choice",
    "Composer",
    "Menu",
    "PromptCancelled",
    "Session",
    "ask",
    "ask_many",
    "clock",
    "convert",
    "hint_for",
    "print_notice",
    "print_plan",
    "print_report",
]
