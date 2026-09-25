"""The command line.

- ``library``   finding recordings from what a user typed
- ``options``   the argument parser, and how preferences become its defaults
- ``commands``  one handler per command, including the run
- ``app``       entry point and dispatch
"""

from whisperx_local.cli.app import main, scan_option
from whisperx_local.cli.commands import HANDLERS
from whisperx_local.cli.library import (
    candidates,
    discover,
    expand_paths,
    is_media,
    presentable,
)
from whisperx_local.cli.options import build_parser, parser_with_preferences

__all__ = [
    "HANDLERS",
    "build_parser",
    "candidates",
    "discover",
    "expand_paths",
    "is_media",
    "main",
    "parser_with_preferences",
    "presentable",
    "scan_option",
]
