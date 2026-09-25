"""What this tool can do, declared as data.

The pipe from an input file to a finished transcript is a set of actions, each naming
the facts it needs and the facts it leaves behind. From that one declaration come:

- the ability to run any subset, in dependency order, skipping what already exists
- the options each action exposes, for flags and for prompts alike
- the list a user browses in the composer

``base`` holds the types, ``catalog`` the actions themselves, ``plan`` the resolution
of a request into the smallest set of work that satisfies it.
"""

from whisperx_local.actions.base import (
    AUDIO,
    FILES,
    PLAN,
    REPORT,
    SEGMENTS,
    SOURCE,
    TURNS,
    WORDS,
    Action,
    Option,
    RunRequest,
)
from whisperx_local.actions.catalog import (
    ACTIONS,
    BY_KEY,
    DEFAULT_ACTIONS,
    get,
    keys,
    options_for,
    ordered,
)
from whisperx_local.actions.plan import (
    Plan,
    Selection,
    canonical,
    parse_only,
    resolve,
)

__all__ = [
    "ACTIONS",
    "AUDIO",
    "BY_KEY",
    "DEFAULT_ACTIONS",
    "FILES",
    "PLAN",
    "REPORT",
    "SEGMENTS",
    "SOURCE",
    "TURNS",
    "WORDS",
    "Action",
    "Option",
    "Plan",
    "RunRequest",
    "Selection",
    "canonical",
    "get",
    "keys",
    "options_for",
    "ordered",
    "parse_only",
    "resolve",
]
