"""Entry point.

Keeps three things and nothing else: read the local token file, work out which preset
was asked for before parsing, and dispatch. Everything else lives behind the handler
map in ``commands``.
"""

from __future__ import annotations

import sys

from whisperx_local.cli.options import build_parser, parser_with_preferences
from whisperx_local.paths import load_local_env


def scan_option(tokens: list[str], name: str) -> str | None:
    """Find a single-value option before parsing.

    Needed for ``--preset``, which changes the parser's own defaults and therefore has
    to be known before the parser is built.
    """
    for index, token in enumerate(tokens):
        if token == name and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith(f"{name}="):
            return token.split("=", 1)[1]
    return None


def main(arguments: list[str] | None = None) -> int:
    from whisperx_local.cli.commands import HANDLERS

    load_local_env()
    tokens = list(sys.argv[1:] if arguments is None else arguments)
    # The one-shot command pins every setting in its own handler, so the saved settings
    # are not even loaded for it. That is the whole point: the same path produces the same
    # result for a person and for a script, and cannot inherit what someone last chose.
    if tokens and tokens[0] == "transcribe":
        parser = build_parser()
    else:
        parser = parser_with_preferences(scan_option(tokens, "--preset"))
    args = parser.parse_args(tokens)
    # Kept so run_command can re-resolve its own options against preferences without
    # guessing which flags the user actually passed.
    args.argv = tokens
    try:
        HANDLERS[args.handler](args)
    except KeyboardInterrupt:
        print("\nInterrupted. Partial work is saved; run again to resume.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
