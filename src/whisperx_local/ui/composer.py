"""The flexible composer.

Deliberately not a wizard. There is no first step and no required order: files, actions
and settings can be changed at any point, repeatedly, and the current state is shown
after every change. A user who only wants subtitles re-rendered never has to walk past
a transcription question to get there.

Every command maps to one function, and the same functions back the command line, so
the two interfaces cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from whisperx_local.actions import catalog, plan as planning
from whisperx_local.actions.plan import Selection
from whisperx_local.config import Preferences
from whisperx_local.services import WorkSpace
from whisperx_local.ui.prompts import PromptCancelled, ask_many

PROMPT = "[bold cyan]whisperx[/] [dim]›[/] "


@dataclass
class Session:
    """Everything the composer can change, with no step order implied."""

    preferences: Preferences
    files: list[Path] = field(default_factory=list)
    only: tuple[str, ...] = ()
    without: tuple[str, ...] = ()
    overrides: dict[str, object] = field(default_factory=dict)
    preset: str | None = None

    def selection(self) -> Selection:
        return Selection(only=self.only, without=self.without)

    def action_keys(self) -> tuple[str, ...]:
        if self.only:
            return planning.canonical(self.selection())
        enabled = list(planning.canonical(self.selection()))
        return tuple(key for key in catalog.keys() if key in set(enabled))

    def toggle(self, key: str, enabled: bool) -> None:
        """Turn one action on or off, keeping the rest of the request intact."""
        catalog.get(key)  # raises with the valid names if this is a typo
        current = set(self.action_keys())
        if enabled:
            current.add(key)
            self.without = tuple(name for name in self.without if name != key)
        else:
            current.discard(key)
            if key not in self.without:
                self.without = (*self.without, key)
        self.only = tuple(name for name in catalog.keys() if name in current)

    def summary_rows(self) -> list[tuple[str, str]]:
        files = ", ".join(path.name for path in self.files) or "none chosen"
        enabled = self.action_keys()
        actions = "  ".join(
            f"[green]{key}[/]" if key in enabled else f"[dim]{key}[/]"
            for key in catalog.keys()
        )
        applied = self.preferences.resolve(preset=self.preset, cli=self.overrides)
        notable = ", ".join(
            f"{name}={applied[name]}"
            for name in ("language", "output_format", "chunk_seconds", "speakers")
            if name in applied
        )
        return [
            ("Files", files),
            ("Actions", actions),
            ("Settings", notable or "from preferences"),
            ("Preset", self.preset or "none"),
        ]

    def render(self) -> Panel:
        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim", min_width=10)
        table.add_column()
        for label, value in self.summary_rows():
            table.add_row(label, value)
        return Panel(table, title="[bold]Composer[/]", border_style="cyan", padding=(0, 1))


class Composer:
    """A command console over one :class:`Session`.

    Running is injected as a callback rather than imported, which keeps the interface
    free of the pipeline and the pipeline free of prompts.
    """

    def __init__(self, *, run, console: Console | None = None, session: Session | None = None):
        self.console = console or Console()
        self.run = run
        self.session = session or Session(preferences=Preferences.load())

    # -- the loop ---------------------------------------------------------
    def loop(self) -> int:
        self.greet()
        while True:
            try:
                raw = self.console.input(PROMPT).strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                return 0
            if not raw:
                continue
            if raw in {"quit", "exit", "q"}:
                return 0
            try:
                done = self.dispatch(raw)
            except (KeyError, ValueError) as error:
                self.console.print(f"[yellow]![/] {error}")
                continue
            if done is False:
                return 0

    def greet(self) -> None:
        self.console.print(self.session.render())
        self.console.print(
            "[dim]  type [bold]help[/] for commands, [bold]actions[/] to see and change the "
            "work, [bold]run[/] to start[/]"
        )

    def dispatch(self, raw: str) -> bool | None:
        """Route one line. Returns False when the session should end."""
        parts = raw.split()
        command, arguments = parts[0].lower(), parts[1:]
        handler = {
            "files": self.command_files,
            "file": self.command_files,
            "add": self.command_files,
            "actions": self.command_actions,
            "action": self.command_actions,
            "set": self.command_set,
            "edit": self.command_set,
            "show": self.command_show,
            "settings": self.command_show,
            "preset": self.command_preset,
            "plan": self.command_plan,
            "run": self.command_run,
            "help": self.command_help,
            "?": self.command_help,
        }.get(command)
        if handler is None:
            raise ValueError(f"unknown command {command!r}. Try: help")
        return handler(arguments)

    # -- commands ---------------------------------------------------------
    def command_help(self, arguments: list[str]) -> None:
        if arguments:
            return self.explain(arguments[0])
        self.console.print(
            Panel(
                "\n".join(
                    [
                        "[bold]files add[/] <path>…      add recordings (a folder is searched)",
                        "[bold]files clear[/]            forget the current selection",
                        "[bold]actions[/]                list the work, and what is enabled",
                        "[bold]actions +diarize -align[/] enable or disable work",
                        "[bold]actions only export[/]     run exactly one thing",
                        "[bold]set[/] <name> <value>      change a setting",
                        "[bold]set[/] <name>              be prompted, with suggestions",
                        "[bold]show[/]                   every setting and where it came from",
                        "[bold]preset save|use|list[/]    reusable bundles of settings",
                        "[bold]plan[/]                   what would actually run, and why",
                        "[bold]run[/] [--dry-run]        start; a plan that already exists is reused",
                        "[bold]quit[/]                   leave",
                    ]
                ),
                title="[bold]Commands[/]",
                border_style="cyan",
            )
        )

    def explain(self, topic: str) -> None:
        try:
            action = catalog.get(topic)
        except KeyError:
            names = ", ".join(catalog.keys())
            self.console.print(f"[dim]known actions: {names}[/]")
            return
        self.console.print(f"[bold]{action.key}[/] — {action.summary}")
        self.console.print(f"  needs:    {', '.join(action.requires) or 'nothing'}")
        self.console.print(f"  produces: {', '.join(action.provides)}")
        for option in action.options:
            self.console.print(f"  [bold]{option.name}[/]  {option.help}")

    def command_files(self, arguments: list[str]) -> None:
        from whisperx_local.cli.library import expand_paths

        if not arguments:
            self.console.print(self.session.render())
            return
        if arguments[0] == "clear":
            self.session.files.clear()
            return self.greet()
        found = expand_paths(arguments)
        if not found:
            raise ValueError("nothing usable at " + " ".join(arguments))
        for path in found:
            if path not in self.session.files:
                self.session.files.append(path)
        self.greet()

    def command_actions(self, arguments: list[str]) -> None:
        if not arguments:
            table = Table.grid(padding=(0, 2))
            enabled = set(self.session.action_keys())
            for action in catalog.ACTIONS:
                mark = "[green]✓[/]" if action.key in enabled else "[dim]○[/]"
                table.add_row(mark, f"[bold]{action.key}[/]", action.summary)
            self.console.print(table)
            return
        if arguments[0] == "only":
            wanted = planning.parse_only(" ".join(arguments[1:]))
            self.session.only = wanted
            self.session.without = ()
            return self.greet()
        if arguments[0] == "all":
            self.session.only = ()
            self.session.without = ()
            return self.greet()
        for token in arguments:
            if token.startswith("+") or token.startswith("-"):
                self.session.toggle(token[1:], token[0] == "+")
            else:
                raise ValueError(f"write +{token} to enable or -{token} to disable")
        self.greet()

    def command_set(self, arguments: list[str]) -> None:
        if not arguments:
            return self.command_show([])
        name = arguments[0]
        available = {option.name: option for option in catalog.options_for(self.session.action_keys())}
        option = available.get(name)
        if option is None:
            raise ValueError(
                f"{name} is not part of the current work. Try: actions +transcribe"
            )
        if len(arguments) == 1:
            from whisperx_local.ui.prompts import ask

            try:
                value = ask(option, self.session.overrides.get(name))
            except PromptCancelled:
                return
            if value is not None:
                self.session.overrides[name] = value
            return self.greet()
        from whisperx_local.ui.prompts import convert

        self.session.overrides[name] = convert(option, " ".join(arguments[1:]))
        self.greet()

    def command_show(self, arguments: list[str]) -> None:
        from whisperx_local.cli.options import effective_values

        resolved = effective_values(
            self.session.preferences,
            preset=self.session.preset,
            overrides=self.session.overrides,
        )
        origins = self.session.preferences.origins(preset=self.session.preset)
        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim", min_width=18)
        table.add_column()
        for option in catalog.options_for(self.session.action_keys()):
            if option.name in self.session.overrides:
                source = "this session"
            else:
                # Not every adjustable option is a savable setting, so some have no
                # recorded origin and their built-in value is all there is to report.
                source = origins.get(option.name, "built-in")
            value = resolved.get(option.name)
            # A value nobody chose is decided at run time -- by the profile, or by the
            # default folder. Printing ``None`` for it reads like a fault.
            shown = "automatic" if value is None else value
            table.add_row(f"{option.name}", f"{shown}  [dim]({source})[/]")
        self.console.print(table)
        del arguments

    def command_preset(self, arguments: list[str]) -> None:
        preferences = self.session.preferences
        if not arguments or arguments[0] == "list":
            names = preferences.preset_names()
            self.console.print(", ".join(names) if names else "[dim]no presets saved[/]")
            return
        action = arguments[0]
        name = arguments[1] if len(arguments) > 1 else None
        if action == "save":
            if not name:
                raise ValueError("usage: preset save <name>")
            preferences.save_preset(name, {**preferences.values, **self.session.overrides})
            self.session.preset = name
            self.console.print(f"saved preset [bold]{name}[/]")
        elif action == "use":
            if name not in preferences.preset_names():
                raise ValueError(f"no preset named {name}")
            self.session.preset = name
            self.greet()
        elif action == "delete":
            if not name or not preferences.delete_preset(name):
                raise ValueError(f"no preset named {name}")
            self.console.print(f"deleted preset [bold]{name}[/]")
        else:
            raise ValueError("usage: preset save|use|delete|list")

    def command_plan(self, arguments: list[str]) -> None:
        if not self.session.files:
            raise ValueError("choose a recording first: files add <path>")
        for path in self.session.files:
            workspace = WorkSpace(source=path)
            present = workspace.available(
                [], speakers=None, output_format="txt"
            )
            result = planning.resolve(self.session.selection(), available=present)
            self.console.print(
                f"[bold]{path.name}[/] → {', '.join(result.keys()) or 'nothing'}"
                + (f"   [dim]blocked: {', '.join(result.blocked)}[/]" if result.blocked else "")
            )
        del arguments

    def command_run(self, arguments: list[str]) -> bool | None:
        if not self.session.files:
            raise ValueError("choose a recording first: files add <path>")
        dry_run = "--dry-run" in arguments or "dry" in arguments
        self.run(
            files=list(self.session.files),
            selection=self.session.selection(),
            overrides=dict(self.session.overrides),
            preset=self.session.preset,
            dry_run=dry_run,
        )
        return None

    def option_group(self, name: str):
        """Every option so named, for a prompt that spans actions."""
        return [option for option in catalog.options_for(self.session.action_keys()) if option.name == name]

    def prompt_through(self, names=None) -> None:
        """Ask for settings in a group, used by `set` with no argument."""
        options = [
            option
            for option in catalog.options_for(self.session.action_keys())
            if not option.advanced and (names is None or option.name in names)
        ]
        ask_many(options, self.session.overrides, console=self.console)
        self.greet()
