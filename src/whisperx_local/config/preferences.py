"""Persistent user preferences and named presets.

Stored as JSON in the user's config directory so a default survives reinstalls and is
shared by every checkout. Layering is explicit and ordered, and later layers win:

    built-in defaults  →  preferences  →  preset  →  environment  →  CLI flags

A damaged or hand-edited file is tolerated rather than fatal: unknown names and
unconvertible values are dropped on load, because losing your saved defaults is a
worse outcome than ignoring one bad line. Values arriving through :meth:`set` are
validated strictly, since there the user is present and can be told what is wrong.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from whisperx_local.config.settings import BY_NAME, SETTINGS, coerce
from whisperx_local.paths import SETTINGS_FILE

SCHEMA = "prefs-v1"


@dataclass
class Preferences:
    """User-chosen defaults and named presets, backed by one JSON file."""

    path: Path = SETTINGS_FILE
    values: dict[str, object] = field(default_factory=dict)
    presets: dict[str, dict[str, object]] = field(default_factory=dict)

    # -- loading and saving ----------------------------------------------
    @classmethod
    def load(cls, path: Path | None = None) -> "Preferences":
        store = cls(path=Path(path) if path is not None else SETTINGS_FILE)
        store._read()
        return store

    def _read(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
            return
        raw_values = payload.get("values")
        if isinstance(raw_values, dict):
            self.values = self._clean(raw_values)
        raw_presets = payload.get("presets")
        if isinstance(raw_presets, dict):
            self.presets = {
                str(name): self._clean(body)
                for name, body in raw_presets.items()
                if isinstance(body, dict)
            }

    @staticmethod
    def _clean(raw: dict) -> dict[str, object]:
        """Keep only known settings with usable values."""
        clean: dict[str, object] = {}
        for name, value in raw.items():
            if str(name) not in BY_NAME:
                continue
            try:
                clean[str(name)] = coerce(str(name), value)
            except (KeyError, ValueError):
                continue
        return clean

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": SCHEMA, "values": self.values, "presets": self.presets}
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)

    # -- single values ----------------------------------------------------
    def set(self, name: str, value: object, *, persist: bool = True) -> object:
        """Validate, store and return one preference."""
        converted = coerce(name, value)
        self.values[name] = converted
        if persist:
            self.save()
        return converted

    def unset(self, name: str, *, persist: bool = True) -> bool:
        """Forget one preference. Returns whether it was set."""
        existed = self.values.pop(name, None) is not None
        if existed and persist:
            self.save()
        return existed

    def get(self, name: str, fallback: object = None) -> object:
        return self.values.get(name, fallback)

    def clear(self, *, keep_presets: bool = True) -> None:
        """Drop every preference, optionally keeping the named presets."""
        self.values.clear()
        if not keep_presets:
            self.presets.clear()
        self.save()

    # -- presets ----------------------------------------------------------
    def preset_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.presets))

    def save_preset(self, name: str, values: dict[str, object] | None = None) -> tuple[str, ...]:
        """Save a named preset, defaulting to the current preferences.

        Returns the setting names it holds, so a caller can report what was saved.
        """
        cleaned = self._clean(values if values is not None else dict(self.values))
        if not cleaned:
            raise ValueError("a preset needs at least one usable setting")
        self.presets[str(name)] = cleaned
        self.save()
        return tuple(sorted(cleaned))

    def delete_preset(self, name: str) -> bool:
        existed = self.presets.pop(str(name), None) is not None
        if existed:
            self.save()
        return existed

    def preset(self, name: str) -> dict[str, object]:
        return dict(self.presets.get(str(name), {}))

    # -- resolution -------------------------------------------------------
    def resolve(
        self,
        *,
        preset: str | None = None,
        cli: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """The effective value of every setting, with later layers winning.

        ``cli`` values that are ``None`` are ignored, because a flag that was not
        given must not overwrite a preference.
        """
        resolved: dict[str, object] = {s.name: s.default for s in SETTINGS}
        resolved.update(self.values)
        if preset:
            resolved.update(self.presets.get(str(preset), {}))
        for name, value in (cli or {}).items():
            if value is not None and name in BY_NAME:
                resolved[name] = value
        return resolved

    def origins(self, *, preset: str | None = None) -> dict[str, str]:
        """Where each effective value came from, for display."""
        origins = {s.name: "default" for s in SETTINGS}
        for name in self.values:
            origins[name] = "preference"
        if preset:
            for name in self.presets.get(str(preset), {}):
                origins[name] = "preset"
        return origins

    def apply_to(
        self,
        parser: argparse.ArgumentParser,
        values: dict[str, object] | None = None,
    ) -> tuple[str, ...]:
        """Make these preferences the parser's defaults.

        Layering falls out of this: a flag given on the command line overrides a
        parser default, and a parser default overrides the built-in one. Only values
        that were actually chosen are applied, so a change to a built-in default is
        never shadowed by a stale copy of itself.

        ``values`` overrides the stored preferences, which is how a preset is applied
        on top of them without being written to disk.
        """
        chosen = dict(self.values if values is None else values)
        applied: list[str] = []
        for target in self._walk(parser):
            for action in target._actions:  # noqa: SLF001 - argparse has no alternative
                name = action.dest
                if name not in chosen:
                    continue
                if action.default is not None and action.default == chosen[name]:
                    continue
                action.default = chosen[name]
                applied.append(name)
        return tuple(sorted(applied))

    @staticmethod
    def _walk(parser: argparse.ArgumentParser):
        """Yield a parser and every subparser it holds.

        Subparsers are essential here, not incidental: every run option lives on the
        ``run`` subparser, so applying preferences to the top-level parser alone left
        them with no effect at all.
        """
        yield parser
        for action in parser._actions:  # noqa: SLF001
            children = getattr(action, "choices", None)
            if not isinstance(children, dict):
                continue
            for child in children.values():
                if isinstance(child, argparse.ArgumentParser):
                    yield from Preferences._walk(child)

    def as_dict(self) -> dict[str, object]:
        """The whole store, for showing or exporting."""
        return {"values": dict(self.values), "presets": dict(self.presets)}
