"""Preferences: validation, persistence, presets, and layer resolution.

The risky behaviours are pinned here: a damaged file must not lose saved defaults,
a typo must not silently become a default, and a flag that was not given must never
overwrite a preference.
"""

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from whisperx_local.config.preferences import Preferences
from whisperx_local.config.settings import (
    SETTINGS,
    InvalidSetting,
    UnknownSetting,
    coerce,
    describe,
)


class PreferencesTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.path = Path(self._temporary.name) / "settings.json"

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def store(self) -> Preferences:
        return Preferences.load(self.path)


class ValueTest(PreferencesTestCase):
    def test_a_value_round_trips_through_disk(self):
        self.store().set("language", "en")
        self.assertEqual(self.store().get("language"), "en")

    def test_values_are_coerced_to_their_declared_type(self):
        store = self.store()
        self.assertEqual(store.set("chunk_seconds", "300"), 300.0)
        self.assertIsInstance(store.set("chunk_seconds", "300"), float)
        self.assertEqual(store.set("beam_size", 5), 5)

    def test_booleans_accept_human_spellings(self):
        store = self.store()
        for text in ("true", "yes", "on", "1", "Y"):
            self.assertTrue(store.set("diarize", text))
        for text in ("false", "no", "off", "0", "N"):
            self.assertFalse(store.set("diarize", text))
        self.assertTrue(store.set("diarize", True))

    def test_an_unknown_setting_is_rejected(self):
        with self.assertRaises(UnknownSetting):
            self.store().set("nonsense", 1)

    def test_a_wrong_type_is_rejected(self):
        store = self.store()
        with self.assertRaises(InvalidSetting):
            store.set("chunk_seconds", "not a number")
        with self.assertRaises(InvalidSetting):
            store.set("diarize", "maybe")

    def test_a_value_outside_the_choices_is_rejected(self):
        store = self.store()
        with self.assertRaises(InvalidSetting):
            store.set("profile", "turbo")
        with self.assertRaises(InvalidSetting):
            store.set("language", "de")

    def test_unset_reports_whether_it_was_set(self):
        store = self.store()
        store.set("language", "en")
        self.assertTrue(store.unset("language"))
        self.assertFalse(store.unset("language"))
        self.assertIsNone(store.get("language"))

    def test_get_returns_the_fallback_for_an_unset_name(self):
        self.assertEqual(self.store().get("speakers", "none"), "none")

    def test_saving_creates_the_directory(self):
        deep = Path(self._temporary.name) / "a" / "b" / "settings.json"
        Preferences.load(deep).set("language", "en")
        self.assertTrue(deep.is_file())

    def test_clear_removes_values_but_keeps_presets(self):
        store = self.store()
        store.set("language", "en")
        store.save_preset("p", {"language": "fa"})
        store.clear()
        reloaded = self.store()
        self.assertEqual(reloaded.values, {})
        self.assertEqual(reloaded.preset_names(), ("p",))

    def test_clear_can_remove_presets_too(self):
        store = self.store()
        store.save_preset("p", {"language": "fa"})
        store.clear(keep_presets=False)
        self.assertEqual(self.store().preset_names(), ())

    def test_as_dict_exposes_values_and_presets(self):
        store = self.store()
        store.set("language", "en")
        store.save_preset("p", {"profile": "fast"})
        payload = store.as_dict()
        self.assertEqual(payload["values"]["language"], "en")
        self.assertEqual(payload["presets"]["p"]["profile"], "fast")


class DamagedFileTest(PreferencesTestCase):
    def test_a_missing_file_is_not_an_error(self):
        self.assertEqual(self.store().values, {})

    def test_a_damaged_file_is_tolerated(self):
        self.path.write_text("{ this is not json", encoding="utf-8")
        self.assertEqual(self.store().values, {})

    def test_a_file_from_another_schema_is_ignored(self):
        self.path.write_text(
            json.dumps({"schema": "ancient", "values": {"language": "en"}}),
            encoding="utf-8",
        )
        self.assertEqual(self.store().values, {})

    def test_unknown_and_unusable_entries_are_dropped_on_load(self):
        # Losing every saved default because of one bad line would be a worse
        # outcome than ignoring that line.
        self.path.write_text(
            json.dumps(
                {
                    "schema": "prefs-v1",
                    "values": {
                        "language": "en",
                        "nonsense": 1,
                        "profile": "turbo",
                        "chunk_seconds": "later",
                    },
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(self.store().values, {"language": "en"})

    def test_a_broken_preset_body_is_dropped(self):
        self.path.write_text(
            json.dumps(
                {
                    "schema": "prefs-v1",
                    "values": {},
                    "presets": {"good": {"language": "fa"}, "bad": "not a mapping"},
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(self.store().preset_names(), ("good",))


class PresetTest(PreferencesTestCase):
    def test_a_preset_round_trips(self):
        self.store().save_preset("persian-slow", {"language": "fa", "chunk_seconds": 300})
        reloaded = self.store()
        self.assertEqual(reloaded.preset_names(), ("persian-slow",))
        self.assertEqual(
            reloaded.preset("persian-slow"), {"language": "fa", "chunk_seconds": 300.0}
        )

    def test_a_preset_defaults_to_the_current_preferences(self):
        store = self.store()
        store.set("language", "en")
        store.set("chunk_seconds", 120)
        self.assertEqual(store.save_preset("mine"), ("chunk_seconds", "language"))

    def test_an_empty_preset_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store().save_preset("empty")

    def test_saving_the_same_name_replaces_it(self):
        store = self.store()
        store.save_preset("p", {"language": "fa"})
        store.save_preset("p", {"language": "en"})
        self.assertEqual(store.preset("p"), {"language": "en"})
        self.assertEqual(len(self.store().preset_names()), 1)

    def test_deleting_reports_whether_it_existed(self):
        store = self.store()
        store.save_preset("p", {"language": "fa"})
        self.assertTrue(store.delete_preset("p"))
        self.assertFalse(store.delete_preset("p"))
        self.assertEqual(self.store().preset_names(), ())

    def test_an_unknown_preset_resolves_to_nothing(self):
        self.assertEqual(self.store().preset("missing"), {})


class ResolutionTest(PreferencesTestCase):
    def test_layers_apply_in_order(self):
        store = self.store()
        store.set("language", "en")
        store.set("chunk_seconds", 120.0)
        store.save_preset("p", {"language": "fa"})
        resolved = store.resolve(preset="p", cli={"chunk_seconds": 60, "profile": "fast"})

        self.assertEqual(resolved["language"], "fa")  # preset beats preference
        self.assertEqual(resolved["chunk_seconds"], 60)  # flag beats everything
        self.assertEqual(resolved["profile"], "fast")  # flag beats the default
        self.assertEqual(resolved["min_turn"], 0.35)  # untouched default survives

    def test_a_flag_left_unset_does_not_overwrite_a_preference(self):
        store = self.store()
        store.set("language", "en")
        self.assertEqual(store.resolve(cli={"language": None})["language"], "en")

    def test_unknown_keys_in_a_flag_layer_are_ignored(self):
        self.assertNotIn("nonsense", self.store().resolve(cli={"nonsense": 1}))

    def test_every_setting_resolves_to_something(self):
        resolved = self.store().resolve()
        self.assertEqual(set(resolved), {setting.name for setting in SETTINGS})

    def test_origins_name_the_layer_that_won(self):
        store = self.store()
        store.set("language", "en")
        store.save_preset("p", {"profile": "fast"})
        origins = store.origins(preset="p")
        self.assertEqual(origins["language"], "preference")
        self.assertEqual(origins["profile"], "preset")
        self.assertEqual(origins["min_turn"], "default")


class ApplyToParserTest(PreferencesTestCase):
    def parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument("--language", default="fa")
        parser.add_argument("--chunk-seconds", dest="chunk_seconds", type=float, default=600.0)
        parser.add_argument("--profile", default="balanced")
        return parser

    def test_only_chosen_values_become_parser_defaults(self):
        store = self.store()
        store.set("language", "en")
        store.set("chunk_seconds", 120)
        parser = self.parser()
        self.assertEqual(store.apply_to(parser), ("chunk_seconds", "language"))
        self.assertEqual(parser.parse_args([]).language, "en")
        self.assertEqual(parser.parse_args([]).chunk_seconds, 120.0)
        self.assertEqual(parser.parse_args([]).profile, "balanced")

    def test_a_flag_still_overrides_a_preference(self):
        store = self.store()
        store.set("language", "en")
        parser = self.parser()
        store.apply_to(parser)
        self.assertEqual(parser.parse_args(["--language", "fa"]).language, "fa")

    def test_an_untouched_parser_is_left_alone(self):
        parser = self.parser()
        self.assertEqual(self.store().apply_to(parser), ())
        self.assertEqual(parser.parse_args([]).language, "fa")

    def test_unrelated_arguments_are_ignored(self):
        parser = self.parser()
        parser.add_argument("--audio")
        store = self.store()
        store.set("language", "en")
        self.assertEqual(store.apply_to(parser), ("language",))

    def test_options_on_a_subparser_are_reached(self):
        # Regression: run options live on the ``run`` subparser, so walking only the
        # top-level parser silently made every saved preference do nothing.
        top = argparse.ArgumentParser()
        subparsers = top.add_subparsers(dest="command", required=True)
        run = subparsers.add_parser("run")
        run.add_argument("--language", default="fa")
        run.add_argument("--chunk-seconds", dest="chunk_seconds", type=float, default=600.0)

        store = self.store()
        store.set("language", "en")
        applied = store.apply_to(top)
        self.assertEqual(applied, ("language",))
        self.assertEqual(top.parse_args(["run"]).language, "en")
        self.assertEqual(top.parse_args(["run", "--language", "fa"]).language, "fa")


class DeclarationTest(unittest.TestCase):
    def test_every_setting_is_described(self):
        rows = describe()
        self.assertEqual(len(rows), len(SETTINGS))
        names = [row[0] for row in rows]
        self.assertIn("language", names)
        self.assertIn("chunk_seconds", names)

    def test_choices_are_shown_for_enum_like_settings(self):
        row = next(row for row in describe() if row[0] == "profile")
        self.assertIn("fast", row[1])
        self.assertIn("|", row[1])

    def test_automatic_is_shown_for_settings_without_a_built_in_default(self):
        row = next(row for row in describe() if row[0] == "model")
        self.assertEqual(row[1], "automatic")

    def test_setting_names_are_unique(self):
        names = [setting.name for setting in SETTINGS]
        self.assertEqual(len(names), len(set(names)))

    def test_coerce_rejects_an_unknown_name(self):
        with self.assertRaises(UnknownSetting):
            coerce("nope", 1)


if __name__ == "__main__":
    unittest.main()
