"""The parser, and the guarantee that it cannot drift from the action catalog.

The composer offers options from the catalog and the command line exposes them as
flags. If those two ever disagree, a setting becomes reachable in one interface and
invisible in the other, which is invisible until someone complains. The parity test
here is what prevents it.
"""

import argparse
import unittest
from unittest import mock

from whisperx_local.actions import catalog
from whisperx_local.cli.options import build_parser, parser_with_preferences
from whisperx_local.config import Preferences


def run_dests() -> set[str]:
    """Every destination the ``run`` subcommand accepts."""
    parser = build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices["run"]._defaults) | {
                child.dest for child in action.choices["run"]._actions
            }
    raise AssertionError("no run subcommand found")


class ParityTest(unittest.TestCase):
    def test_every_catalog_option_is_reachable_as_a_flag(self):
        # This is the contract that keeps the composer and the command line together.
        dests = run_dests()
        for action in catalog.ACTIONS:
            for option in action.options:
                with self.subTest(option=option.name):
                    self.assertIn(option.name, dests)

    def test_every_action_can_be_requested_by_name(self):
        args = build_parser().parse_args(
            ["run", "a.wav", "--only", ",".join(catalog.keys())]
        )
        self.assertEqual(args.only, ",".join(catalog.keys()))

    def test_unknown_action_names_are_rejected(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["run", "a.wav", "--only", "summarise"])


class ParserTest(unittest.TestCase):
    def parse(self, argv):
        return build_parser().parse_args(argv)

    def test_a_recording_is_required(self):
        with self.assertRaises(SystemExit):
            self.parse(["run"])

    def test_defaults_are_none_where_a_profile_decides(self):
        args = self.parse(["run", "a.wav"])
        # None means "the user did not choose"; the profile fills it later.
        for name in ("model", "compute_type", "beam_size", "batch_size", "threads"):
            with self.subTest(name=name):
                self.assertIsNone(getattr(args, name))

    def test_behavioural_defaults_are_real_values(self):
        args = self.parse(["run", "a.wav"])
        self.assertEqual(args.language, "fa")
        self.assertTrue(args.normalize)
        self.assertTrue(args.silence_split)
        self.assertEqual(args.output_format, "txt")

    def test_boolean_negatives_are_available(self):
        args = self.parse(["run", "a.wav", "--no-normalize", "--no-diarize"])
        self.assertFalse(args.normalize)
        self.assertFalse(args.diarize)

    def test_only_and_without_are_separate_flags(self):
        args = self.parse(["run", "a.wav", "--only", "export", "--without", "align"])
        self.assertEqual(args.only, "export")
        self.assertEqual(args.without, "align")

    def test_a_preset_can_be_named(self):
        self.assertEqual(self.parse(["run", "a.wav", "--preset", "p"]).preset, "p")


class PreferencesTest(unittest.TestCase):
    def test_the_plain_builder_is_preference_free(self):
        # Tests build a parser directly, so their behaviour must not depend on the
        # user's saved file.
        self.assertEqual(build_parser().parse_args(["run", "a.wav"]).language, "fa")

    def test_preferences_become_defaults_and_flags_still_win(self):
        # The store is patched rather than written, so the test cannot leave a
        # preference behind on the machine running it.
        preferences = Preferences.load()
        preferences.values["language"] = "en"
        with mock.patch.object(Preferences, "load", return_value=preferences):
            parser = parser_with_preferences()
        self.assertEqual(parser.parse_args(["run", "a.wav"]).language, "en")
        self.assertEqual(
            parser.parse_args(["run", "a.wav", "--language", "fa"]).language, "fa"
        )

    def test_an_override_beats_a_preference(self):
        parser = parser_with_preferences(overrides={"language": "en"})
        self.assertEqual(parser.parse_args(["run", "a.wav"]).language, "en")


if __name__ == "__main__":
    unittest.main()
