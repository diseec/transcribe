"""The composer's command surface.

The composer is the second way into the same pipeline: every command maps to a function
the command line also uses, so the risk here is not that a command is missing but that one
fails quietly -- a setting that is accepted and then ignored, a toggle that undoes an
earlier toggle, an unknown command that stops the session instead of saying so. Those are
what is tested.
"""

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rich.console import Console

from whisperx_local.actions import catalog
from whisperx_local.config import Preferences
from whisperx_local.ui.composer import Composer, Session
from whisperx_local.ui.menu import Choice


def composer(**overrides) -> Composer:
    """A composer that prints nowhere and needs no terminal."""
    session = Session(
        preferences=Preferences(values={}, presets={}),
        files=overrides.pop("files", [Path("meeting.m4a")]),
        **overrides,
    )
    entry = Composer(session=session, run=lambda **_: None)
    entry.console = Console(file=io.StringIO(), width=120)
    return entry


class DispatchTest(unittest.TestCase):
    def test_the_common_spellings_reach_the_same_command(self):
        # ``files``/``add``, ``actions``/``action``, ``show``/``settings``, ``?``/``help``.
        for first, second in (
            ("files", "add"),
            ("actions", "action"),
            ("show", "settings"),
            ("help", "?"),
        ):
            with self.subTest(first=first, second=second):
                self.assertIs(
                    composer().dispatch(first),
                    composer().dispatch(second),
                )

    def test_an_unknown_command_says_what_is_known(self):
        with self.assertRaises(ValueError) as raised:
            composer().dispatch("flibbertigibbet")
        self.assertIn("unknown command", str(raised.exception))
        self.assertIn("help", str(raised.exception))

    def test_a_command_is_matched_regardless_of_case(self):
        composer().dispatch("HELP")

    def test_extra_spaces_do_not_matter(self):
        composer().dispatch("   actions    ")


class ActionsCommandTest(unittest.TestCase):
    def test_only_narrows_the_work_to_exactly_that(self):
        entry = composer()
        entry.dispatch("actions only export")
        self.assertEqual(entry.session.only, ("export",))

    def test_all_clears_the_selection(self):
        entry = composer()
        entry.dispatch("actions only export")
        entry.dispatch("actions all")
        self.assertEqual((entry.session.only, entry.session.without), ((), ()))

    def test_a_plus_enables_one_action(self):
        entry = composer()
        entry.dispatch("actions only export")
        entry.dispatch("actions +diarize")
        self.assertIn("diarize", entry.session.action_keys())

    def test_a_minus_disables_one_action(self):
        entry = composer()
        entry.dispatch("actions -transcribe")
        self.assertNotIn("transcribe", entry.session.action_keys())

    def test_a_bare_action_name_is_refused_with_the_right_syntax(self):
        with self.assertRaises(ValueError) as raised:
            composer().dispatch("actions diarize")
        self.assertIn("+diarize", str(raised.exception))

    def test_an_unknown_action_name_is_refused(self):
        with self.assertRaises(KeyError):
            composer().dispatch("actions +nonsense")

    def test_toggle_is_reversible_for_an_action_in_the_default_set(self):
        entry = composer()
        before = set(entry.session.action_keys())
        entry.dispatch("actions -transcribe")
        entry.dispatch("actions +transcribe")
        self.assertEqual(set(entry.session.action_keys()), before)

    def test_enabling_an_action_outside_the_default_set_adds_it(self):
        # Toggling is not a round trip for these: the default set has no diarize to
        # return to, so switching it on is a change rather than an undo.
        entry = composer()
        self.assertNotIn("diarize", entry.session.action_keys())
        entry.dispatch("actions +diarize")
        self.assertIn("diarize", entry.session.action_keys())


class SetCommandTest(unittest.TestCase):
    def test_a_value_is_recorded(self):
        entry = composer()
        entry.dispatch("set beam_size 7")
        self.assertEqual(entry.session.overrides["beam_size"], 7)

    def test_a_value_is_converted_to_the_option_type(self):
        entry = composer()
        entry.dispatch("set chunk_seconds 120")
        self.assertEqual(entry.session.overrides["chunk_seconds"], 120.0)

    def test_a_boolean_is_understood_from_words(self):
        entry = composer()
        entry.dispatch("set silence_split false")
        self.assertIs(entry.session.overrides["silence_split"], False)

    def test_a_choice_outside_the_allowed_values_is_refused(self):
        with self.assertRaises(ValueError) as raised:
            composer().dispatch("set compute_type float64")
        self.assertIn("int8", str(raised.exception))

    def test_a_setting_that_is_not_part_of_the_work_says_how_to_enable_it(self):
        entry = composer()
        entry.dispatch("actions only export")
        with self.assertRaises(ValueError) as raised:
            entry.dispatch("set min_turn 0.5")
        self.assertIn("not part of the current work", str(raised.exception))

    def test_set_with_no_arguments_shows_the_settings(self):
        entry = composer(files=[Path("meeting.m4a")])
        with mock.patch.object(entry, "command_show") as shown:
            entry.dispatch("set")
        shown.assert_called_once()

    def test_being_cancelled_at_the_prompt_records_nothing(self):
        from whisperx_local.ui import prompts

        entry = composer()
        with mock.patch.object(prompts, "ask", side_effect=prompts.PromptCancelled):
            entry.dispatch("set beam_size")
        self.assertNotIn("beam_size", entry.session.overrides)

    def test_an_empty_answer_at_the_prompt_records_nothing(self):
        from whisperx_local.ui import prompts

        entry = composer()
        with mock.patch.object(prompts, "ask", return_value=None):
            entry.dispatch("set beam_size")
        self.assertNotIn("beam_size", entry.session.overrides)


class PlanCommandTest(unittest.TestCase):
    """Planning needs a real recording, so only the missing-recording path is here.

    The composer raises rather than printing, and its loop turns the exception into a
    ``!`` line. That contract is what these check; the happy path belongs in a run.
    """

    def test_planning_without_a_recording_says_to_choose_one(self):
        with self.assertRaises(ValueError) as raised:
            composer(files=[]).dispatch("plan")
        self.assertIn("choose a recording", str(raised.exception))

    def test_the_message_names_the_command_that_fixes_it(self):
        with self.assertRaises(ValueError) as raised:
            composer(files=[]).dispatch("plan")
        self.assertIn("files add", str(raised.exception))


class RunCommandTest(unittest.TestCase):
    def test_running_without_a_recording_says_to_choose_one(self):
        with self.assertRaises(ValueError) as raised:
            composer(files=[]).dispatch("run")
        self.assertIn("choose a recording", str(raised.exception))

    def test_a_dry_run_is_asked_for_rather_than_executed_here(self):
        # The composer hands the request on with dry_run set; deciding what that means
        # is the run handler's job, not the composer's.
        calls: list[dict] = []
        session = Session(
            preferences=Preferences(values={}, presets={}), files=[Path("meeting.m4a")]
        )
        entry = Composer(session=session, run=lambda **kwargs: calls.append(kwargs))
        entry.console = Console(file=io.StringIO(), width=120)
        entry.dispatch("run --dry-run")
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["dry_run"], True)

    def test_running_asks_for_the_recording_and_the_request(self):
        calls: list[dict] = []
        session = Session(
            preferences=Preferences(values={}, presets={}), files=[Path("meeting.m4a")]
        )
        entry = Composer(session=session, run=lambda **kwargs: calls.append(kwargs))
        entry.console = Console(file=io.StringIO(), width=120)
        entry.dispatch("run")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["files"], [Path("meeting.m4a")])


class FilesCommandTest(unittest.TestCase):
    def test_files_with_no_arguments_shows_the_selection(self):
        entry = composer()
        entry.dispatch("files")
        self.assertIn("meeting.m4a", entry.console.file.getvalue())

    def test_clearing_forgets_everything(self):
        entry = composer()
        entry.dispatch("files clear")
        self.assertEqual(entry.session.files, [])

    def test_a_path_that_does_not_exist_is_refused(self):
        with self.assertRaises(ValueError):
            composer(files=[]).dispatch("files add /nowhere/at/all.m4a")

    def test_a_real_file_is_added(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "talk.m4a"
            path.write_bytes(b"audio")
            entry = composer(files=[])
            entry.dispatch(f"files add {path}")
            # Compared resolved: the temporary directory is a symlink on macOS, and the
            # composer stores where the path really is.
            self.assertEqual(
                [stored.resolve() for stored in entry.session.files], [path.resolve()]
            )


class SessionStateTest(unittest.TestCase):
    """The state the commands act on, tested without the commands."""

    def entry(self) -> Session:
        return Session(
            preferences=Preferences(values={}, presets={}), files=[Path("meeting.m4a")]
        )

    def test_a_toggle_off_and_on_restores_an_action_that_was_there(self):
        session = self.entry()
        session.toggle("transcribe", False)
        session.toggle("transcribe", True)
        self.assertIn("transcribe", session.action_keys())

    def test_the_enabled_set_never_contains_an_unknown_action(self):
        session = self.entry()
        session.toggle("analyze", True)
        self.assertTrue(set(session.action_keys()) <= set(catalog.keys()))

    def test_toggling_something_unknown_raises_with_the_valid_names(self):
        with self.assertRaises(KeyError) as raised:
            self.entry().toggle("nonsense", True)
        self.assertIn("prepare", str(raised.exception))

    def test_the_summary_names_the_recording(self):
        rows = dict(self.entry().summary_rows())
        self.assertIn("meeting.m4a", rows["Files"])

    def test_the_summary_says_when_nothing_is_chosen(self):
        session = Session(preferences=Preferences(values={}, presets={}), files=[])
        self.assertEqual(dict(session.summary_rows())["Files"], "none chosen")

    def test_a_session_choice_overrides_the_saved_preference(self):
        session = Session(
            preferences=Preferences(values={"beam_size": 3}, presets={}),
            files=[],
            overrides={"beam_size": 9},
        )
        resolved = session.preferences.resolve(cli=session.overrides)
        self.assertEqual(resolved["beam_size"], 9)


if __name__ == "__main__":
    unittest.main()
