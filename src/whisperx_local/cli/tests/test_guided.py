"""The guided flow's own decisions, tested without a terminal.

The screens are not tested here -- a test that presses arrow keys tells you less than
using the thing does. What is tested is the part that can be wrong quietly: which stages
a given answer selects, and whether the request that comes out still says what the screen
appeared to promise.
"""

import unittest
from pathlib import Path
from unittest import mock

from whisperx_local.actions import catalog
from whisperx_local.cli.guided import (
    CORE_STAGES,
    WORK,
    Request,
    _confirm_rows,
    _fine_tune,
    _value_text,
    apply_work,
    guided,
)
from whisperx_local.config import Preferences
from whisperx_local.ui.composer import Session
from whisperx_local.ui.menu import Choice, Menu


def session(**overrides) -> Session:
    base = dict(preferences=Preferences(values={}, presets={}), files=[Path("meeting.m4a")])
    base.update(overrides)
    return Session(**base)


class WorkChoiceTest(unittest.TestCase):
    def test_every_option_keeps_the_core_stages(self):
        for work in WORK:
            with self.subTest(work=work.label):
                self.assertTrue(set(CORE_STAGES) <= set(work.stages))

    def test_transcription_only_adds_nothing(self):
        self.assertEqual(WORK[0].stages, CORE_STAGES)

    def test_the_speaker_option_asks_for_speakers(self):
        speakers = next(work for work in WORK if work.diarize)
        self.assertIn("diarize", speakers.stages)
        self.assertNotIn("align", speakers.stages)

    def test_everything_asks_for_everything(self):
        everything = next(work for work in WORK if work.align and work.diarize and work.analyze)
        self.assertEqual(everything.stages, catalog.keys())

    def test_stages_come_back_in_dependency_order(self):
        everything = WORK[-1]
        self.assertEqual(everything.stages, tuple(action.key for action in catalog.ACTIONS))

    def test_no_option_selects_an_unknown_stage(self):
        known = set(catalog.keys())
        for work in WORK:
            with self.subTest(work=work.label):
                self.assertEqual(set(work.stages) - known, set())


class ApplyWorkTest(unittest.TestCase):
    def test_the_chosen_stages_become_the_request(self):
        entry = session()
        apply_work(entry, WORK[0])
        self.assertEqual(entry.only, WORK[0].stages)

    def test_speakers_are_switched_off_explicitly_when_not_chosen(self):
        # Left unset, this falls back to "on if a token is configured" and runs anyway.
        entry = session()
        apply_work(entry, WORK[0])
        self.assertIs(entry.overrides["diarize"], False)

    def test_speakers_are_switched_on_explicitly_when_chosen(self):
        entry = session()
        apply_work(entry, WORK[2])
        self.assertIs(entry.overrides["diarize"], True)

    def test_the_report_is_carried_by_the_stages_not_by_a_setting(self):
        # ``analyze`` is an action, not a stored setting; writing it as one crashed a
        # preview, because there is nothing to look it up in.
        entry = session()
        apply_work(entry, WORK[0])
        self.assertNotIn("analyze", entry.overrides)
        self.assertNotIn("analyze", entry.only)

    def test_a_previous_exclusion_does_not_survive_a_new_answer(self):
        entry = session(without=("diarize",))
        apply_work(entry, WORK[3])
        self.assertEqual(entry.without, ())
        self.assertIn("diarize", entry.only)

    def test_choosing_a_second_time_replaces_the_first_answer(self):
        entry = session()
        apply_work(entry, WORK[3])
        apply_work(entry, WORK[0])
        self.assertNotIn("diarize", entry.only)
        self.assertNotIn("align", entry.only)


class RequestTest(unittest.TestCase):
    def test_a_request_exposes_the_recording_it_is_about(self):
        request = Request(session=session(), dry_run=True)
        self.assertEqual(request.source, Path("meeting.m4a"))
        self.assertTrue(request.dry_run)


class GuidedCancellationTest(unittest.TestCase):
    def test_finding_no_recordings_ends_the_session_quietly(self):
        # No candidates and a cancelled path prompt must not raise, nest, or loop.
        menu = _RefusingMenu()
        self.assertIsNone(guided(Preferences(values={}, presets={}), menu=menu))


class FineTuneTest(unittest.TestCase):
    """The per-setting screen, which crashed for every option the moment it opened.

    It handed the prompt a saved-setting record for options that are adjustable but not
    savable, and the prompt read an attribute the two do not share.
    """

    def setUp(self):
        self.session = Session(preferences=Preferences(values={}, presets={}), files=[])
        apply_work(self.session, WORK[0])

    def offered(self) -> list[Choice]:
        """The rows the screen would build, without drawing anything."""
        options = catalog.options_for(self.session.only or catalog.DEFAULT_ACTIONS)
        return [Choice(option.name, "", value=option) for option in options]

    def test_every_offered_option_is_a_prompt_option_not_a_saved_setting(self):
        from whisperx_local.actions.base import Option

        for choice in self.offered():
            with self.subTest(option=choice.value.name):
                self.assertIsInstance(choice.value, Option)

    def test_an_option_that_cannot_be_saved_is_still_offered(self):
        # ``vad_onset`` is real and adjustable but is not a stored default; it is the
        # exact case that raised KeyError in `show` and AttributeError here.
        names = [choice.value.name for choice in self.offered()]
        self.assertIn("vad_onset", names)

    def test_every_offered_option_is_reachable_as_a_flag(self):
        # An option offered here that the parser does not accept would be a dead row.
        from whisperx_local.cli.tests.test_options import run_dests

        dests = run_dests()
        for choice in self.offered():
            with self.subTest(option=choice.value.name):
                self.assertIn(choice.value.name, dests)

    def test_the_screen_returns_when_done_is_chosen(self):
        done = len(self.offered())

        class Picking(Menu):
            def __init__(self):
                super().__init__(out=Sink())

            def banner(self, *_args, **_kwargs):
                return None

            def choose(self, *_args, **_kwargs):
                return done

        self.assertTrue(_fine_tune(Picking(), self.session))

    def test_choosing_a_setting_records_it_as_this_session_choice(self):
        from whisperx_local.ui import prompts

        rows = self.offered()
        target = next(index for index, row in enumerate(rows) if row.value.name == "beam_size")
        answers = iter([target, len(rows)])

        class Picking(Menu):
            def __init__(self):
                super().__init__(out=Sink())

            def banner(self, *_args, **_kwargs):
                return None

            def choose(self, *_args, **_kwargs):
                return next(answers)

        with mock.patch.object(prompts, "ask", return_value=7) as asked:
            self.assertTrue(_fine_tune(Picking(), self.session))
        self.assertEqual(self.session.overrides["beam_size"], 7)
        # The prompt was given an option it understands, and the current value.
        self.assertEqual(asked.call_args.args[0].name, "beam_size")

    def test_cancelling_the_prompt_leaves_the_value_alone(self):
        from whisperx_local.ui import prompts

        rows = self.offered()
        target = next(index for index, row in enumerate(rows) if row.value.name == "beam_size")
        answers = iter([target, len(rows)])

        class Picking(Menu):
            def __init__(self):
                super().__init__(out=Sink())

            def banner(self, *_args, **_kwargs):
                return None

            def choose(self, *_args, **_kwargs):
                return next(answers)

        with mock.patch.object(prompts, "ask", side_effect=prompts.PromptCancelled):
            _fine_tune(Picking(), self.session)
        self.assertNotIn("beam_size", self.session.overrides)

class ValueTextTest(unittest.TestCase):
    def test_an_unsupplied_value_says_automatic_rather_than_none(self):
        self.assertEqual(_value_text(None), "automatic")

    def test_a_boolean_reads_as_on_or_off(self):
        self.assertEqual(_value_text(True), "on")
        self.assertEqual(_value_text(False), "off")

    def test_anything_else_is_shown_as_it_is(self):
        self.assertEqual(_value_text(600.0), "600.0")


class ConfirmRowsTest(unittest.TestCase):
    """The last screen, which has to read as plain text because it is written as such."""

    def rows(self, work=WORK[0]) -> dict[str, str]:
        entry = Session(
            preferences=Preferences(values={}, presets={}), files=[Path("meeting.m4a")]
        )
        apply_work(entry, work)
        return dict(_confirm_rows(entry))

    def test_no_rich_markup_leaks_into_the_banner(self):
        # The composer's summary carries markup for a rich panel; this one is written
        # straight out, so markup would appear as literal square brackets.
        for label, value in self.rows().items():
            with self.subTest(label=label):
                self.assertNotIn("[", value)

    def test_what_will_run_is_named(self):
        self.assertIn("Transcribing", self.rows()["Will run"])

    def test_what_will_not_run_is_named_too(self):
        # A stage left out deliberately must not look the same as one forgotten.
        self.assertIn("Identifying speakers", self.rows()["Not running"])

    def test_naming_the_stages_that_are_off_stops_once_they_are_on(self):
        self.assertNotIn("Not running", self.rows(WORK[-1]))

    def test_the_recording_is_named(self):
        self.assertEqual(self.rows()["Recording"], "meeting.m4a")

    def test_the_language_is_spelled_out(self):
        self.assertEqual(self.rows()["Language"], "Persian")

    def test_a_boolean_reads_as_prose_not_as_python(self):
        self.assertEqual(self.rows()["Cut on pauses"], "on")

    def test_there_is_no_recording_line_when_nothing_is_chosen(self):
        entry = Session(preferences=Preferences(values={}, presets={}), files=[])
        apply_work(entry, WORK[0])
        self.assertEqual(dict(_confirm_rows(entry))["Recording"], "none chosen")


class Sink:
    """Swallows writes so a test can drive a screen without a terminal."""

    def write(self, _text):
        return 0

    def flush(self):
        return None

    def isatty(self):
        return False


class _RefusingMenu(Menu):
    """A menu that cancels at the first screen, standing in for pressing q."""

    def __init__(self):
        super().__init__(out=Sink())

    def banner(self, *_args, **_kwargs):
        return None

    def choose(self, *_args, **_kwargs):
        return None

    def ask_text(self, *_args, **_kwargs):
        return None


if __name__ == "__main__":
    unittest.main()
