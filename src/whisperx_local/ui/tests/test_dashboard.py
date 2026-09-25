"""The dashboard's own rules, most of which exist because a version broke them.

The progress display is the part of this program a user stares at for half an hour, and
it has been wrong in specific ways: a bar that filled and reset once per stage, a bar that
froze during model loading, a stage label showing the previous stage's chunk count, and
warnings that were printed and then erased by the next repaint. Each of those has a test
here, because they are not visible in a passing run -- they are visible in a *failing* one,
which is exactly when nobody is reading the code.

The live region is not entered in these tests. ``_refresh`` returns immediately when there
is no live display, so the bookkeeping can be driven directly and asserted on, which is
the part that can be wrong.
"""

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rich.console import Console

from whisperx_local.ui import dashboard
from whisperx_local.ui.dashboard import AccuracyStudio


class StudioTestCase(unittest.TestCase):
    """A studio that never takes over the screen, with its output captured."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "meeting.txt"
        self.log = self.root / "meeting.log"
        self.studio = AccuracyStudio(output=self.output, log_path=self.log)
        self.addCleanup(lambda: self.studio.__exit__(None, None, None))
        self.text = io.StringIO()
        self.studio.console = Console(file=self.text, width=120)

    def printed(self) -> str:
        return self.text.getvalue()


class TerminalDetectionTest(unittest.TestCase):
    """A bar cannot animate into a file, and an unobservable run looks like a hang."""

    def build(self, is_terminal: bool) -> AccuracyStudio:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(dashboard, "Console") as console:
                console.return_value.is_terminal = is_terminal
                studio = AccuracyStudio(
                    output=root / "out.txt", log_path=root / "run.log"
                )
        return studio

    def test_redirected_output_gets_plain_lines(self):
        self.assertTrue(self.build(False)._plain)

    def test_a_terminal_gets_the_live_bar(self):
        self.assertFalse(self.build(True)._plain)


class AnnounceTest(StudioTestCase):
    def test_a_plain_run_says_where_it_is(self):
        self.studio._plain = True
        self.studio._announce(force=True)
        self.assertIn("%", self.printed())

    def test_the_line_is_rate_limited_so_it_does_not_flood_a_log(self):
        self.studio._plain = True
        self.studio._announce(force=True)
        first = self.printed()
        self.studio._announce()
        self.assertEqual(self.printed(), first)

    def test_a_stage_change_is_always_announced(self):
        # The rate limit must not be able to swallow the line that says a new phase began.
        self.studio._plain = True
        self.studio._announce(force=True)
        self.studio.begin_stage("Transcribing")
        self.assertIn("Transcribing", self.printed())


class WarningQueueTest(StudioTestCase):
    """Warnings used to be printed and then erased by the next repaint."""

    def setUp(self):
        super().setUp()
        # The path under test is the live one; plain mode prints immediately instead.
        self.studio._plain = False

    def test_a_warning_is_not_printed_while_the_display_is_live(self):
        self.studio.warn("free memory is low; batch size reduced")
        self.assertNotIn("batch size reduced", self.printed())

    def test_a_warning_is_kept_until_it_can_be_shown(self):
        self.studio.warn("something to know")
        self.assertEqual(self.studio._pending, ["something to know"])

    def test_warnings_appear_once_the_display_closes(self):
        self.studio.warn("something to know")
        self.studio.__exit__(None, None, None)
        self.assertIn("something to know", self.printed())

    def test_shown_warnings_are_not_shown_again(self):
        self.studio.warn("once only")
        self.studio.__exit__(None, None, None)
        before = self.printed()
        self.studio._flush_notices()
        self.assertEqual(self.printed(), before)

    def test_several_warnings_all_survive(self):
        for index in range(3):
            self.studio.warn(f"note {index}")
        self.studio.__exit__(None, None, None)
        for index in range(3):
            self.assertIn(f"note {index}", self.printed())


class StageTest(StudioTestCase):
    def test_an_unknown_stage_name_is_ignored(self):
        # A typo in a stage name must not silently create a stage the bar waits on.
        before = self.studio.current_stage
        self.studio.begin_stage("Not a real stage")
        self.assertEqual(self.studio.current_stage, before)

    def test_starting_the_next_stage_closes_the_previous_one(self):
        # Leaving it open made the bar wait on work that had already finished.
        self.studio.begin_stage("Transcribing")
        self.studio.begin_stage("Aligning words")
        self.assertTrue(self.studio.model.is_done("transcribe"))

    def test_ending_a_stage_closes_it(self):
        self.studio.begin_stage("Transcribing")
        self.studio.end_stage()
        self.assertTrue(self.studio.model.is_done("transcribe"))

    def test_ending_a_stage_twice_is_harmless(self):
        self.studio.begin_stage("Transcribing")
        self.studio.end_stage()
        self.studio.end_stage()

    def test_a_new_stage_forgets_the_previous_chunk_counts(self):
        # This is how the label came to read "Aligning words 3/3" before aligning began.
        self.studio.begin_stage("Transcribing")
        self.studio.set_chunks(3, 3)
        self.studio.begin_stage("Aligning words")
        self.assertEqual((self.studio.chunk_position, self.studio.chunk_total), (1, 1))

    def test_skipping_a_stage_stops_the_bar_waiting_for_it(self):
        # The dashboard keys stages by its own names, not the action names: the speaker
        # stage is ``speakers``, and asking for it by its label has to reach that key.
        self.studio.skip_stage("Identifying speakers")
        self.assertTrue(self.studio.model.is_done("speakers"))

    def test_skipping_an_unknown_stage_is_ignored(self):
        self.studio.skip_stage("Nonsense")

    def test_every_stage_the_pipeline_names_is_known_here(self):
        # A stage the dashboard does not recognise shows no progress at all, silently.
        named = (
            "Preparing audio",
            "Finding speech",
            "Transcribing",
            "Aligning words",
            "Identifying speakers",
            "Writing transcript",
        )
        for label in named:
            with self.subTest(stage=label):
                self.assertIn(label, self.studio._labels)

    def test_finishing_closes_every_stage(self):
        self.studio.begin_stage("Transcribing")
        self.studio.finish()
        self.assertTrue(
            all(self.studio.model.is_done(stage.key) for stage in self.studio.model.stages)
        )

    def test_a_stage_can_be_started_by_its_key_too(self):
        self.studio.begin_stage("transcribe")
        self.assertEqual(self.studio._started_stage, "transcribe")


class ProgressTest(StudioTestCase):
    def test_progress_moves_the_bar(self):
        self.studio.begin_stage("Transcribing")
        self.studio.progress_to(50.0)
        self.assertGreater(self.studio.model.percent(), 0.0)

    def test_progress_records_that_something_happened(self):
        # The idle detector keys off this; without it a busy run is called idle.
        self.studio.begin_stage("Transcribing")
        before = self.studio._last_advance
        self.studio.progress_to(10.0)
        self.assertGreaterEqual(self.studio._last_advance, before)

    def test_the_word_total_can_be_set_for_a_resumed_run(self):
        self.studio.set_words(1234)
        self.assertEqual(self.studio.words, 1234)

    def test_the_speaker_count_is_kept(self):
        self.studio.set_speakers(3)
        self.assertEqual(self.studio.speakers, 3)

    def test_the_partial_copy_is_recorded(self):
        self.studio.mark_partial()
        self.assertTrue(self.studio.partial_ready)
        self.studio.mark_partial(False)
        self.assertFalse(self.studio.partial_ready)


class AbsorbTest(StudioTestCase):
    """Engine output is read for progress, but never for which stage is running."""

    def test_a_progress_line_moves_the_bar(self):
        self.studio.begin_stage("Transcribing")
        self.studio.absorb("Progress: 42.5%")
        self.assertGreater(self.studio.model.percent(), 0.0)

    def test_a_transcript_line_becomes_an_excerpt(self):
        self.studio.absorb("Transcript: [0.00 --> 2.00]  سلام خوبی")
        self.assertTrue(any("سلام خوبی" in excerpt for excerpt in self.studio.excerpts))

    def test_a_whispercpp_progress_line_moves_the_bar(self):
        # whisper.cpp writes ``whisper_print_progress_callback: progress =  42%``. The
        # pattern matched ``Progress:`` alone, so none of those lines registered and the
        # bar sat still for the whole run, which reads as a hang.
        self.studio.begin_stage("Transcribing")
        self.studio.absorb("whisper_print_progress_callback: progress =  42%")
        self.assertGreater(self.studio.model.percent(), 0.0)

    def test_a_whispercpp_segment_line_becomes_an_excerpt(self):
        # It prints a bare timestamped line, with no ``Transcript:`` prefix.
        self.studio.absorb("[00:00:00.031 --> 00:00:03.560]  بخواد برای بعضی از کارمنداش")
        self.assertTrue(any("بخواد" in excerpt for excerpt in self.studio.excerpts))

    def test_a_whispercpp_segment_counts_towards_the_word_total(self):
        self.studio.absorb("[00:00:00.031 --> 00:00:03.560]  one two three four")
        self.assertEqual(self.studio.words, 4)

    def test_the_engines_own_chatter_is_not_mistaken_for_speech(self):
        # whisper.cpp prints its configuration as ``key = value`` lines. None is a
        # timestamped segment, and counting one would invent words that were never said.
        before = self.studio.words
        self.studio.absorb("whisper_full_with_state: too many decoders requested (10), max = 8")
        self.studio.absorb("system_info: n_threads = 8 / 10 | WHISPER : COREML = 0")
        self.assertEqual(self.studio.words, before)

    def test_an_excerpt_counts_towards_the_word_total(self):
        self.studio.absorb("Transcript: [0.00 --> 2.00]  one two three")
        self.assertEqual(self.studio.words, 3)

    def test_a_stage_name_in_an_unrelated_line_does_not_change_the_stage(self):
        # Guessing the stage from text made the bar hop whenever a line mentioned one.
        self.studio.begin_stage("Transcribing")
        self.studio.absorb("Aligning words is a separate stage")
        self.assertEqual(self.studio.current_stage, "Transcribing")

    def test_an_unrecognised_line_changes_nothing(self):
        before = self.studio.model.percent()
        self.studio.absorb("nothing useful here")
        self.assertEqual(self.studio.model.percent(), before)

    def test_the_excerpt_list_is_bounded(self):
        for index in range(20):
            self.studio.absorb(f"Transcript: [0.00 --> 1.00]  line {index}")
        self.assertLessEqual(len(self.studio.excerpts), 4)


class RenderTest(StudioTestCase):
    """A crash while painting must not be able to kill a run, so rendering has to hold."""

    def test_it_renders_before_anything_has_happened(self):
        self.assertIsNotNone(self.studio.render())

    def test_it_renders_mid_run(self):
        self.studio.begin_stage("Transcribing")
        self.studio.set_chunks(2, 6, offset=120.0)
        self.studio.set_status("recognising speech")
        self.assertIsNotNone(self.studio.render())


class LogTest(StudioTestCase):
    def test_diagnostics_go_to_the_log_and_not_to_the_user(self):
        self.studio.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.studio._log = self.log.open("w", encoding="utf-8")
        self.studio.log_line("child process said something dull")
        self.studio._log.flush()
        self.assertIn("child process said something dull", self.log.read_text())
        self.assertNotIn("child process said something dull", self.printed())

    def test_a_line_without_a_newline_still_starts_a_new_log_line(self):
        self.studio.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.studio._log = self.log.open("w", encoding="utf-8")
        self.studio.log_line("first")
        self.studio.log_line("second")
        self.studio._log.flush()
        self.assertEqual(self.log.read_text().splitlines(), ["first", "second"])

    def test_logging_without_an_open_log_is_harmless(self):
        self.studio.log_line("nowhere to put this")


if __name__ == "__main__":
    unittest.main()
