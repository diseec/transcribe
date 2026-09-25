"""Reading a run's own numbers, and refusing to call a short transcript complete.

The scenario pinned down here really happened: a 52-minute recording was transcribed
into 143 tidy speaker-labelled lines covering 42% of the audio, and reported as a
success. Nothing about a shortened transcript looks wrong, so "is this complete?" has
to be answered from measurements rather than from the fact that the run ended.
"""

import unittest

from whisperx_local.services.report import COVERAGE_FLOOR, Gap, RunReport, union_seconds


def report(**overrides) -> RunReport:
    base = dict(duration=1000.0, segments=500, words=9000, chunks=6, covered=1000.0)
    base.update(overrides)
    return RunReport(**base)


class UnionSecondsTest(unittest.TestCase):
    """Chunks overlap on purpose, so their lengths cannot simply be added."""

    def test_nothing_covers_nothing(self):
        self.assertEqual(union_seconds([]), 0.0)

    def test_overlapping_intervals_are_counted_once(self):
        self.assertEqual(union_seconds([(0.0, 10.0), (8.0, 20.0)]), 20.0)

    def test_touching_intervals_join_without_a_seam(self):
        self.assertEqual(union_seconds([(0.0, 10.0), (10.0, 20.0)]), 20.0)

    def test_a_gap_is_left_out(self):
        self.assertEqual(union_seconds([(0.0, 5.0), (10.0, 20.0)]), 15.0)

    def test_unordered_input_is_sorted(self):
        self.assertEqual(union_seconds([(10.0, 20.0), (0.0, 5.0)]), 15.0)

    def test_an_interval_inside_another_is_absorbed(self):
        self.assertEqual(union_seconds([(0.0, 100.0), (10.0, 20.0)]), 100.0)

    def test_a_degenerate_interval_is_ignored(self):
        self.assertEqual(union_seconds([(5.0, 5.0)]), 0.0)


class PartialDetectionTest(unittest.TestCase):
    def test_a_full_run_is_complete(self):
        self.assertFalse(report().partial)
        self.assertEqual(report().state, "COMPLETE")

    def test_text_that_stops_at_42_percent_is_partial_with_no_error_anywhere(self):
        # The real failure: nothing raised, nothing logged, the text simply stopped.
        thin = report(covered=420.0, words=3006)
        self.assertTrue(thin.partial)
        self.assertEqual(thin.state, "PARTIAL")
        self.assertEqual(thin.unaccounted, 580.0)

    def test_the_floor_is_exactly_where_the_shortfall_starts(self):
        self.assertFalse(report(covered=COVERAGE_FLOOR * 1000.0).partial)
        self.assertTrue(report(covered=COVERAGE_FLOOR * 1000.0 - 1.0).partial)

    def test_a_named_gap_is_partial(self):
        holed = report(gaps=[Gap(3, 600.0, 700.0, "not transcribed")])
        self.assertTrue(holed.partial)
        self.assertEqual(holed.gap_lines(), ["600s–700s (not transcribed)"])

    def test_a_recording_that_is_quiet_in_places_is_not_a_failure(self):
        # Silence with no speech in it is explained, so it is not unaccounted for.
        quiet = report(covered=0.0, silent_seconds=1000.0)
        self.assertFalse(quiet.partial)
        self.assertEqual(quiet.unaccounted, 0.0)

    def test_a_failed_refinement_is_partial_even_when_the_text_is_whole(self):
        self.assertTrue(
            report(diarization_requested=True, diarization_complete=False).partial
        )
        self.assertTrue(
            report(alignment_requested=True, alignment_complete=False).partial
        )

    def test_a_refinement_that_was_asked_for_and_finished_is_not_partial(self):
        self.assertFalse(
            report(
                alignment_requested=True,
                alignment_complete=True,
                diarization_requested=True,
                diarization_complete=True,
            ).partial
        )

    def test_a_refinement_that_was_never_asked_for_cannot_make_a_run_partial(self):
        self.assertFalse(
            report(
                alignment_requested=False,
                alignment_complete=False,
                diarization_requested=False,
                diarization_complete=False,
            ).partial
        )

    def test_a_zero_length_recording_is_not_called_partial(self):
        self.assertFalse(report(duration=0.0, covered=0.0).partial)


class AdviceTest(unittest.TestCase):
    """A partial result has to say what to do next, or it just worries the reader."""

    def test_a_complete_run_needs_no_advice(self):
        self.assertIsNone(report().advice())

    def test_a_missing_chunk_is_answered_with_a_rerun(self):
        advice = report(gaps=[Gap(0, 0.0, 10.0, "not transcribed")]).advice()
        self.assertIn("Re-run", advice)

    def test_an_unexplained_shortfall_says_the_cause_was_never_recorded(self):
        self.assertIn("no cause was recorded", report(covered=400.0).advice())

    def test_a_failed_stage_is_answered_with_that_stage(self):
        advice = report(
            diarization_requested=True, diarization_complete=False
        ).advice()
        self.assertIn("run on its own", advice)


class DisplayTest(unittest.TestCase):
    def test_the_summary_names_the_share_covered(self):
        self.assertEqual(report(covered=500.0).headline(), "500s of 1000s (50%)")

    def test_the_rows_state_the_result(self):
        rows = dict(report().lines())
        self.assertEqual(rows["Result"], "COMPLETE")
        self.assertEqual(rows["Covered"], "1000s of 1000s (100%)")

    def test_unaccounted_audio_is_shown_when_there_is_any(self):
        rows = dict(report(covered=400.0).lines())
        self.assertEqual(rows["Unaccounted"], "600s with no text")

    def test_silence_is_reported_as_silence(self):
        rows = dict(report(silent_seconds=120.0).lines())
        self.assertEqual(rows["No speech"], "120s of silence")

    def test_a_stage_that_did_not_finish_says_so(self):
        rows = dict(report(alignment_requested=True, alignment_complete=False).lines())
        self.assertEqual(rows["Word timing"], "NOT completed")

    def test_the_raw_copy_is_named_because_that_is_the_one_to_keep(self):
        rows = dict(report(canonical_files=["meeting.raw.txt"]).lines())
        self.assertEqual(rows["Raw copy"], "meeting.raw.txt")


if __name__ == "__main__":
    unittest.main()
