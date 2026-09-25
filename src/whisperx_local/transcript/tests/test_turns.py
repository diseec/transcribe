"""Cleaning raw diarization and labelling a span."""

import unittest

from whisperx_local.transcript.tests.support import RAW_TURNS, cleaned_raw, turns_from
from whisperx_local.transcript.turns import Turn, clean_turns, merge_adjacent, speaker_for_span


class CleanTurnsTest(unittest.TestCase):
    def setUp(self):
        self.cleaned = clean_turns(turns_from(RAW_TURNS))

    def test_flickers_are_removed(self):
        self.assertLess(len(self.cleaned), len(RAW_TURNS))
        for turn in self.cleaned:
            self.assertGreaterEqual(turn.duration, 0.35)

    def test_a_short_flicker_is_removed_but_a_real_turn_survives(self):
        # RAW_TURNS holds two SPEAKER_00 turns: a 20ms flicker at 4.35s and a real
        # 1.46s contribution at 24.15s. Only the flicker may vanish, otherwise the
        # cleaning rule is throwing away genuine speech.
        before = sum(1 for _, _, speaker in RAW_TURNS if speaker == "SPEAKER_00")
        after = sum(1 for turn in self.cleaned if turn.speaker == "SPEAKER_00")
        self.assertEqual((before, after), (2, 1))

    def test_same_speaker_neighbours_merge(self):
        merged = merge_adjacent(
            [Turn(0.0, 1.0, "A"), Turn(1.1, 2.0, "A"), Turn(2.1, 3.0, "B")]
        )
        self.assertEqual([turn.speaker for turn in merged], ["A", "B"])
        self.assertAlmostEqual(merged[0].end, 2.0)

    def test_a_real_gap_is_not_bridged(self):
        merged = merge_adjacent([Turn(0.0, 1.0, "A"), Turn(5.0, 6.0, "A")])
        self.assertEqual(len(merged), 2)

    def test_empty_input_is_fine(self):
        self.assertEqual(clean_turns([]), [])


class SpeakerForSpanTest(unittest.TestCase):
    def setUp(self):
        self.turns = cleaned_raw()

    def test_overlap_wins(self):
        self.assertEqual(speaker_for_span(1.6, 2.0, self.turns), "SPEAKER_02")

    def test_majority_overlap_wins_when_two_turns_apply(self):
        turns = [Turn(0.0, 5.0, "A"), Turn(4.0, 10.0, "B")]
        # Span 3.5-6.0 overlaps A by 1.5s and B by 2.0s.
        self.assertEqual(speaker_for_span(3.5, 6.0, turns), "B")

    def test_gap_falls_back_to_nearest(self):
        self.assertIsNotNone(speaker_for_span(5.0, 5.2, self.turns))

    def test_a_distant_gap_uses_the_fallback(self):
        turns = [Turn(0.0, 1.0, "A")]
        self.assertEqual(speaker_for_span(60.0, 61.0, turns, fallback="A"), "A")

    def test_a_distant_gap_without_a_fallback_is_unlabelled(self):
        turns = [Turn(0.0, 1.0, "A")]
        self.assertIsNone(speaker_for_span(60.0, 61.0, turns))

    def test_no_turns_gives_the_fallback(self):
        self.assertIsNone(speaker_for_span(0.0, 1.0, []))

    def test_no_turns_passes_the_fallback_through(self):
        self.assertEqual(speaker_for_span(0.0, 1.0, [], fallback="A"), "A")

    def test_a_span_inside_a_turn_is_labelled_even_with_no_other_overlap(self):
        self.assertEqual(speaker_for_span(0.5, 0.6, [Turn(0.0, 1.0, "A")]), "A")


if __name__ == "__main__":
    unittest.main()
