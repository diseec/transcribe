"""Grouping words into lines, absorbing flicker, and naming speakers.

These rules decide whether a word gets the right label or a plausible-looking wrong
one, and whether a real handover survives or is smoothed away. Both directions are
tested, because a rule that absorbs too much is as damaging as one that absorbs too
little and is much harder to notice.
"""

import unittest

from whisperx_local.transcript.assemble import (
    build_lines,
    prepare_lines,
    rename_speakers,
    smooth_lines,
)
from whisperx_local.transcript.tests.support import SPLIT_SEGMENTS, cleaned_raw
from whisperx_local.transcript.turns import Line


def line(speaker, start, end, *words):
    return Line(speaker, start, end, [{"word": word, "start": start, "end": end} for word in words])


class BuildLinesTest(unittest.TestCase):
    def setUp(self):
        self.turns = cleaned_raw()

    def test_no_word_is_dropped(self):
        lines = build_lines(SPLIT_SEGMENTS, self.turns)
        self.assertEqual(sum(len(line.words) for line in lines), 4)

    def test_every_line_gets_a_speaker(self):
        # A blank label is worse than a wrong one: it silently loses who spoke.
        lines = build_lines(SPLIT_SEGMENTS, self.turns)
        self.assertTrue(all(line.speaker is not None for line in lines))

    def test_speaker_changes_split_lines(self):
        self.assertGreaterEqual(len(build_lines(SPLIT_SEGMENTS, self.turns)), 2)

    def test_segment_without_words_keeps_its_text(self):
        lines = build_lines([{"start": 1.5, "end": 2.6, "text": "بدون کلمه"}], self.turns)
        self.assertEqual(lines[0].text, "بدون کلمه")

    def test_a_long_pause_starts_a_new_line(self):
        segments = [
            {
                "start": 0.0,
                "end": 20.0,
                "words": [
                    {"word": "a", "start": 0.0, "end": 1.0},
                    {"word": "b", "start": 15.0, "end": 16.0},
                ],
            }
        ]
        turns = [Line("A", 0.0, 20.0, [])]
        from whisperx_local.transcript.turns import Turn

        self.assertEqual(len(build_lines(segments, [Turn(0.0, 20.0, "SPEAKER_00")])), 2)

    def test_a_long_run_is_split_by_the_character_budget(self):
        words = [
            {"word": "word", "start": index * 0.2, "end": index * 0.2 + 0.2}
            for index in range(60)
        ]
        from whisperx_local.transcript.turns import Turn

        lines = build_lines(
            [{"start": 0.0, "end": 12.0, "words": words}],
            [Turn(0.0, 12.0, "SPEAKER_00")],
            max_chars=40,
        )
        self.assertGreater(len(lines), 1)
        self.assertTrue(all(len(line.text) <= 45 for line in lines))

    def test_no_segments_gives_no_lines(self):
        self.assertEqual(build_lines([], self.turns), [])


class SmoothLinesTest(unittest.TestCase):
    def test_flicker_surrounded_by_one_speaker_is_absorbed(self):
        lines = [line("A", 0.0, 2.0, "i"), line("B", 2.0, 2.3, "ii"), line("A", 2.3, 4.0, "iii")]
        smoothed = smooth_lines(lines)
        self.assertEqual(len(smoothed), 1)
        self.assertEqual(len(smoothed[0].words), 3)
        self.assertEqual(smoothed[0].speaker, "A")

    def test_short_repeat_of_previous_speaker_is_absorbed(self):
        lines = [line("A", 0.0, 2.0, "i"), line("A", 3.0, 3.2, "ii")]
        self.assertEqual(len(smooth_lines(lines)), 1)

    def test_genuine_handover_survives(self):
        lines = [line("A", 0.0, 2.0, "i"), line("B", 2.0, 3.5, "ii"), line("A", 3.5, 5.0, "iii")]
        self.assertEqual(len(smooth_lines(lines)), 3)

    def test_a_long_utterance_between_speakers_is_never_absorbed(self):
        # Multi-word and long: this is a real contribution, however inconvenient.
        long_line = Line(
            "B", 2.0, 5.0,
            [{"word": f"w{index}", "start": 2.0 + index, "end": 2.5 + index} for index in range(4)],
        )
        lines = [line("A", 0.0, 2.0, "i"), long_line, line("A", 5.0, 7.0, "iii")]
        self.assertEqual(len(smooth_lines(lines)), 3)

    def test_input_is_not_mutated(self):
        lines = [line("A", 0.0, 2.0, "i"), line("B", 2.0, 2.3, "ii"), line("A", 2.3, 4.0, "iii")]
        before = [list(entry.words) for entry in lines]
        smooth_lines(lines)
        self.assertEqual([list(entry.words) for entry in lines], before)

    def test_a_single_line_is_returned_unchanged(self):
        lines = [line("A", 0.0, 2.0, "i")]
        self.assertEqual(len(smooth_lines(lines)), 1)

    def test_no_lines_is_fine(self):
        self.assertEqual(smooth_lines([]), [])


class NamingTest(unittest.TestCase):
    def test_speakers_are_numbered_by_first_appearance(self):
        lines = [line("SPEAKER_02", 0, 1, "a"), line("SPEAKER_00", 1, 2, "b")]
        named = rename_speakers(lines)
        self.assertEqual([entry.speaker for entry in named], ["Speaker 1", "Speaker 2"])

    def test_an_unlabelled_line_is_left_alone(self):
        lines = [line(None, 0, 1, "a"), line("SPEAKER_00", 1, 2, "b")]
        named = rename_speakers(lines)
        self.assertIsNone(named[0].speaker)
        self.assertEqual(named[1].speaker, "Speaker 1")

    def test_empty_input_is_fine(self):
        self.assertEqual(rename_speakers([]), [])


class SmoothingFidelityTest(unittest.TestCase):
    """What smoothing must *not* do.

    Absorbing too much is the more damaging direction and the harder one to notice: a
    merged line reads perfectly well. These pin the cases where merging would move one
    speaker's words onto somebody else's line.
    """

    def test_a_one_word_reply_keeps_its_own_speaker(self):
        # ``بله`` between two turns by A is a real contribution, not a flicker.
        lines = [
            line("A", 0.0, 2.0, "i"),
            line("B", 2.0, 2.8, "بله"),
            line("A", 2.8, 4.0, "iii"),
        ]
        smoothed = smooth_lines(lines)
        self.assertEqual(len(smoothed), 3)
        self.assertEqual(smoothed[1].speaker, "B")

    def test_a_sub_half_second_flicker_is_still_absorbed(self):
        lines = [
            line("A", 0.0, 2.0, "i"),
            line("B", 2.0, 2.2, "ii"),
            line("A", 2.2, 4.0, "iii"),
        ]
        self.assertEqual(len(smooth_lines(lines)), 1)

    def test_smoothing_never_deletes_a_word(self):
        # Whatever is absorbed, the text survives; only the grouping may change.
        lines = [
            line("A", 0.0, 2.0, "one", "two"),
            line("B", 2.0, 2.7, "بله"),
            line("A", 2.7, 4.0, "three"),
        ]
        words = [word["word"] for entry in smooth_lines(lines) for word in entry.words]
        self.assertEqual(words, ["one", "two", "بله", "three"])


class SegmentationPassthroughTest(unittest.TestCase):
    """With no speaker turns there is nothing to regroup for, so nothing is regrouped."""

    def test_the_recognition_segmentation_is_kept(self):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "یک"},
            {"start": 1.0, "end": 1.4, "text": "دو"},
            {"start": 1.4, "end": 5.0, "text": "سه"},
        ]
        lines = prepare_lines(segments, [])
        self.assertEqual([entry.text for entry in lines], ["یک", "دو", "سه"])

    def test_short_segments_are_not_merged_away(self):
        # Merging would be a presentational choice made on the reader's behalf.
        segments = [
            {"start": 0.0, "end": 3.0, "text": "long enough to be its own line"},
            {"start": 3.0, "end": 3.2, "text": "بله"},
        ]
        self.assertEqual(len(prepare_lines(segments, [])), 2)

    def test_no_speaker_is_invented(self):
        lines = prepare_lines([{"start": 0.0, "end": 1.0, "text": "سلام"}], [])
        self.assertIsNone(lines[0].speaker)

    def test_word_timings_are_kept_when_they_exist(self):
        words = [{"word": "سلام", "start": 0.0, "end": 0.8}]
        segments = [{"start": 0.0, "end": 0.8, "text": "سلام", "words": words}]
        self.assertEqual(prepare_lines(segments, [])[0].words, words)

    def test_it_says_the_same_thing_as_the_input(self):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "یک"},
            {"start": 1.0, "end": 2.0, "text": "دو"},
        ]
        joined = " ".join(entry.text for entry in prepare_lines(segments, []))
        self.assertEqual(joined, "یک دو")

    def test_turns_still_switch_on_the_speaker_path(self):
        lines = prepare_lines(SPLIT_SEGMENTS, cleaned_raw())
        self.assertTrue(all(entry.speaker is not None for entry in lines))

    def test_no_segments_is_fine(self):
        self.assertEqual(prepare_lines([], []), [])


if __name__ == "__main__":
    unittest.main()
