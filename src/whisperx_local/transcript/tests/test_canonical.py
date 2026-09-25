"""The canonical transcript: what was recognised, kept as it was recognised.

Two properties are worth defending here, because both were lost once and both are
invisible when they are lost. It must not be cleaned -- a tidier file that dropped a
line looks exactly like a tidier file -- and it must exist even when a later stage
fails, because everything downstream can be recomputed and this cannot.
"""

import json
import tempfile
import unittest
from pathlib import Path

from whisperx_local.transcript.canonical import (
    coverage_seconds,
    render_raw_json,
    render_raw_text,
    usable_segments,
    write_canonical,
)

SEGMENTS = [
    {
        "start": 0.0,
        "end": 2.5,
        "text": " سلام ",
        "words": [{"word": "سلام", "start": 0.0, "end": 2.5}],
    },
    {"start": 2.5, "end": 4.0, "text": "خوبی"},
    {"start": 9.0, "end": 11.0, "text": "بله"},
]


class UsableSegmentsTest(unittest.TestCase):
    def test_text_with_no_timing_is_dropped(self):
        self.assertEqual(usable_segments([{"text": "orphan"}]), [])

    def test_a_blank_segment_is_dropped(self):
        self.assertEqual(usable_segments([{"start": 0.0, "end": 1.0, "text": "   "}]), [])

    def test_order_is_restored(self):
        shuffled = [SEGMENTS[2], SEGMENTS[0]]
        self.assertEqual(
            [segment["text"] for segment in usable_segments(shuffled)], [" سلام ", "بله"]
        )

    def test_every_usable_segment_survives(self):
        self.assertEqual(len(usable_segments(SEGMENTS)), 3)


class CoverageTest(unittest.TestCase):
    def test_coverage_is_the_furthest_point_reached(self):
        self.assertEqual(coverage_seconds(SEGMENTS), 11.0)

    def test_no_segments_means_no_coverage(self):
        self.assertEqual(coverage_seconds([]), 0.0)


class RawTextTest(unittest.TestCase):
    def test_each_segment_is_one_timestamped_line(self):
        body = render_raw_text(SEGMENTS)
        self.assertEqual(len(body.strip().splitlines()), 3)
        self.assertIn("[00:00:00.000 --> 00:00:02.500]", body)

    def test_surrounding_whitespace_inside_a_segment_is_trimmed(self):
        self.assertIn("]  سلام\n", render_raw_text(SEGMENTS))

    def test_an_empty_transcript_is_an_empty_file(self):
        self.assertEqual(render_raw_text([]), "")

    def test_nothing_is_merged_or_removed(self):
        # No smoothing, no joining, no filler removal: this is the safety copy, and it
        # has to be usable as evidence of what the recogniser actually produced.
        body = render_raw_text(SEGMENTS)
        for text in ("سلام", "خوبی", "بله"):
            self.assertIn(text, body)


class RawJsonTest(unittest.TestCase):
    def test_word_timings_are_kept_where_they_exist(self):
        segments = json.loads(render_raw_json(SEGMENTS, "fa"))["segments"]
        self.assertIn("words", segments[0])
        self.assertNotIn("words", segments[1])

    def test_word_timings_are_simply_absent_when_alignment_never_ran(self):
        plain = [{"start": 0.0, "end": 1.0, "text": "بله"}]
        self.assertNotIn("words", json.loads(render_raw_json(plain, "fa"))["segments"][0])

    def test_persian_survives_without_being_escaped(self):
        self.assertIn("بله", render_raw_json(SEGMENTS, "fa"))

    def test_the_language_is_recorded(self):
        self.assertEqual(json.loads(render_raw_json(SEGMENTS, "fa"))["language"], "fa")

    def test_the_recognisers_own_confidence_is_kept(self):
        # Without ground truth this is the only signal about which engine struggled, so
        # dropping it left a comparison unable to say anything but "they disagree".
        scored = [{**SEGMENTS[0], "avg_logprob": -0.25}]
        segments = json.loads(render_raw_json(scored, "fa"))["segments"]
        self.assertAlmostEqual(segments[0]["avg_logprob"], -0.25)

    def test_a_segment_without_a_confidence_score_does_not_gain_a_null(self):
        segments = json.loads(render_raw_json(SEGMENTS, "fa"))["segments"]
        self.assertNotIn("avg_logprob", segments[0])

    def test_a_confidence_that_is_not_a_number_is_ignored(self):
        odd = [{**SEGMENTS[0], "avg_logprob": "unknown"}]
        segments = json.loads(render_raw_json(odd, "fa"))["segments"]
        self.assertNotIn("avg_logprob", segments[0])

    def test_it_declares_itself_the_canonical_copy(self):
        self.assertTrue(json.loads(render_raw_json(SEGMENTS, "fa"))["canonical"])


class WriteTest(unittest.TestCase):
    def test_both_forms_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            written = write_canonical(Path(directory), "meeting", SEGMENTS, "fa")
            self.assertEqual(
                sorted(path.name for path in written),
                ["meeting.raw.json", "meeting.raw.txt"],
            )
            self.assertTrue(all(path.is_file() for path in written))

    def test_the_output_directory_is_created_if_it_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "deep" / "out"
            write_canonical(target, "meeting", SEGMENTS, "fa")
            self.assertTrue((target / "meeting.raw.txt").is_file())

    def test_the_names_cannot_collide_with_a_formatted_export(self):
        # A raw copy overwritten by the readable transcript would defeat its purpose.
        with tempfile.TemporaryDirectory() as directory:
            written = write_canonical(Path(directory), "meeting", SEGMENTS, "fa")
            self.assertTrue(all(".raw." in path.name for path in written))

    def test_an_empty_transcript_still_produces_both_files(self):
        with tempfile.TemporaryDirectory() as directory:
            written = write_canonical(Path(directory), "meeting", [], "fa")
            self.assertTrue(all(path.is_file() for path in written))
            self.assertEqual(written[0].read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
