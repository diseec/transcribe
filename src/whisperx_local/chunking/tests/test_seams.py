"""Rejoining chunks: ownership must tile, and no word may be lost or duplicated.

The ownership tests are the important ones. Subtracting a fixed overlap from
contiguous chunks left a window that no chunk owned, and every word inside it was
silently dropped -- a bug that produced a transcript missing words at each boundary
with nothing in the log to indicate it.
"""

import unittest

from whisperx_local.chunking.plan import plan_chunks
from whisperx_local.chunking.seams import merge_segments, owned_ranges, shift_segments


class OwnedRangesTest(unittest.TestCase):
    @staticmethod
    def ordered(chunks, duration):
        ranges = owned_ranges(chunks, duration)
        return [ranges[chunk.index] for chunk in chunks]

    def assert_tiles(self, chunks, duration):
        ordered = self.ordered(chunks, duration)
        self.assertEqual(ordered[0][0], 0.0)
        self.assertEqual(ordered[-1][1], duration)
        for (_, high), (low, _) in zip(ordered, ordered[1:]):
            self.assertAlmostEqual(high, low)

    def test_ranges_tile_exactly_without_overlap(self):
        self.assert_tiles(plan_chunks(100.0, chunk_seconds=30, overlap_seconds=0), 100.0)

    def test_ranges_tile_exactly_with_overlapping_chunks(self):
        self.assert_tiles(plan_chunks(100.0, chunk_seconds=30, overlap_seconds=2), 100.0)

    def test_ranges_tile_exactly_for_snapped_chunks(self):
        self.assert_tiles(
            plan_chunks(100.0, chunk_seconds=30, overlap_seconds=2, cut_points=[30.0, 60.0]),
            100.0,
        )

    def test_each_range_lies_inside_its_chunk_audio(self):
        # A chunk can only produce words from audio it actually decoded.
        duration = 100.0
        chunks = plan_chunks(duration, chunk_seconds=30, overlap_seconds=2)
        ranges = owned_ranges(chunks, duration)
        for chunk in chunks:
            low, high = ranges[chunk.index]
            self.assertGreaterEqual(low, chunk.start - 1e-6)
            self.assertLessEqual(high, chunk.end + 1e-6)

    def test_no_duration_means_no_ranges(self):
        self.assertEqual(owned_ranges([], 0.0), {})


class MergeSegmentsTest(unittest.TestCase):
    def test_word_in_overlap_appears_once(self):
        chunks = plan_chunks(60.0, chunk_seconds=30, overlap_seconds=2)
        payloads = {
            0: [
                {"start": 0.0, "end": 5.0, "text": "a"},
                {"start": 26.0, "end": 31.0, "text": "b"},
            ],
            1: [
                {"start": 0.0, "end": 3.0, "text": "b"},
                {"start": 20.0, "end": 25.0, "text": "c"},
            ],
        }
        merged = merge_segments(chunks, payloads, duration=60.0)
        self.assertEqual([segment["text"] for segment in merged], ["a", "b", "c"])
        self.assertAlmostEqual(merged[1]["start"], 28.0)

    def test_word_after_a_silence_boundary_is_not_dropped(self):
        duration = 100.0
        chunks = plan_chunks(
            duration, chunk_seconds=30, overlap_seconds=2.0, cut_points=[30.0, 60.0, 90.0]
        )
        payloads = {
            0: [{"start": 10.0, "end": 11.0, "text": "before"}],
            1: [{"start": 28.5, "end": 29.5, "text": "straddles"}],
        }
        merged = merge_segments(chunks, payloads, duration=duration)
        self.assertEqual([segment["text"] for segment in merged], ["before", "straddles"])

    def test_missing_chunk_leaves_a_gap(self):
        chunks = plan_chunks(60.0, chunk_seconds=30, overlap_seconds=2)
        payloads = {0: [{"start": 0.0, "end": 5.0, "text": "a"}]}
        merged = merge_segments(chunks, payloads, duration=60.0)
        self.assertEqual([segment["text"] for segment in merged], ["a"])

    def test_results_are_sorted(self):
        chunks = plan_chunks(60.0, chunk_seconds=30, overlap_seconds=2)
        payloads = {
            1: [{"start": 0.0, "end": 2.0, "text": "second"}],
            0: [{"start": 1.0, "end": 3.0, "text": "first"}],
        }
        merged = merge_segments(chunks, payloads, duration=60.0)
        self.assertEqual(
            [segment["start"] for segment in merged],
            sorted(segment["start"] for segment in merged),
        )

    def test_a_segment_at_the_very_end_is_kept(self):
        chunks = plan_chunks(60.0, chunk_seconds=30, overlap_seconds=0)
        payloads = {1: [{"start": 29.0, "end": 30.0, "text": "last"}]}
        merged = merge_segments(chunks, payloads, duration=60.0)
        self.assertEqual([segment["text"] for segment in merged], ["last"])

    def test_no_chunks_merges_nothing(self):
        self.assertEqual(merge_segments([], {}, duration=10.0), [])


class ShiftSegmentsTest(unittest.TestCase):
    def test_segments_and_words_are_offset(self):
        shifted = shift_segments(
            [
                {
                    "start": 1.0,
                    "end": 2.0,
                    "text": "x",
                    "words": [{"word": "x", "start": 1.0, "end": 2.0}],
                }
            ],
            28.0,
        )
        self.assertEqual(shifted[0]["start"], 29.0)
        self.assertEqual(shifted[0]["end"], 30.0)
        self.assertEqual(shifted[0]["words"][0]["start"], 29.0)
        self.assertEqual(shifted[0]["words"][0]["end"], 30.0)
        self.assertEqual(shifted[0]["text"], "x")

    def test_original_is_not_mutated(self):
        original = [{"start": 1.0, "end": 2.0, "text": "x"}]
        shift_segments(original, 10.0)
        self.assertEqual(original[0]["start"], 1.0)

    def test_words_without_timings_are_passed_through(self):
        shifted = shift_segments([{"start": 0.0, "end": 1.0, "words": [{"word": "x"}]}], 5.0)
        self.assertEqual(shifted[0]["words"], [{"word": "x"}])

    def test_a_segment_without_word_list_is_left_alone(self):
        shifted = shift_segments([{"start": 0.0, "end": 1.0}], 5.0)
        self.assertNotIn("words", shifted[0])


if __name__ == "__main__":
    unittest.main()
