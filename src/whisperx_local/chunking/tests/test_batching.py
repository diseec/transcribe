"""Grouping chunks for one model load, and splitting the result back apart.

Both halves are pure, which is the point: the part that decides what audio the recogniser
sees, and the part that decides which chunk each recognised word belongs to, can be wrong
in ways that produce a plausible transcript, so neither is left to an end-to-end run to
check.
"""

import unittest

from whisperx_local.chunking import Chunk
from whisperx_local.chunking.batching import (
    CONTIGUOUS_TOLERANCE,
    DEFAULT_CHUNKS_PER_CALL,
    chunk_spans,
    distribute_segments,
    group_chunks,
    is_contiguous,
)


def chunk(index: int, start: float, length: float = 600.0) -> Chunk:
    return Chunk(index=index, start=start, length=length)


def contiguous(count: int, length: float = 600.0) -> list[Chunk]:
    return [chunk(index, index * length, length) for index in range(count)]


class ContiguityTest(unittest.TestCase):
    def test_touching_chunks_can_be_joined(self):
        self.assertTrue(is_contiguous(chunk(0, 0.0), chunk(1, 600.0)))

    def test_rounding_is_tolerated(self):
        self.assertTrue(is_contiguous(chunk(0, 0.0, 600.0), chunk(1, 600.001)))

    def test_an_overlap_is_not_contiguous(self):
        # Joining these would duplicate two seconds of speech.
        self.assertFalse(is_contiguous(chunk(0, 0.0), chunk(1, 598.0)))

    def test_a_gap_is_not_contiguous(self):
        self.assertFalse(is_contiguous(chunk(0, 0.0), chunk(1, 610.0)))

    def test_the_tolerance_is_far_smaller_than_any_real_overlap(self):
        self.assertLess(CONTIGUOUS_TOLERANCE, 0.1)


class GroupingTest(unittest.TestCase):
    def test_contiguous_chunks_are_grouped_up_to_the_limit(self):
        groups = group_chunks(contiguous(6), per_call=3)
        self.assertEqual([len(group) for group in groups], [3, 3])

    def test_a_remainder_becomes_its_own_smaller_group(self):
        groups = group_chunks(contiguous(7), per_call=3)
        self.assertEqual([len(group) for group in groups], [3, 3, 1])

    def test_one_per_call_is_the_old_behaviour(self):
        groups = group_chunks(contiguous(4), per_call=1)
        self.assertEqual([len(group) for group in groups], [1, 1, 1, 1])

    def test_a_gap_stops_a_group(self):
        # The second chunk overlaps the first, so those two may not be joined; the third
        # is contiguous with the second and may be.
        chunks = [chunk(0, 0.0), chunk(1, 598.0), chunk(2, 1198.0)]
        groups = group_chunks(chunks, per_call=3)
        self.assertEqual([[entry.index for entry in group] for group in groups], [[0], [1, 2]])

    def test_no_chunks_gives_no_groups(self):
        self.assertEqual(group_chunks([]), [])

    def test_finished_chunks_are_left_out_entirely(self):
        groups = group_chunks(contiguous(6), per_call=3, pending={0, 1, 4})
        self.assertEqual([[entry.index for entry in group] for group in groups], [[0, 1], [4]])

    def test_a_finished_chunk_is_a_gap_that_nothing_is_joined_across(self):
        # Joining chunk 0's audio to chunk 2's would make the recogniser read speech whose
        # result is then discarded.
        groups = group_chunks(contiguous(4), per_call=4, pending={2, 3})
        self.assertEqual([[entry.index for entry in group] for group in groups], [[2, 3]])

    def test_nothing_pending_gives_no_groups(self):
        self.assertEqual(group_chunks(contiguous(4), pending=set()), [])

    def test_every_pending_chunk_appears_exactly_once(self):
        chunks = [chunk(index, index * 600.0) for index in range(5)]
        groups = group_chunks(chunks, per_call=2)
        seen = [entry.index for group in groups for entry in group]
        self.assertEqual(sorted(seen), [0, 1, 2, 3, 4])

    def test_the_default_is_off_because_it_measured_neutral(self):
        # Kept honest by a test: if someone turns this on they should have a measurement.
        self.assertEqual(DEFAULT_CHUNKS_PER_CALL, 1)

    def test_a_nonsense_limit_is_treated_as_one(self):
        self.assertEqual([len(group) for group in group_chunks(contiguous(3), per_call=0)], [1, 1, 1])


class SpanTest(unittest.TestCase):
    def test_spans_start_at_zero_and_accumulate(self):
        spans = chunk_spans(contiguous(3))
        self.assertEqual([(low, high) for _, low, high in spans], [(0.0, 600.0), (600.0, 1200.0), (1200.0, 1800.0)])

    def test_uneven_chunks_still_tile_the_joined_audio(self):
        spans = chunk_spans([chunk(0, 0.0, 100.0), chunk(1, 100.0, 250.0)])
        self.assertEqual([(low, high) for _, low, high in spans], [(0.0, 100.0), (100.0, 350.0)])

    def test_no_chunks_gives_no_spans(self):
        self.assertEqual(chunk_spans([]), [])


class DistributionTest(unittest.TestCase):
    """Which chunk each recognised word belongs to, and on which clock."""

    def group(self) -> list[Chunk]:
        return [chunk(0, 0.0, 600.0), chunk(1, 600.0, 600.0), chunk(2, 1200.0, 600.0)]

    def test_the_first_chunk_keeps_its_own_timings(self):
        placed = distribute_segments(self.group(), [{"start": 10.0, "end": 20.0, "text": "a"}])
        self.assertAlmostEqual(placed[0][0]["start"], 10.0)

    def test_a_later_chunk_gets_timings_relative_to_itself(self):
        # Chunk 1 begins 600 s into the joined file, so its text starts near zero.
        placed = distribute_segments(self.group(), [{"start": 610.0, "end": 620.0, "text": "b"}])
        self.assertEqual(placed[1][0]["text"], "b")
        self.assertAlmostEqual(placed[1][0]["start"], 10.0)

    def test_nothing_is_lost(self):
        segments = [{"start": float(start), "end": float(start + 5), "text": str(start)} for start in range(0, 1800, 100)]
        placed = distribute_segments(self.group(), segments)
        self.assertEqual(sum(len(entries) for entries in placed.values()), len(segments))

    def test_every_chunk_gets_a_bucket_even_when_empty(self):
        placed = distribute_segments(self.group(), [])
        self.assertEqual(sorted(placed), [0, 1, 2])

    def test_a_segment_past_the_end_goes_to_the_last_chunk(self):
        placed = distribute_segments(self.group(), [{"start": 1800.5, "end": 1801.0, "text": "x"}])
        self.assertEqual(len(placed[2]), 1)

    def test_placement_follows_the_middle_not_the_start(self):
        # 598-1210 is mostly in chunk 1 even though it begins inside chunk 0.
        placed = distribute_segments(self.group(), [{"start": 598.0, "end": 1210.0, "text": "long"}])
        self.assertEqual(len(placed[1]), 1)
        self.assertEqual(len(placed[0]), 0)

    def test_word_timings_are_moved_with_their_segment(self):
        segment = {
            "start": 610.0,
            "end": 615.0,
            "text": "b",
            "words": [{"word": "b", "start": 610.0, "end": 615.0}],
        }
        placed = distribute_segments(self.group(), [segment])
        self.assertAlmostEqual(placed[1][0]["words"][0]["start"], 10.0)

    def test_a_word_without_timings_is_left_alone(self):
        segment = {"start": 610.0, "end": 615.0, "text": "b", "words": [{"word": "b"}]}
        placed = distribute_segments(self.group(), [segment])
        self.assertNotIn("start", placed[1][0]["words"][0])

    def test_the_input_segments_are_not_modified(self):
        segment = {"start": 610.0, "end": 615.0, "text": "b"}
        distribute_segments(self.group(), [segment])
        self.assertEqual(segment["start"], 610.0)

    def test_a_group_of_one_returns_timings_unchanged(self):
        placed = distribute_segments([chunk(7, 4200.0, 600.0)], [{"start": 3.0, "end": 4.0, "text": "z"}])
        # A single-chunk group is never joined, so its segments are already local.
        self.assertAlmostEqual(placed[7][0]["start"], 3.0)

    def test_no_chunks_places_nothing(self):
        self.assertEqual(distribute_segments([], [{"start": 0.0, "end": 1.0, "text": "x"}]), {})


if __name__ == "__main__":
    unittest.main()
