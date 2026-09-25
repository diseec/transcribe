"""Chunk planning: the plan shapes, and when each is chosen.

Wrong decisions here are expensive but silent, so the cases that bit in practice are
all pinned: a lone early measurement producing a stunted first chunk, an uninformative
loudness scan being trusted anyway, and a target at the very end being dropped.
"""

import unittest

from whisperx_local.chunking.plan import (
    Chunk,
    loudness_spread,
    plan_chunks,
    quiet_cut_points,
)


class PlanChunksTest(unittest.TestCase):
    def test_short_file_is_one_chunk(self):
        self.assertEqual(
            plan_chunks(30.0, chunk_seconds=600, overlap_seconds=2),
            [Chunk(0, 0.0, 30.0)],
        )

    def test_fixed_mode_is_unchanged(self):
        chunks = plan_chunks(100.0, chunk_seconds=30, overlap_seconds=2)
        self.assertEqual([chunk.start for chunk in chunks], [0.0, 28.0, 56.0, 84.0])
        self.assertAlmostEqual(chunks[-1].end, 100.0)

    def test_sliver_tail_is_folded(self):
        chunks = plan_chunks(57.9, chunk_seconds=30, overlap_seconds=2)
        self.assertEqual(len(chunks), 2)
        self.assertAlmostEqual(chunks[-1].end, 57.9)

    def test_boundaries_snap_into_silence(self):
        chunks = plan_chunks(
            100.0, chunk_seconds=30, overlap_seconds=2, cut_points=[29.4, 58.7, 88.1]
        )
        # 88.1 would leave an 11.9s tail, below the minimum, so it is folded in.
        self.assertEqual([chunk.start for chunk in chunks], [0.0, 29.4, 58.7])
        self.assertAlmostEqual(chunks[-1].end, 100.0)

    def test_unusable_cut_points_fall_back_to_fixed(self):
        chunks = plan_chunks(100.0, chunk_seconds=30, overlap_seconds=2, cut_points=[0.5])
        self.assertEqual([chunk.start for chunk in chunks], [0.0, 28.0, 56.0, 84.0])

    def test_no_cut_points_uses_fixed(self):
        self.assertEqual(
            [chunk.start for chunk in plan_chunks(100.0, chunk_seconds=30, overlap_seconds=2)],
            [0.0, 28.0, 56.0, 84.0],
        )

    def test_snapped_mode_rejects_cuts_that_are_too_close(self):
        chunks = plan_chunks(
            120.0,
            chunk_seconds=40,
            overlap_seconds=0,
            cut_points=[5.0, 41.0, 80.0],
            min_chunk_seconds=20.0,
        )
        # 5.0 is far too early to be a boundary for the first 40s chunk.
        self.assertEqual(chunks[0].start, 0.0)
        self.assertGreaterEqual(chunks[1].start, 20.0)

    def test_aligned_chunks_are_contiguous_so_no_overlap_is_needed(self):
        chunks = plan_chunks(
            100.0, chunk_seconds=30, overlap_seconds=5, cut_points=[30.0, 60.0]
        )
        for earlier, later in zip(chunks, chunks[1:]):
            self.assertAlmostEqual(earlier.end, later.start)

    def test_overlapping_chunks_are_not_contiguous(self):
        chunks = plan_chunks(100.0, chunk_seconds=30, overlap_seconds=5)
        self.assertLess(chunks[1].start, chunks[0].end)

    def test_zero_chunk_seconds_rejected(self):
        with self.assertRaises(ValueError):
            plan_chunks(10.0, chunk_seconds=0, overlap_seconds=0)

    def test_the_overlap_cannot_exceed_half_a_chunk(self):
        # Otherwise the step becomes zero or negative and the loop never advances.
        chunks = plan_chunks(100.0, chunk_seconds=20, overlap_seconds=60)
        self.assertGreater(len(chunks), 1)
        self.assertLess(chunks[1].start, 20.0)


class CandidateRequirementTest(unittest.TestCase):
    def test_a_snapped_plan_needs_a_real_candidate(self):
        # 42s is outside the tolerance of every target, so snapping would be
        # arbitrary. Keeping the overlapping intervals protects a straddling word.
        chunks = plan_chunks(100.0, chunk_seconds=30, overlap_seconds=2, cut_points=[42.0])
        self.assertEqual([chunk.start for chunk in chunks], [0.0, 28.0, 56.0, 84.0])

    def test_one_usable_candidate_is_enough_to_snap(self):
        chunks = plan_chunks(100.0, chunk_seconds=30, overlap_seconds=2, cut_points=[30.0])
        self.assertEqual(chunks[1].start, 30.0)


class LoudnessSpreadTest(unittest.TestCase):
    def test_flat_audio_has_a_narrow_spread(self):
        samples = [(index * 0.25, -13.0 + (index % 3) * 0.2) for index in range(40)]
        self.assertLess(loudness_spread(samples), 1.0)

    def test_dynamic_audio_has_a_wide_spread(self):
        samples = [(0.0, -60.0), (0.25, -60.0), (0.5, -10.0), (0.75, -10.0)]
        self.assertGreater(loudness_spread(samples), 40.0)

    def test_a_single_outlier_does_not_look_like_dynamics(self):
        samples = [(index * 0.25, -13.0) for index in range(40)]
        samples[20] = (5.0, -90.0)
        self.assertLess(loudness_spread(samples), 3.0)

    def test_too_few_samples_is_unknown(self):
        self.assertIsNone(loudness_spread([(0.0, -10.0)]))


class QuietCutPointsTest(unittest.TestCase):
    @staticmethod
    def tone(duration=100.0, level=-5.0, deep=-60.0, dips=(), step=0.25):
        """A steady tone with loudness dips covering whole seconds."""
        samples = []
        time = 0.0
        while time < duration - 1e-9:
            samples.append((round(time, 3), deep if int(time) in dips else level))
            time += step
        return samples

    def test_cuts_land_on_the_calmest_windows(self):
        samples = self.tone(dips=(28, 29, 31, 32, 58, 59, 61, 62))
        cuts = quiet_cut_points(100.0, samples, chunk_seconds=30.0)
        levels = dict(samples)
        self.assertEqual(len(cuts), 2)
        for cut in cuts:
            self.assertLessEqual(levels[cut], -59.0)
        self.assertAlmostEqual(cuts[0], 29.75, delta=1.0)
        self.assertAlmostEqual(cuts[1], 59.75, delta=1.0)

    def test_flat_audio_is_refused(self):
        # Limited audio is flat to within a dB, so a "calmest" window there is noise.
        samples = [(index * 0.25, -13.0 + (index % 3) * 0.2) for index in range(400)]
        self.assertEqual(quiet_cut_points(100.0, samples, chunk_seconds=30.0), [])

    def test_no_samples_is_refused(self):
        self.assertEqual(quiet_cut_points(100.0, [], chunk_seconds=30.0), [])

    def test_a_single_early_sample_cannot_stunt_the_first_chunk(self):
        self.assertEqual(quiet_cut_points(100.0, [(1.0, -10.0)], chunk_seconds=30.0), [])

    def test_boundaries_are_kept_apart(self):
        cuts = quiet_cut_points(
            100.0, self.tone(dips=(25, 26, 45, 46, 65, 66, 85, 86)), chunk_seconds=30.0
        )
        self.assertTrue(cuts)
        for earlier, later in zip(cuts, cuts[1:]):
            self.assertGreaterEqual(later - earlier, 20.0)

    def test_the_tail_is_never_orphaned(self):
        cuts = quiet_cut_points(
            100.0, self.tone(dips=(25, 26, 45, 46, 65, 66, 85, 86)), chunk_seconds=30.0
        )
        self.assertTrue(cuts)
        for cut in cuts:
            self.assertGreaterEqual(100.0 - cut, 20.0)

    def test_a_target_at_the_minimum_tail_is_still_used(self):
        cuts = quiet_cut_points(
            60.0, self.tone(duration=60.0, dips=(20, 21, 40, 41)), chunk_seconds=20.0
        )
        self.assertEqual(cuts, [20.0, 40.0])

    def test_a_non_positive_duration_is_refused(self):
        samples = self.tone(dips=(20, 21, 40, 41))
        self.assertEqual(quiet_cut_points(0.0, samples, chunk_seconds=20.0), [])


if __name__ == "__main__":
    unittest.main()
