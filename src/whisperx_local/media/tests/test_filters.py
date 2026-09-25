"""The filter chains, which encode measurement results rather than preferences."""

import unittest

from whisperx_local.media import filters


class FilterTest(unittest.TestCase):
    def test_recognition_and_speaker_tracks_differ(self):
        # Dynamic range processing helps recognition but blurs speaker identity, so
        # the two tracks must not be the same chain.
        self.assertNotEqual(filters.RECOGNITION_FILTER, filters.SPEAKER_FILTER)

    def test_speaker_track_avoids_dynamic_normalisation(self):
        self.assertNotIn("dynaudnorm", filters.SPEAKER_FILTER)

    def test_recognition_track_recovers_quiet_speech(self):
        self.assertIn("speechnorm", filters.RECOGNITION_FILTER)
        self.assertIn("loudnorm", filters.RECOGNITION_FILTER)

    def test_the_highpass_removes_rumble_the_model_cannot_use(self):
        self.assertIn("highpass", filters.RECOGNITION_FILTER)
        self.assertIn("highpass", filters.SPEAKER_FILTER)

    def test_the_energy_chain_measures_one_window_at_a_time(self):
        chain = filters.energy_filter(0.25)
        # 0.25s at 16 kHz is 4000 samples, which is what makes each reading a window
        # rather than a running total.
        self.assertIn("asetnsamples=n=4000", chain)
        self.assertIn("reset=1", chain)

    def test_a_tiny_window_still_asks_for_one_sample(self):
        self.assertIn("asetnsamples=n=1", filters.energy_filter(0.00001))


if __name__ == "__main__":
    unittest.main()
