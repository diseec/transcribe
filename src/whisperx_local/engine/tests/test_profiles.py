"""Profiles: option filling, and the memory tiers that prevent native aborts.

The memory rules are the ones that matter. An abort two hours into a run is caused by
peak memory, so the tuning deliberately overrides an explicit batch size rather than
respecting it -- a decision that needs a test to stop it looking like a bug.
"""

import argparse
import unittest

from whisperx_local.engine import profiles
from whisperx_local.paths import GIB

# A complete stand-in for parsed options, so tests in this package can build a
# command or apply a profile without running the real argument parser. Options a
# profile may fill start as None, which is how "not chosen by the user" is expressed.
BASE_OPTIONS: dict[str, object] = {
    "profile": "fast",
    "language": "fa",
    "model": None,
    "compute_type": None,
    "beam_size": None,
    "best_of": None,
    "patience": None,
    "batch_size": None,
    "vad_method": None,
    "threads": None,
    "vad_onset": 0.10,
    "vad_offset": 0.05,
    "chunk_size": 8,
    "compression_ratio_threshold": 3.0,
    "logprob_threshold": -2.0,
    "no_speech_threshold": 0.95,
    "hotwords": None,
    "prompt": None,
    "normalize": True,
    "chunk_seconds": 600.0,
    "chunk_overlap": 2.0,
    "silence_split": True,
    "silence_min": 0.5,
    "speakers": None,
    "min_turn": 0.35,
    "retries": 3,
    "align_retries": 3,
    "diarize_retries": 2,
    "diarize": None,
    "diarize_audio": "gentle",
    "network": "auto",
    "output_format": "txt",
}


def options(**overrides) -> argparse.Namespace:
    """Parsed-options stand-in with every field the engine reads."""
    values = dict(BASE_OPTIONS)
    values.update(overrides)
    return argparse.Namespace(**values)


class ResolveTest(unittest.TestCase):
    def test_a_known_profile_resolves(self):
        self.assertEqual(profiles.resolve("fast").name, "fast")

    def test_an_unknown_profile_explains_the_options(self):
        with self.assertRaises(SystemExit) as caught:
            profiles.resolve("turbo")
        message = str(caught.exception)
        self.assertIn("turbo", message)
        for name in profiles.PROFILES:
            self.assertIn(name, message)

    def test_every_profile_describes_itself(self):
        for name in profiles.PROFILES:
            with self.subTest(profile=name):
                self.assertTrue(profiles.describe(name))

    def test_the_default_profile_exists(self):
        self.assertIn(profiles.DEFAULT_PROFILE, profiles.PROFILES)


class ApplyTest(unittest.TestCase):
    def test_unset_options_are_filled_from_the_profile(self):
        tuned = options()
        profile, _ = profiles.apply(tuned, available=8 * GIB)
        self.assertEqual(profile.name, "fast")
        self.assertEqual(tuned.model, "large-v3")
        self.assertEqual(tuned.compute_type, "int8")
        self.assertEqual(tuned.beam_size, 1)
        self.assertEqual(tuned.batch_size, 16)
        self.assertGreaterEqual(tuned.threads, 1)

    def test_an_explicit_option_wins_over_the_profile(self):
        tuned = options(model="medium", beam_size=7)
        profiles.apply(tuned, available=8 * GIB)
        self.assertEqual(tuned.model, "medium")
        self.assertEqual(tuned.beam_size, 7)

    def test_batching_is_never_left_at_one(self):
        # batch_size 1 selects WhisperX's slow sequential path, which is several
        # times slower for identical output.
        for name in profiles.PROFILES:
            with self.subTest(profile=name):
                self.assertGreater(profiles.PROFILES[name].batch_size, 1)

    def test_plenty_of_memory_changes_nothing(self):
        tuned = options(batch_size=None)
        _, notes = profiles.apply(tuned, available=8 * GIB)
        self.assertEqual(notes, [])
        self.assertEqual(tuned.batch_size, 16)

    def test_tight_memory_reduces_the_batch_and_explains_why(self):
        tuned = options()
        _, notes = profiles.apply(tuned, available=0.5 * GIB)
        self.assertEqual(tuned.batch_size, 1)
        self.assertTrue(any("batch size" in note for note in notes))

    def test_tight_memory_caps_threads_and_shortens_chunks(self):
        tuned = options()
        profiles.apply(tuned, available=0.5 * GIB)
        self.assertEqual(tuned.threads, 2)
        self.assertEqual(tuned.chunk_seconds, 240.0)

    def test_moderate_memory_uses_the_middle_tier(self):
        tuned = options()
        profiles.apply(tuned, available=1.0 * GIB)
        self.assertEqual(tuned.batch_size, 2)
        self.assertEqual(tuned.chunk_seconds, 300.0)

    def test_unknown_memory_skips_tuning_rather_than_assuming_none(self):
        # Treating "cannot measure" as "no memory" would silently cripple the run.
        tuned = options()
        _, notes = profiles.apply(tuned, available=None)
        self.assertEqual(notes, [])
        self.assertEqual(tuned.batch_size, 16)


if __name__ == "__main__":
    unittest.main()
