"""Learned timing: rates, averaging, and tolerance of a damaged store.

The point of this store is that a prediction survives the process, so the second
run of the same profile is predictable where the first was guesswork.
"""

import json
import tempfile
import unittest
from pathlib import Path

from whisperx_local.progress.timing import TimingStore, context_key


class TimingStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.path = Path(self._temporary.name) / "timings.json"

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def store(self, **kwargs) -> TimingStore:
        return TimingStore(path=self.path, **kwargs)

    def test_unknown_context_predicts_nothing(self):
        self.assertEqual(self.store().predict("unseen", 600.0), {})

    def test_a_recording_is_learned_as_a_rate_per_second_of_audio(self):
        store = self.store()
        store.record("c", audio_seconds=100.0, measured={"transcribe": 50.0})
        self.assertAlmostEqual(store.predict("c", 100.0)["transcribe"], 50.0)
        self.assertAlmostEqual(store.predict("c", 200.0)["transcribe"], 100.0)

    def test_repeated_measurements_move_by_the_average(self):
        store = self.store(alpha=0.5)
        store.record("c", audio_seconds=100.0, measured={"transcribe": 100.0})
        store.record("c", audio_seconds=100.0, measured={"transcribe": 50.0})
        self.assertAlmostEqual(store.predict("c", 100.0)["transcribe"], 75.0)

    def test_an_alpha_of_one_uses_only_the_latest_measurement(self):
        store = self.store(alpha=1.0)
        store.record("c", audio_seconds=100.0, measured={"transcribe": 100.0})
        store.record("c", audio_seconds=100.0, measured={"transcribe": 50.0})
        self.assertAlmostEqual(store.predict("c", 100.0)["transcribe"], 50.0)

    def test_very_short_audio_is_ignored(self):
        store = self.store()
        store.record("c", audio_seconds=0.2, measured={"transcribe": 5.0})
        self.assertEqual(store.predict("c", 100.0), {})

    def test_zero_durations_are_ignored(self):
        store = self.store()
        store.record("c", audio_seconds=100.0, measured={"transcribe": 0.0})
        self.assertEqual(store.predict("c", 100.0), {})

    def test_contexts_do_not_contaminate_each_other(self):
        store = self.store()
        store.record("fast", audio_seconds=100.0, measured={"transcribe": 50.0})
        self.assertEqual(store.predict("accurate", 100.0), {})

    def test_values_survive_a_new_store(self):
        self.store().record("c", audio_seconds=100.0, measured={"transcribe": 50.0})
        self.assertAlmostEqual(
            TimingStore(path=self.path).predict("c", 100.0)["transcribe"], 50.0
        )

    def test_a_damaged_file_is_tolerated(self):
        self.path.write_text("{ this is not json", encoding="utf-8")
        self.assertEqual(self.store().predict("c", 100.0), {})

    def test_a_file_from_an_older_schema_is_ignored(self):
        self.path.write_text(
            json.dumps({"schema": "ancient", "contexts": {"c": {"transcribe": 1.0}}}),
            encoding="utf-8",
        )
        self.assertEqual(self.store().predict("c", 100.0), {})

    def test_bogus_rates_are_dropped(self):
        self.path.write_text(
            json.dumps(
                {
                    "schema": "timing-v1",
                    "contexts": {"c": {"bad": -1.0, "text": "x", "good": 2.0}},
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(self.store().predict("c", 10.0), {"good": 20.0})

    def test_non_positive_audio_length_predicts_nothing(self):
        store = self.store()
        store.record("c", audio_seconds=100.0, measured={"transcribe": 50.0})
        self.assertEqual(store.predict("c", 0.0), {})

    def test_forget_removes_everything(self):
        store = self.store()
        store.record("c", audio_seconds=100.0, measured={"transcribe": 50.0})
        store.forget()
        self.assertEqual(store.predict("c", 100.0), {})

    def test_forgetting_a_missing_file_is_harmless(self):
        self.store().forget()

    def test_context_key_separates_the_settings_that_change_speed(self):
        base = dict(model="large-v3", language="fa", device="cpu")
        self.assertNotEqual(
            context_key(profile="fast", **base), context_key(profile="accurate", **base)
        )
        self.assertNotEqual(
            context_key(profile="fast", **base),
            context_key(profile="fast", model="small", language="fa", device="cpu"),
        )
        self.assertEqual(
            context_key(profile="fast", **base), context_key(profile="fast", **base)
        )


if __name__ == "__main__":
    unittest.main()
