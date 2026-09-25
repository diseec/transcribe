"""Cached chunk invalidation, for both reasons a cache can go stale.

A cache that only notices the wrong kind of change is worse than no cache, because the
result is confidently wrong and looks fast. There are two independent reasons to throw
work away here:

* the *slicing* moved, so the audio the text was read from is not the audio we now have;
* the *recogniser* was configured differently, so the text is stale even though the audio
  is identical.

The second was not implemented. ``recognition_signature`` says in its own documentation
that it is used as a cache key, and was only wired into the completion marker -- so
switching model, precision, beam, language, prompt or hotwords silently reused the old
text. An A/B between two engines would have compared a transcript with itself.
"""

import json
import tempfile
import unittest
from pathlib import Path

from whisperx_local.chunking import discard_recognition, plan_chunks, plan_signature
from whisperx_local.cli.options import build_parser
from whisperx_local.engine.commands import SIGNATURE_FIELDS, recognition_signature
from whisperx_local.services.artifacts import WorkSpace


class DiscardRecognitionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.chunks = Path(temporary.name) / "chunks"
        self.chunks.mkdir()
        for name in (
            "part-0000.wav",
            "part-0000.asr.json",
            "part-0000.aligned.json",
            "part-0001.wav",
            "part-0001.asr.json",
            "plan.json",
        ):
            (self.chunks / name).write_text("{}", encoding="utf-8")

    def remaining(self) -> set[str]:
        return {path.name for path in self.chunks.iterdir()}

    def test_recognised_and_aligned_text_is_removed(self):
        discard_recognition(self.chunks)
        self.assertEqual(
            self.remaining(),
            {"part-0000.wav", "part-0001.wav", "plan.json"},
        )

    def test_the_sliced_audio_is_kept(self):
        # Re-slicing would be wasted work: the audio does not depend on the recogniser.
        discard_recognition(self.chunks)
        self.assertTrue((self.chunks / "part-0000.wav").is_file())

    def test_it_reports_how_much_was_thrown_away(self):
        self.assertEqual(discard_recognition(self.chunks), 3)

    def test_a_missing_directory_is_not_an_error(self):
        self.assertEqual(discard_recognition(self.chunks / "absent"), 0)

    def test_nothing_to_discard_reports_nothing(self):
        discard_recognition(self.chunks)
        self.assertEqual(discard_recognition(self.chunks), 0)


class RecognitionSignatureTest(unittest.TestCase):
    def signature(self, *arguments):
        options = build_parser().parse_args(["run", "a.wav", *arguments])
        return recognition_signature(options)

    def test_it_covers_the_things_that_change_the_text(self):
        for name in ("model", "language", "compute_type", "beam_size", "hotwords", "prompt"):
            with self.subTest(field=name):
                self.assertIn(name, SIGNATURE_FIELDS)

    def test_precision_is_part_of_it(self):
        self.assertNotEqual(
            self.signature("--compute-type", "int8"),
            self.signature("--compute-type", "float32"),
        )

    def test_a_different_model_is_part_of_it(self):
        self.assertNotEqual(
            self.signature("--model", "large-v3"),
            self.signature("--model", "large-v3-turbo"),
        )

    def test_beam_width_is_part_of_it(self):
        self.assertNotEqual(
            self.signature("--beam-size", "1"), self.signature("--beam-size", "10")
        )

    def test_vocabulary_biasing_is_part_of_it(self):
        self.assertNotEqual(
            self.signature("--hotwords", "diarize"),
            self.signature("--hotwords", "transcribe"),
        )

    def test_an_identical_configuration_gives_an_identical_signature(self):
        self.assertEqual(self.signature("--beam-size", "5"), self.signature("--beam-size", "5"))

    def test_slicing_alone_does_not_move_the_signature(self):
        # Chunk length is a slicing concern; it must not look like a recognition change.
        self.assertEqual(
            self.signature("--chunk-seconds", "600"),
            self.signature("--chunk-seconds", "300"),
        )


class PlanDocumentTest(unittest.TestCase):
    """The stored plan has to carry both fingerprints, in a readable shape."""

    def test_slicing_can_be_read_apart_from_recognition(self):
        from whisperx_local.services.runner import _slicing_of

        stored = {"source": "a.wav", "cuts": "abc", "recognition": {"model": "large-v3"}}
        self.assertEqual(_slicing_of(stored), {"source": "a.wav", "cuts": "abc"})

    def test_slicing_ignores_a_recognition_change(self):
        from whisperx_local.services.runner import _slicing_of

        before = {"cuts": "abc", "recognition": {"compute_type": "int8"}}
        after = {"cuts": "abc", "recognition": {"compute_type": "float32"}}
        self.assertEqual(_slicing_of(before), _slicing_of(after))

    def test_slicing_notices_a_moved_boundary(self):
        from whisperx_local.services.runner import _slicing_of

        before = plan_signature(
            source=Path(__file__), duration=10.0, chunk_seconds=5.0,
            overlap_seconds=0.0, cut_points=[5.0],
        )
        after = plan_signature(
            source=Path(__file__), duration=10.0, chunk_seconds=5.0,
            overlap_seconds=0.0, cut_points=[4.0],
        )
        self.assertNotEqual(_slicing_of(before), _slicing_of(after))
        self.assertEqual(plan_chunks(10.0, chunk_seconds=5.0, overlap_seconds=0.0)[0].start, 0.0)


class WorkspaceRevalidationTest(unittest.TestCase):
    """The check must run before anything asks what is already on disk.

    Order is the whole point. Clearing stale text after the planner has been told what
    exists is too late: the planner skips transcription, and the export stage renders the
    previous engine's words as though it had just produced them.
    """

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        source = Path(temporary.name) / "meeting.m4a"
        source.write_bytes(b"audio")
        self.workspace = WorkSpace(source=source, output_dir=Path(temporary.name) / "out")
        self.addCleanup(self.workspace.reset)
        self.workspace.chunks.mkdir(parents=True, exist_ok=True)
        (self.workspace.chunks / "part-0000.asr.json").write_text('{"segments": []}', encoding="utf-8")
        (self.workspace.chunks / "part-0000.wav").write_text("", encoding="utf-8")
        self.workspace.plan_path.write_text('{"cuts": "abc"}', encoding="utf-8")

    def text_exists(self) -> bool:
        return (self.workspace.chunks / "part-0000.asr.json").is_file()

    def test_an_unrecorded_origin_is_treated_as_unknown_and_discarded(self):
        # A plan with no recognition record predates the signature, or was interrupted
        # before the text existed. Nobody wrote down what produced that text, so keeping
        # it would be trusting a guess -- and the safe direction costs one re-transcribe.
        self.assertEqual(self.workspace.discard_stale_recognition({"model": "large-v3"}), 1)

    def test_an_unknown_origin_with_no_text_to_clear_discards_nothing(self):
        (self.workspace.chunks / "part-0000.asr.json").unlink()
        self.assertEqual(self.workspace.discard_stale_recognition({"model": "large-v3"}), 0)

    def test_the_same_settings_keep_the_text(self):
        self.workspace.remember_recognition({"model": "large-v3"})
        self.assertEqual(self.workspace.discard_stale_recognition({"model": "large-v3"}), 0)
        self.assertTrue(self.text_exists())

    def test_different_settings_clear_the_text(self):
        self.workspace.remember_recognition({"model": "large-v3"})
        self.assertEqual(
            self.workspace.discard_stale_recognition({"model": "large-v3-turbo"}), 1
        )
        self.assertFalse(self.text_exists())

    def test_the_recording_of_what_produced_the_text_can_be_read_back(self):
        self.workspace.remember_recognition({"compute_type": "float32"})
        self.assertEqual(
            self.workspace.stored_plan().get("recognition"), {"compute_type": "float32"}
        )

    def test_remembering_keeps_the_slicing_record(self):
        self.workspace.remember_recognition({"model": "large-v3"})
        self.assertEqual(self.workspace.stored_plan().get("cuts"), "abc")

    def test_remembering_without_a_plan_is_harmless(self):
        self.workspace.plan_path.unlink()
        self.workspace.remember_recognition({"model": "large-v3"})
        self.assertIsNone(self.workspace.stored_plan())


if __name__ == "__main__":
    unittest.main()
