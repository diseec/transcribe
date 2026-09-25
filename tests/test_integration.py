"""Cross-cutting tests: the seams between packages, not any one of them.

Each package is tested beside its own code. What cannot be tested there is whether the
pieces agree -- whether a plan and the artifact scanner describe the same recording the
same way, and whether text that went through chunking, merging and rendering comes out
whole. Those are the failures that only appear in a real run.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from whisperx_local.actions import catalog, plan as planning
from whisperx_local.actions.plan import Selection
from whisperx_local.chunking import merge_segments, plan_chunks
from whisperx_local.services.artifacts import WorkSpace
from whisperx_local.services.jobs import fingerprint, is_complete, mark_complete
from whisperx_local.transcript import RENDERERS, write_formats


class SelectionAgreementTest(unittest.TestCase):
    def test_planning_defaults_matches_the_catalog(self):
        # If these ever disagree, `resolve` would add or drop the wrong stages.
        result = planning.resolve(Selection(), available=())
        self.assertEqual(result.keys(), catalog.DEFAULT_ACTIONS)

    def test_a_subset_plan_only_contains_known_actions(self):
        for action in catalog.ACTIONS:
            with self.subTest(action=action.key):
                result = planning.resolve(Selection(only=(action.key,)), available=())
                self.assertTrue(set(result.keys()) <= set(catalog.keys()))
                self.assertIn(action.key, result.keys())


class WorkspaceAgreementTest(unittest.TestCase):
    """The scanner and the planner have to describe a recording the same way."""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.source = self.root / "meeting.m4a"
        self.source.write_bytes(b"audio")

    def tearDown(self):
        self._temporary.cleanup()

    def workspace(self) -> WorkSpace:
        return WorkSpace(source=self.source, output_dir=self.root / "out")

    def test_an_untouched_recording_has_nothing_available(self):
        present = self.workspace().available([], speakers=None, output_format="txt")
        self.assertNotIn("segments", present)
        self.assertNotIn("words", present)
        self.assertNotIn("turns", present)
        self.assertNotIn("files", present)

    def test_recognised_chunks_make_segments_available(self):
        workspace = self.workspace()
        chunks = [plan_chunks(60.0, chunk_seconds=30, overlap_seconds=2)[0]]
        workspace.chunks.mkdir(parents=True, exist_ok=True)
        for chunk in chunks:
            (workspace.chunks / f"part-{chunk.index:04d}.asr.json").write_text(
                json.dumps({"segments": []}), encoding="utf-8"
            )
        present = workspace.available(chunks, speakers=None, output_format="txt")
        self.assertIn("segments", present)
        self.assertNotIn("words", present)

    def test_aligned_chunks_imply_both_segments_and_words(self):
        workspace = self.workspace()
        chunks = plan_chunks(60.0, chunk_seconds=30, overlap_seconds=2)
        workspace.chunks.mkdir(parents=True, exist_ok=True)
        for chunk in chunks:
            (workspace.chunks / f"part-{chunk.index:04d}.aligned.json").write_text(
                json.dumps({"segments": []}), encoding="utf-8"
            )
        present = workspace.available(chunks, speakers=None, output_format="txt")
        self.assertIn("segments", present)
        self.assertIn("words", present)

    def test_a_partly_recognised_recording_is_not_treated_as_done(self):
        # Claiming segments exist while a chunk is still missing would skip it forever.
        workspace = self.workspace()
        chunks = plan_chunks(60.0, chunk_seconds=30, overlap_seconds=2)
        workspace.chunks.mkdir(parents=True, exist_ok=True)
        (workspace.chunks / "part-0000.asr.json").write_text(
            json.dumps({"segments": []}), encoding="utf-8"
        )
        self.assertNotIn(
            "segments", workspace.available(chunks, speakers=None, output_format="txt")
        )

    def test_a_stored_plan_reproduces_its_own_chunks(self):
        # A run that starts mid-plan must slice exactly as the plan describes.
        workspace = self.workspace()
        workspace.chunks.mkdir(parents=True, exist_ok=True)
        cuts = [31.0, 61.0]
        (workspace.plan_path).write_text(
            json.dumps(
                {"chunk_seconds": 30.0, "overlap_seconds": 2.0, "boundaries": cuts}
            ),
            encoding="utf-8",
        )
        rebuilt = workspace.stored_chunks(
            100.0, chunk_seconds=30.0, overlap_seconds=2.0
        )
        expected = plan_chunks(
            100.0, chunk_seconds=30.0, overlap_seconds=2.0, cut_points=cuts
        )
        self.assertEqual(
            [chunk.start for chunk in rebuilt], [chunk.start for chunk in expected]
        )

    def test_speaker_turns_are_keyed_by_the_requested_count(self):
        # Reusing a previous answer for a different speaker count would be wrong.
        workspace = self.workspace()
        self.assertNotEqual(
            workspace.turns_path(2), workspace.turns_path(None)
        )


class EndToEndTextTest(unittest.TestCase):
    """Words in, transcript out, through chunking and merging as a run would do it."""

    def test_words_survive_chunking_merging_and_every_renderer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = plan_chunks(12.0, chunk_seconds=6, overlap_seconds=1)
            payloads = {
                0: [{"start": 0.0, "end": 3.0, "text": "یک دو",
                     "words": [{"word": "یک", "start": 0.0, "end": 1.5},
                               {"word": "دو", "start": 1.5, "end": 3.0}]}],
                1: [{"start": 0.0, "end": 3.0, "text": "سه",
                     "words": [{"word": "سه", "start": 0.0, "end": 3.0}]}],
            }
            merged = merge_segments(chunks, payloads, duration=12.0)
            self.assertTrue(merged)
            turns = []

            for name in RENDERERS:
                with self.subTest(renderer=name):
                    written = write_formats(
                        root, "meeting", merged, turns, language="fa", formats=[name]
                    )
                    self.assertTrue(written[0].is_file())

            plain = (root / "meeting.txt").read_text(encoding="utf-8")
            self.assertIn("یک", plain)
            for word in ("یک", "دو", "سه"):
                self.assertIn(word, plain)

    def test_no_word_is_lost_or_duplicated_across_a_boundary(self):
        chunks = plan_chunks(12.0, chunk_seconds=6, overlap_seconds=2)
        # The same moment of audio, expressed in each chunk's own local time: the
        # second chunk's clock starts two seconds earlier. Both land on 4.0-6.0s of
        # the recording, and exactly one of them may survive the merge.
        first = {"start": 4.0, "end": 6.0, "text": "boundary"}
        second = {"start": 0.0, "end": 2.0, "text": "boundary"}
        merged = merge_segments(chunks, {0: [first], 1: [second]}, duration=12.0)
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0]["start"], 4.0)
        self.assertAlmostEqual(merged[0]["end"], 6.0)


class JobStateTest(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.source = self.root / "meeting.m4a"
        self.source.write_bytes(b"audio")
        self.output = self.root / "meeting.txt"

    def tearDown(self):
        self._temporary.cleanup()

    def options(self, **overrides):
        base = {"language": "fa", "output_format": "txt", "profile": "fast"}
        base.update(overrides)
        return type("Options", (), base)()

    def test_a_finished_run_is_recognised(self):
        self.output.write_text("text", encoding="utf-8")
        mark_complete(self.source, self.options())
        self.assertTrue(is_complete(self.source, self.options(), self.output))

    def test_a_changed_setting_defeats_the_marker(self):
        self.output.write_text("text", encoding="utf-8")
        mark_complete(self.source, self.options())
        self.assertFalse(
            is_complete(self.source, self.options(language="en"), self.output)
        )

    def test_a_missing_output_defeats_the_marker(self):
        mark_complete(self.source, self.options())
        self.assertFalse(is_complete(self.source, self.options(), self.output))

    def test_the_fingerprint_notices_a_changed_source(self):
        before = fingerprint(self.source, self.options())
        self.source.write_bytes(b"different audio entirely")
        self.assertNotEqual(before, fingerprint(self.source, self.options()))


if __name__ == "__main__":
    unittest.main()
