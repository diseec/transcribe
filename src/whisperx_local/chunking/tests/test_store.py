"""The chunk cache: fingerprints, atomic writes, and what counts as finished.

The fingerprint exists because counting cut points was not enough: moving every
boundary while keeping the count identical silently reused chunks cut on the old
boundaries. The "finished" rule exists because treating recognised-but-unaligned
chunks as done would skip their alignment forever.
"""

import tempfile
import unittest
from pathlib import Path

from whisperx_local.chunking.plan import Chunk
from whisperx_local.chunking.store import (
    atomic_write_json,
    chunk_paths,
    completed_indices,
    plan_signature,
    read_segments,
)


class SignatureTest(unittest.TestCase):
    def test_slicing_changes_change_the_signature(self):
        source = Path(__file__)
        first = plan_signature(source=source, duration=100.0, chunk_seconds=600, overlap_seconds=2)
        second = plan_signature(source=source, duration=100.0, chunk_seconds=300, overlap_seconds=2)
        third = plan_signature(source=source, duration=100.0, chunk_seconds=600, overlap_seconds=5)
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertNotEqual(second, third)

    def test_the_signature_follows_the_actual_boundaries(self):
        # Same count, different positions: without this the stale chunks survive.
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "a.wav"
            source.write_text("x", encoding="utf-8")
            first = plan_signature(
                source=source, duration=100, chunk_seconds=30,
                overlap_seconds=2, cut_points=[30.0, 60.0],
            )
            moved = plan_signature(
                source=source, duration=100, chunk_seconds=30,
                overlap_seconds=2, cut_points=[31.0, 61.0],
            )
        self.assertEqual(first["cut_points"], moved["cut_points"])
        self.assertNotEqual(first["cuts"], moved["cuts"])

    def test_a_changed_duration_changes_the_signature(self):
        source = Path(__file__)
        first = plan_signature(source=source, duration=100.0, chunk_seconds=600, overlap_seconds=2)
        second = plan_signature(source=source, duration=101.0, chunk_seconds=600, overlap_seconds=2)
        self.assertNotEqual(first, second)


class CacheIoTest(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "part.json"
            atomic_write_json(target, {"segments": [{"start": 1.0, "text": "x"}]})
            self.assertEqual(read_segments(target), [{"start": 1.0, "text": "x"}])

    def test_directories_are_created(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "a" / "b" / "part.json"
            atomic_write_json(target, {"segments": []})
            self.assertTrue(target.is_file())

    def test_no_partial_file_is_left_behind(self):
        # A half-written file would be indistinguishable from a finished one.
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "part.json"
            atomic_write_json(target, {"segments": []})
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_a_bare_list_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "part.json"
            target.write_text('[{"start": 1.0}]', encoding="utf-8")
            self.assertEqual(read_segments(target), [{"start": 1.0}])

    def test_a_missing_segments_key_gives_an_empty_list(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "part.json"
            target.write_text('{"language": "fa"}', encoding="utf-8")
            self.assertEqual(read_segments(target), [])

    def test_unicode_survives_the_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "part.json"
            atomic_write_json(target, {"segments": [{"text": "برد"}]})
            self.assertEqual(read_segments(target)[0]["text"], "برد")


class ChunkPathsTest(unittest.TestCase):
    def test_names_are_zero_padded_and_ordered(self):
        paths = chunk_paths(Path("/w"), Chunk(7, 0.0, 10.0))
        self.assertEqual(paths["audio"].name, "part-0007.wav")
        self.assertEqual(paths["recognised"].name, "part-0007.asr.json")
        self.assertEqual(paths["aligned"].name, "part-0007.aligned.json")

    def test_sorting_by_name_matches_chunk_order(self):
        # Two-digit padding is not enough past 99 chunks, so four is used.
        names = [chunk_paths(Path("/w"), Chunk(index, 0.0, 1.0))["audio"].name for index in (2, 10, 100)]
        self.assertEqual(names, sorted(names))

    def test_only_aligned_chunks_count_as_finished(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = [Chunk(0, 0.0, 10.0), Chunk(1, 10.0, 10.0)]
            chunk_paths(root, chunks[0])["recognised"].write_text("[]", encoding="utf-8")
            self.assertEqual(completed_indices(chunks, root), set())
            chunk_paths(root, chunks[0])["aligned"].write_text("[]", encoding="utf-8")
            self.assertEqual(completed_indices(chunks, root), {0})

    def test_nothing_is_finished_in_an_empty_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(completed_indices([Chunk(0, 0.0, 10.0)], Path(directory)), set())


if __name__ == "__main__":
    unittest.main()
