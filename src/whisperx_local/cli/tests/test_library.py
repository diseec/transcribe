"""Finding recordings, and expanding what a user typed."""

import tempfile
import unittest
from pathlib import Path

from whisperx_local.cli import library


class IsMediaTest(unittest.TestCase):
    def test_audio_and_video_are_both_media(self):
        # Screen recordings are a normal input for this tool, not an edge case.
        for name in ("a.mp4", "a.mov", "a.mkv", "a.webm", "a.m4a", "a.wav", "a.flac"):
            with self.subTest(name=name):
                self.assertIn(Path(name).suffix, library.MEDIA_SUFFIXES)

    def test_documents_are_not_media(self):
        self.assertNotIn(".txt", library.MEDIA_SUFFIXES)
        self.assertNotIn(".pdf", library.MEDIA_SUFFIXES)


class DiscoverTest(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self):
        self._temporary.cleanup()

    def touch(self, *names: str) -> None:
        for name in names:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x", encoding="utf-8")

    def test_a_directory_is_searched_for_media(self):
        self.touch("a.wav", "b.mp4", "notes.txt")
        found = {path.name for path in library.discover(self.root)}
        self.assertEqual(found, {"a.wav", "b.mp4"})

    def test_subdirectories_are_searched(self):
        self.touch("nested/deep/a.wav")
        self.assertEqual(len(library.discover(self.root)), 1)

    def test_recursion_can_be_turned_off(self):
        self.touch("nested/deep/a.wav")
        self.assertEqual(library.discover(self.root, recursive=False), [])

    def test_dotfiles_are_ignored(self):
        self.touch(".hidden.wav", "shown.wav")
        self.assertEqual([path.name for path in library.discover(self.root)], ["shown.wav"])

    def test_a_missing_directory_gives_nothing(self):
        self.assertEqual(library.discover(self.root / "nope"), [])


class ExpandPathsTest(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self):
        self._temporary.cleanup()

    def write(self, name: str) -> Path:
        target = self.root / name
        target.write_text("x", encoding="utf-8")
        return target

    def test_a_named_file_is_taken_as_given(self):
        target = self.write("recording.mov")
        self.assertEqual(library.expand_paths([str(target)]), [target.resolve()])

    def test_an_unfamiliar_suffix_is_still_accepted_when_named(self):
        # ffmpeg knows formats this list does not; an explicit path is a decision.
        target = self.write("recording.weird")
        self.assertEqual(library.expand_paths([str(target)]), [target.resolve()])

    def test_a_directory_is_expanded(self):
        self.write("a.wav")
        self.write("b.wav")
        self.assertEqual(len(library.expand_paths([str(self.root)])), 2)

    def test_order_is_preserved(self):
        first = self.write("a.wav")
        second = self.write("b.wav")
        found = library.expand_paths([str(second), str(first)])
        self.assertEqual(found, [second.resolve(), first.resolve()])

    def test_a_missing_path_is_skipped_quietly(self):
        self.assertEqual(library.expand_paths([str(self.root / "nope.wav")]), [])

    def test_nothing_typed_gives_nothing(self):
        self.assertEqual(library.expand_paths([]), [])


class PresentableTest(unittest.TestCase):
    def test_the_name_and_size_are_shown(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "meeting.m4a"
            target.write_bytes(b"x" * 1024)
            text = library.presentable(target)
        self.assertIn("meeting.m4a", text)
        self.assertIn("MB", text)

    def test_the_length_is_shown_when_known(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "meeting.m4a"
            target.write_bytes(b"x")
            text = library.presentable(target, seconds=754.0)
        self.assertIn("12m34s", text)

    def test_no_length_means_no_length_field(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "meeting.m4a"
            target.write_bytes(b"x")
            text = library.presentable(target)
        self.assertNotRegex(text, r"\d+m\d+s")


if __name__ == "__main__":
    unittest.main()
