"""The child-process worker: its interface, and the progress line the parent parses."""

import contextlib
import io
import unittest

from whisperx_local.engine import worker


class PercentReporterTest(unittest.TestCase):
    def collect(self, values) -> list[str]:
        stream = io.StringIO()
        reporter = worker.PercentReporter()
        with contextlib.redirect_stdout(stream):
            for value in values:
                reporter(value)
        return stream.getvalue().splitlines()

    def test_only_whole_percent_changes_are_printed(self):
        # The parent repaints on every line, so identical lines are pure noise.
        lines = self.collect([0.1, 0.4, 1.0, 1.2, 1.9, 2.0])
        self.assertEqual(lines, ["Progress: 0%...", "Progress: 1%...", "Progress: 2%..."])

    def test_a_repeated_value_is_not_reprinted(self):
        self.assertEqual(len(self.collect([50.0, 50.0, 50.0])), 1)

    def test_the_format_is_what_the_parent_matches(self):
        self.assertTrue(self.collect([7.0])[0].startswith("Progress: 7%"))


class ParserTest(unittest.TestCase):
    def parse(self, argv):
        return worker.build_parser().parse_args(argv)

    def test_align_needs_its_four_inputs(self):
        args = self.parse(
            ["align", "--segments", "a.json", "--audio", "a.wav",
             "--language", "fa", "--output", "o.json", "--model-dir", "/m"]
        )
        self.assertEqual(args.handler, worker.align_segments)
        self.assertEqual(args.language, "fa")
        self.assertFalse(args.cache_only)

    def test_diarize_accepts_a_speaker_count(self):
        args = self.parse(
            ["diarize", "--audio", "a.wav", "--output", "o.json",
             "--model-dir", "/m", "--min-speakers", "2", "--max-speakers", "2"]
        )
        self.assertEqual(args.handler, worker.separate_speakers)
        self.assertEqual((args.min_speakers, args.max_speakers), (2, 2))

    def test_diarize_allows_an_unknown_speaker_count(self):
        args = self.parse(
            ["diarize", "--audio", "a.wav", "--output", "o.json", "--model-dir", "/m"]
        )
        self.assertIsNone(args.min_speakers)

    def test_cache_only_is_available_on_both_stages(self):
        for stage, extra in (
            ("align", ["--segments", "a", "--audio", "b", "--language", "fa", "--output", "c"]),
            ("diarize", ["--audio", "b", "--output", "c"]),
        ):
            with self.subTest(stage=stage):
                args = self.parse([stage, *extra, "--model-dir", "/m", "--cache-only"])
                self.assertTrue(args.cache_only)

    def test_a_stage_is_required(self):
        with self.assertRaises(SystemExit):
            self.parse([])


if __name__ == "__main__":
    unittest.main()
