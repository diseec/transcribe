"""Running ffmpeg: progress parsing and the deadlock guard.

A mis-parsed progress line is invisible -- the bar simply never moves -- so the
format is pinned here. The stream handling matters more: if stderr is not drained
continuously, ffmpeg blocks once its pipe buffer fills and the run hangs.
"""

import unittest
from unittest import mock

from whisperx_local.media import runner
from whisperx_local.media.runner import parse_progress_line


class FakeProcess:
    """Stands in for subprocess.Popen with ffmpeg-shaped streams."""

    def __init__(self, *, stdout_lines=(), stderr_text="", returncode=0):
        self.stdout = iter(stdout_lines)
        self.stderr = iter(stderr_text.splitlines(keepends=True))
        self._returncode = returncode

    def wait(self):
        return self._returncode


class ParseProgressTest(unittest.TestCase):
    def test_half_way(self):
        self.assertAlmostEqual(parse_progress_line("out_time=00:00:30.000000\n", 60.0), 50.0)

    def test_complete(self):
        self.assertAlmostEqual(parse_progress_line("out_time=00:01:00.000000\n", 60.0), 100.0)

    def test_hours_are_counted(self):
        self.assertAlmostEqual(parse_progress_line("out_time=01:00:00.000000\n", 3600.0), 100.0)

    def test_overshoot_is_clamped(self):
        self.assertAlmostEqual(parse_progress_line("out_time=00:02:00.000000\n", 60.0), 100.0)

    def test_fractional_seconds_are_used(self):
        self.assertAlmostEqual(
            parse_progress_line("out_time=00:00:01.500000\n", 60.0), 2.5
        )

    def test_irrelevant_lines_are_ignored(self):
        self.assertIsNone(parse_progress_line("progress=continue\n", 60.0))
        self.assertIsNone(parse_progress_line("frame=1\n", 60.0))

    def test_unknown_duration_is_ignored(self):
        self.assertIsNone(parse_progress_line("out_time=00:00:10.000000\n", 0.0))


class RunProgressTest(unittest.TestCase):
    def run_with(self, fake, **kwargs):
        with mock.patch.object(runner.subprocess, "Popen", return_value=fake) as popen:
            result = runner.run_ffmpeg_progress("ffmpeg", ["-i", "x"], env={}, **kwargs)
        return result, popen

    def test_progress_callback_receives_percentages(self):
        fake = FakeProcess(
            stdout_lines=[
                "out_time=00:00:15.000000\n",
                "progress=continue\n",
                "out_time=00:00:45.000000\n",
                "progress=end\n",
            ]
        )
        seen = []
        self.run_with(fake, duration=60.0, on_progress=seen.append)
        self.assertEqual(seen, [25.0, 75.0])

    def test_every_stdout_line_reaches_the_line_callback(self):
        # Filter metadata, which the loudness scan needs, arrives on stdout.
        fake = FakeProcess(stdout_lines=["pts_time:0\n", "RMS_level=-12.0\n"])
        seen = []
        self.run_with(fake, on_line=seen.append)
        self.assertEqual(seen, ["pts_time:0\n", "RMS_level=-12.0\n"])

    def test_stderr_is_returned_for_parsing(self):
        fake = FakeProcess(stderr_text="[silencedetect @ 0x7f] silence_start: 5.0\n")
        text, _ = self.run_with(fake)
        self.assertIn("silence_start: 5.0", text)

    def test_stderr_is_also_written_to_the_log(self):
        fake = FakeProcess(stderr_text="boom\n")
        log = mock.MagicMock()
        self.run_with(fake, log=log)
        self.assertIn("boom", log.write.call_args[0][0])

    def test_failure_raises(self):
        with self.assertRaises(Exception):
            self.run_with(FakeProcess(returncode=1))

    def test_progress_options_are_passed_to_ffmpeg(self):
        _, popen = self.run_with(FakeProcess())
        command = popen.call_args[0][0]
        self.assertIn("-progress", command)
        self.assertIn("pipe:1", command)
        self.assertIn("-nostats", command)

    def test_the_arguments_are_appended_not_replaced(self):
        _, popen = self.run_with(FakeProcess())
        self.assertEqual(popen.call_args[0][0][-2:], ["-i", "x"])


if __name__ == "__main__":
    unittest.main()
