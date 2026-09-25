"""Boundary scanning: silence parsing and the loudness fallback.

Both are parsed from ffmpeg output, so a mis-parse would silently cut words in half
or refuse to cut at all. The loudness side also has to know when its own signal is
worthless, because flat recordings produce convincing-looking minima that mean
nothing.
"""

import unittest
from pathlib import Path
from unittest import mock

from whisperx_local.media import runner, scan
from whisperx_local.media.filters import SILENT_DB
from whisperx_local.media.tests.test_runner import FakeProcess
from whisperx_local.media.scan import silence_cut_points


def silence_lines(*pairs):
    lines = []
    for start, end in pairs:
        lines.append(f"[silencedetect @ 0x7f] silence_start: {start}\n")
        lines.append(
            f"[silencedetect @ 0x7f] silence_end: {end} | silence_duration: {end - start}\n"
        )
    return "".join(lines)


class DetectSilencesTest(unittest.TestCase):
    def detect(self, stderr: str, **kwargs):
        fake = FakeProcess(stderr_text=stderr)
        with mock.patch.object(runner.subprocess, "Popen", return_value=fake):
            return scan.detect_silences(
                "ffmpeg", Path("recording.m4a"), env={}, noise_db=-40.0, **kwargs
            )

    def test_pairs_are_parsed(self):
        spans = self.detect(silence_lines((12.345, 13.567), (40.0, 41.0)))
        self.assertEqual(spans, [(12.345, 13.567), (40.0, 41.0)])

    def test_no_silence_gives_nothing(self):
        self.assertEqual(self.detect("nothing here"), [])

    def test_trailing_silence_uses_the_known_duration(self):
        spans = self.detect("[silencedetect @ 0x7f] silence_start: 95.0\n", duration=100.0)
        self.assertEqual(spans, [(95.0, 100.0)])

    def test_trailing_silence_is_dropped_without_a_duration(self):
        self.assertEqual(self.detect("[silencedetect @ 0x7f] silence_start: 95.0\n"), [])

    def test_inverted_spans_are_discarded(self):
        spans = self.detect(
            "[silencedetect @ 0x7f] silence_start: 10.0\n"
            "[silencedetect @ 0x7f] silence_end: 10.0 | silence_duration: 0.0\n"
        )
        self.assertEqual(spans, [])

    def test_threshold_is_derived_when_not_supplied(self):
        # A quiet recording must not reuse a fixed threshold, or speech reads as
        # silence and the planner starts cutting through words.
        fake = FakeProcess()
        with mock.patch.object(scan, "mean_volume_db", return_value=-30.0) as mean, \
             mock.patch.object(runner.subprocess, "Popen", return_value=fake) as popen:
            scan.detect_silences("ffmpeg", Path("a.m4a"), env={})
        mean.assert_called_once()
        chain = next(item for item in popen.call_args[0][0] if "silencedetect" in item)
        self.assertIn("-52.0dB", chain)

    def test_a_measured_recording_replaces_the_fallback_threshold(self):
        with mock.patch.object(scan, "mean_volume_db", return_value=None), \
             mock.patch.object(runner.subprocess, "Popen", return_value=FakeProcess()) as popen:
            scan.detect_silences("ffmpeg", Path("a.m4a"), env={})
        chain = next(item for item in popen.call_args[0][0] if "silencedetect" in item)
        self.assertIn(f"{scan.FALLBACK_THRESHOLD_DB}dB", chain)


class CutPointsTest(unittest.TestCase):
    def test_midpoints_are_used(self):
        self.assertEqual(silence_cut_points([(10.0, 12.0), (30.0, 31.0)]), [11.0, 30.5])

    def test_rounding_is_stable(self):
        self.assertEqual(silence_cut_points([(0.0001, 0.0003)]), [0.0])

    def test_empty_input(self):
        self.assertEqual(silence_cut_points([]), [])


class ParseEnergyLinesTest(unittest.TestCase):
    def test_timestamp_is_paired_with_the_following_level(self):
        lines = [
            "[Parsed_ametadata_3 @ 0x1] frame:0    pts:0       pts_time:0\n",
            "[Parsed_ametadata_3 @ 0x1] lavfi.astats.Overall.RMS_level=-12.016072\n",
            "[Parsed_ametadata_3 @ 0x1] frame:1    pts:4000    pts_time:0.25\n",
            "[Parsed_ametadata_3 @ 0x1] lavfi.astats.Overall.RMS_level=-13.5\n",
        ]
        self.assertEqual(
            scan.parse_energy_lines(lines), [(0.0, -12.016072), (0.25, -13.5)]
        )

    def test_level_without_a_timestamp_is_ignored(self):
        self.assertEqual(
            scan.parse_energy_lines(["lavfi.astats.Overall.RMS_level=-12.0\n"]), []
        )

    def test_digital_silence_is_clamped(self):
        # -inf cannot survive arithmetic, and digital silence is the best cut point.
        lines = ["pts_time:1.0\n", "lavfi.astats.Overall.RMS_level=-inf\n"]
        self.assertEqual(scan.parse_energy_lines(lines), [(1.0, SILENT_DB)])

    def test_nan_becomes_zero(self):
        self.assertEqual(
            scan.parse_energy_lines(["pts_time:2.0\n", "RMS_level=nan\n"]), [(2.0, 0.0)]
        )

    def test_empty_input(self):
        self.assertEqual(scan.parse_energy_lines([]), [])


class EnergyWindowsTest(unittest.TestCase):
    def scan_with(self, **kwargs):
        fake = FakeProcess(**kwargs)
        with mock.patch.object(runner.subprocess, "Popen", return_value=fake) as popen:
            samples = scan.energy_windows("ffmpeg", Path("a.m4a"), env={})
        return samples, popen.call_args[0][0]

    def test_samples_are_read_from_stdout(self):
        samples, _ = self.scan_with(
            stdout_lines=[
                "pts_time:0\n",
                "lavfi.astats.Overall.RMS_level=-20.0\n",
                "pts_time:0.25\n",
                "lavfi.astats.Overall.RMS_level=-40.0\n",
            ]
        )
        self.assertEqual(samples, [(0.0, -20.0), (0.25, -40.0)])

    def test_stderr_is_used_when_stdout_carries_none(self):
        samples, _ = self.scan_with(
            stderr_text="pts_time:3.0\nlavfi.astats.Overall.RMS_level=-9.0\n"
        )
        self.assertEqual(samples, [(3.0, -9.0)])

    def test_duplicate_timestamps_are_collapsed(self):
        samples, _ = self.scan_with(
            stdout_lines=[
                "pts_time:0\n",
                "RMS_level=-20.0\n",
                "pts_time:0\n",
                "RMS_level=-20.0\n",
            ]
        )
        self.assertEqual(samples, [(0.0, -20.0)])

    def test_the_scan_asks_for_windowed_metadata(self):
        _, command = self.scan_with()
        chain = next(item for item in command if "asetnsamples" in item)
        self.assertIn("astats=metadata=1:reset=1", chain)
        self.assertIn("ametadata=mode=print", chain)
        self.assertIn("RMS_level", chain)


if __name__ == "__main__":
    unittest.main()
