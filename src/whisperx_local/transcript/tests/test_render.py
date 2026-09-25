"""Rendering: exact output shapes for every supported format.

Subtitle files are consumed by other programs, so a timestamp format that is merely
close is a failure. SubRip requires a comma where WebVTT requires a dot, and the
hour rollover is easy to get wrong.
"""

import json
import unittest

from whisperx_local.transcript.render import (
    FORMAT_ORDER,
    RENDERERS,
    render_aud,
    render_srt,
    render_tsv,
    render_txt,
    render_vtt,
    timestamp,
)
from whisperx_local.transcript.turns import Line, Turn


def sample_lines():
    return [Line("Speaker 1", 0.0, 1.5, [{"word": "سلام", "start": 0.0, "end": 1.5}])]


def sample_segments():
    return [
        {
            "start": 0.0,
            "end": 1.5,
            "text": "سلام",
            "words": [{"word": "سلام", "start": 0.0, "end": 1.5}],
        }
    ]


class TimestampTest(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(timestamp(0.0), "00:00:00.000")

    def test_rollover_past_an_hour(self):
        self.assertEqual(timestamp(3661.5), "01:01:01.500")

    def test_milliseconds_are_padded(self):
        self.assertEqual(timestamp(1.01), "00:00:01.010")

    def test_comma_form_is_used_by_subrip(self):
        self.assertEqual(timestamp(1.5, comma=True), "00:00:01,500")

    def test_a_negative_time_is_clamped(self):
        self.assertEqual(timestamp(-5.0), "00:00:00.000")


class RendererOutputTest(unittest.TestCase):
    def test_txt_labels_the_speaker(self):
        self.assertIn("Speaker 1: سلام", render_txt(sample_lines()))

    def test_txt_without_speakers_omits_the_label(self):
        lines = [Line(None, 0.0, 1.0, [{"word": "hello", "start": 0.0, "end": 1.0}])]
        self.assertEqual(render_txt(lines).strip(), "hello")

    def test_txt_can_include_timestamps(self):
        self.assertIn("[00:00:00.000]", render_txt(sample_lines(), timestamps=True))

    def test_srt_uses_comma_timestamps(self):
        self.assertIn("00:00:00,000", render_srt(sample_lines()))

    def test_srt_numbers_its_cues_from_one(self):
        self.assertTrue(render_srt(sample_lines()).startswith("1\n"))

    def test_vtt_starts_with_header(self):
        self.assertTrue(render_vtt(sample_lines()).startswith("WEBVTT"))

    def test_vtt_uses_dot_timestamps(self):
        self.assertIn("00:00:00.000", render_vtt(sample_lines()))

    def test_tsv_has_a_header_row(self):
        self.assertTrue(render_tsv(sample_lines()).startswith("start\tend\tspeaker\ttext"))

    def test_tsv_keeps_its_columns(self):
        row = render_tsv(sample_lines()).splitlines()[1]
        self.assertEqual(len(row.split("\t")), 4)

    def test_aud_uses_six_decimals(self):
        self.assertIn("0.000000\t1.500000", render_aud(sample_lines()))

    def test_an_empty_transcript_still_renders_a_valid_document(self):
        # Subtitles and the JSON envelope are still well-formed with no speech in
        # them; a file that cannot be parsed is worse than an empty one.
        self.assertEqual(render_txt([]), "")
        self.assertEqual(render_srt([]), "")
        self.assertEqual(render_aud([]), "")
        self.assertTrue(render_vtt([]).startswith("WEBVTT"))
        self.assertEqual(render_tsv([]), "start\tend\tspeaker\ttext\n")
        payload = json.loads(RENDERERS["json"]([], [], "fa"))
        self.assertEqual(payload["segments"], [])


class RendererContractTest(unittest.TestCase):
    def test_every_advertised_format_has_a_renderer(self):
        self.assertEqual(set(FORMAT_ORDER), set(RENDERERS))

    def test_all_renderers_produce_output(self):
        turns = [Turn(0.0, 1.5, "SPEAKER_00")]
        for name, renderer in RENDERERS.items():
            with self.subTest(renderer=name):
                self.assertTrue(renderer(sample_segments(), turns, "fa").strip())

    def test_the_json_renderer_carries_word_speakers(self):
        import json

        turns = [Turn(0.0, 1.5, "SPEAKER_00")]
        payload = json.loads(RENDERERS["json"](sample_segments(), turns, "fa"))
        self.assertEqual(payload["language"], "fa")
        self.assertEqual(payload["segments"][0]["words"][0]["speaker"], "Speaker 1")

    def test_the_json_renderer_is_unicode_preserving(self):
        turns = [Turn(0.0, 1.5, "SPEAKER_00")]
        self.assertIn("سلام", RENDERERS["json"](sample_segments(), turns, "fa"))


if __name__ == "__main__":
    unittest.main()
