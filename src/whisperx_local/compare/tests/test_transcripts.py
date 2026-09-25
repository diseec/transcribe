"""Comparing two transcripts: reading, binning by time, and reporting.

The test that matters most here is the one about alignment. Two engines cut the same
audio differently, so a comparison that pairs segment one with segment one is measuring
segmentation. The fixtures below deliberately give the two sides different segment
shapes, and the answers must still match.
"""

import json
import tempfile
import unittest
from pathlib import Path

from whisperx_local.compare.transcripts import (
    DEFAULT_TERMS,
    UNTIMED_END,
    BinResult,
    Transcript,
    Utterance,
    bin_words,
    compare,
    read,
)
from whisperx_local.compare.metrics import Edits


def timed(*rows) -> Transcript:
    return Transcript(
        label="fixture",
        utterances=[Utterance(start, end, text) for start, end, text in rows],
        timed=True,
    )


class ReadingTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, name: str, body: str) -> Path:
        path = self.root / name
        path.write_text(body, encoding="utf-8")
        return path

    def test_it_reads_the_canonical_json(self):
        path = self.write(
            "a.raw.json",
            json.dumps({"segments": [{"start": 1.0, "end": 2.0, "text": "سلام"}]}),
        )
        transcript = read(path)
        self.assertEqual(len(transcript.utterances), 1)
        self.assertEqual(transcript.utterances[0].text, "سلام")

    def test_it_reads_a_bare_list_too(self):
        path = self.write("a.json", json.dumps([{"start": 0.0, "end": 1.0, "text": "x"}]))
        self.assertEqual(len(read(path).utterances), 1)

    def test_it_reads_the_timestamped_text(self):
        path = self.write("a.raw.txt", "[00:00:01.000 --> 00:00:02.500]  سلام خوبی\n")
        utterance = read(path).utterances[0]
        self.assertEqual((utterance.start, utterance.end), (1.0, 2.5))
        self.assertEqual(utterance.text, "سلام خوبی")

    def test_it_reads_hours(self):
        path = self.write("a.raw.txt", "[01:02:03.000 --> 01:02:04.000]  x\n")
        self.assertEqual(read(path).utterances[0].start, 3723.0)

    def test_it_reads_bare_seconds_too(self):
        # What the earlier version of this app wrote. Failing to recognise it made the
        # file read as untimed, and its header comments got counted as speech.
        path = self.write("a.raw.txt", "[3064.379 --> 3070.589]  سلام\n")
        transcript = read(path)
        self.assertTrue(transcript.timed)
        self.assertAlmostEqual(transcript.utterances[0].start, 3064.379)

    def test_header_comments_are_not_speech(self):
        path = self.write(
            "a.raw.txt",
            "# Partial recovery from the crashed run\n"
            "# Source log: output/.logs/thing.log\n"
            "[1.5 --> 2.5]  سلام خوبی\n",
        )
        transcript = read(path)
        self.assertEqual(len(transcript.utterances), 1)
        self.assertNotIn("Partial recovery", transcript.text)

    def test_segments_without_text_are_skipped(self):
        path = self.write(
            "a.raw.json",
            json.dumps({"segments": [{"start": 0.0, "end": 1.0, "text": "  "}]}),
        )
        self.assertEqual(read(path).utterances, [])

    def test_a_file_with_no_timings_is_still_read_as_one_span(self):
        # Coarse, but refusing would mean no answer at all for a plain .txt.
        transcript = read(self.write("a.txt", "سلام خوبی"))
        self.assertFalse(transcript.timed)
        self.assertEqual(transcript.utterances[0].end, UNTIMED_END)

    def test_an_empty_file_reads_as_nothing(self):
        transcript = read(self.write("a.txt", ""))
        self.assertEqual(transcript.utterances, [])
        self.assertEqual(transcript.words, 0)

    def test_an_unreadable_json_is_not_a_crash(self):
        self.assertEqual(read(self.write("a.json", "not json at all")).utterances, [])

    def test_a_missing_file_says_so(self):
        with self.assertRaises(FileNotFoundError):
            read(self.root / "absent.json")

    def test_confidence_is_averaged_when_present(self):
        path = self.write(
            "a.raw.json",
            json.dumps(
                {
                    "segments": [
                        {"start": 0.0, "end": 1.0, "text": "a", "avg_logprob": -0.2},
                        {"start": 1.0, "end": 2.0, "text": "b", "avg_logprob": -0.4},
                    ]
                }
            ),
        )
        self.assertAlmostEqual(read(path).mean_logprob(), -0.3)

    def test_confidence_is_absent_when_the_file_has_none(self):
        path = self.write("a.raw.json", json.dumps({"segments": [{"start": 0.0, "end": 1.0, "text": "a"}]}))
        self.assertIsNone(read(path).mean_logprob())


class BinningTest(unittest.TestCase):
    def test_words_land_in_the_window_they_mostly_occupy(self):
        gathered = bin_words([Utterance(25.0, 30.0, "یک دو")], 30.0)
        self.assertEqual(list(gathered), [0])

    def test_a_later_utterance_lands_later(self):
        gathered = bin_words([Utterance(31.0, 34.0, "سه")], 30.0)
        self.assertEqual(list(gathered), [1])

    def test_an_utterance_mostly_past_a_boundary_belongs_to_the_later_window(self):
        # 28-33 s is two seconds in the first window and three in the second.
        gathered = bin_words([Utterance(28.0, 33.0, "یک")], 30.0)
        self.assertEqual(list(gathered), [1])

    def test_an_untimed_transcript_is_one_window(self):
        gathered = bin_words([Utterance(0.0, UNTIMED_END, "یک دو سه")], 30.0)
        self.assertEqual(list(gathered), [0])
        self.assertEqual(len(gathered[0]), 3)

    def test_a_zero_window_is_refused(self):
        with self.assertRaises(ValueError):
            bin_words([Utterance(0.0, 1.0, "x")], 0.0)


class AlignmentByTimeTest(unittest.TestCase):
    """The fixtures that decide whether any of this is trustworthy."""

    def test_different_segmentation_of_identical_speech_scores_zero(self):
        # Same words, cut in three pieces on one side and two on the other.
        reference = timed((0.0, 10.0, "یک دو سه"), (10.0, 20.0, "چهار"), (20.0, 30.0, "پنج"))
        other = timed((0.0, 15.0, "یک دو سه چهار"), (15.0, 30.0, "پنج"))
        result = compare(reference, other, window=30.0)
        self.assertEqual(result.totals.errors, 0)
        self.assertEqual(result.agreement, 1.0)

    def test_a_reordered_pair_of_segments_is_still_scored_by_time(self):
        # Segment one is not compared with segment one.
        reference = timed((0.0, 10.0, "الف"), (10.0, 20.0, "ب"))
        other = timed((0.0, 10.0, "الف"), (10.0, 20.0, "ب"))
        self.assertEqual(compare(reference, other).totals.errors, 0)

    def test_a_word_only_in_one_side_is_counted(self):
        reference = timed((0.0, 10.0, "یک دو سه"))
        other = timed((0.0, 10.0, "یک سه"))
        result = compare(reference, other)
        self.assertEqual(result.totals.deletions, 1)
        self.assertEqual(result.totals.insertions, 0)

    def test_an_invented_word_is_an_insertion_not_a_deletion(self):
        reference = timed((0.0, 10.0, "یک سه"))
        other = timed((0.0, 10.0, "یک دو سه"))
        result = compare(reference, other)
        self.assertEqual(result.totals.insertions, 1)
        self.assertEqual(result.totals.deletions, 0)

    def test_a_hole_on_one_side_shows_up_as_missing_words(self):
        reference = timed((0.0, 10.0, "یک"), (100.0, 110.0, "دو"))
        other = timed((0.0, 10.0, "یک"))
        result = compare(reference, other)
        self.assertEqual(result.totals.deletions, 1)
        self.assertEqual([entry.index for entry in result.bins], [0, 3])
        self.assertEqual(result.other.end, 10.0)

    def test_only_windows_with_speech_on_either_side_are_compared(self):
        # The empty windows between two utterances would otherwise count as agreement and
        # flatter both sides.
        reference = timed((0.0, 10.0, "یک"), (100.0, 110.0, "دو"))
        self.assertEqual(len(compare(reference, reference).bins), 2)

    def test_persian_spelling_differences_are_not_scored_as_errors(self):
        # The whole point of the folding: same speech, different keyboard.
        reference = timed((0.0, 10.0, "\u0645\u064a\u200c\u0631\u0648\u0645"))
        other = timed((0.0, 10.0, "میروم"))
        self.assertEqual(compare(reference, other).totals.errors, 0)

    def test_no_folding_would_have_scored_that_as_an_error(self):
        reference = timed((0.0, 10.0, "\u0645\u064a\u200c\u0631\u0648\u0645"))
        other = timed((0.0, 10.0, "میروم"))
        unflolded = compare(reference, other, zwnj=False, punctuation=False)
        self.assertGreater(unflolded.totals.errors, 0)

    def test_an_unscoreable_pair_is_reported_as_such(self):
        result = compare(timed((0.0, 10.0, "یک")), timed(), window=30.0)
        self.assertFalse(result.compared)
        self.assertTrue(any("not comparable" in value for _, value in result.rows()))


class TermTest(unittest.TestCase):
    def test_a_term_present_in_both_is_reported_as_such(self):
        reference = timed((0.0, 10.0, "about the subscription"))
        other = timed((0.0, 10.0, "about the subscription"))
        hits = {hit.term: hit for hit in compare(reference, other).terms}
        self.assertEqual((hits["subscription"].reference, hits["subscription"].other), (1, 1))

    def test_a_term_the_other_side_lost_is_visible(self):
        reference = timed((0.0, 10.0, "the subscription and the database"))
        other = timed((0.0, 10.0, "the database"))
        hits = {hit.term: hit for hit in compare(reference, other).terms}
        self.assertEqual(hits["subscription"].missing, 1)
        self.assertIn("0/1", hits["subscription"].describe())

    def test_the_english_and_persian_forms_are_both_tracked(self):
        self.assertIn("subscription", DEFAULT_TERMS)
        self.assertIn("سابسکریپشن", DEFAULT_TERMS)

    def test_a_multi_word_term_is_counted_as_a_sequence(self):
        reference = timed((0.0, 10.0, "the data base is fine"))
        hits = {
            hit.term: hit
            for hit in compare(reference, timed((0.0, 10.0, "x")), terms=("data base",)).terms
        }
        self.assertEqual(hits["data base"].reference, 1)

    def test_a_term_that_is_absent_counts_zero(self):
        result = compare(timed((0.0, 10.0, "hello")), timed((0.0, 10.0, "hello")), terms=("subscription",))
        self.assertEqual(result.terms[0].reference, 0)


class ReportTest(unittest.TestCase):
    def result(self, reference_text: str, other_text: str):
        return compare(
            timed((0.0, 10.0, reference_text), (40.0, 50.0, "پایان")),
            timed((0.0, 10.0, other_text), (40.0, 50.0, "پایان")),
            window=30.0,
        )

    def test_the_rows_name_both_sides(self):
        rows = dict(self.result("یک دو سه", "یک دو سه").rows())
        self.assertIn("Reference", rows)
        self.assertIn("Compared", rows)

    def test_the_rows_give_a_rate_and_a_breakdown(self):
        rows = dict(self.result("یک دو سه", "یک دو").rows())
        self.assertIn("Errors", rows)
        self.assertIn("deleted", rows["Breakdown"])

    def test_an_identical_pair_reports_full_agreement(self):
        rows = dict(self.result("یک دو سه", "یک دو سه").rows())
        self.assertIn("100%", rows["Agreement"])

    def test_coverage_is_reported_when_both_sides_are_timed(self):
        self.assertIn("Covered", dict(self.result("یک", "یک").rows()))

    def test_an_untimed_comparison_says_so_instead_of_implying_coverage(self):
        result = compare(
            Transcript(label="a", utterances=[Utterance(0.0, UNTIMED_END, "یک دو")], timed=False),
            Transcript(label="b", utterances=[Utterance(0.0, UNTIMED_END, "یک دو")], timed=False),
        )
        self.assertIn("single span", dict(result.rows())["Covered"])

    def test_the_worst_windows_are_ranked_and_shown_with_their_text(self):
        result = self.result("یک دو سه چهار پنج", "کاملا متفاوت")
        lines = result.worst_lines(2)
        self.assertTrue(lines)
        self.assertIn("reference", lines[1])
        self.assertIn("other", lines[2])

    def test_a_perfect_pair_has_no_worst_windows(self):
        self.assertEqual(self.result("یک دو", "یک دو").worst_lines(), [])

    def test_the_confidence_row_appears_only_when_a_logprob_exists(self):
        plain = compare(timed((0.0, 10.0, "یک")), timed((0.0, 10.0, "یک")))
        self.assertNotIn("Confidence", dict(plain.rows()))

    def test_the_confidence_row_compares_both_sides(self):
        left = Transcript(label="a", utterances=[Utterance(0.0, 5.0, "یک", logprob=-0.5)])
        right = Transcript(label="b", utterances=[Utterance(0.0, 5.0, "یک", logprob=-0.2)])
        shown = dict(compare(left, right).rows()).get("Confidence", "")
        self.assertIn("-0.50", shown)
        self.assertIn("-0.20", shown)

    def test_the_window_description_names_the_span_it_covers(self):
        entry = BinResult(2, 60.0, 90.0, "یک", "دو", Edits(1, 1))
        self.assertIn("60s–90s", entry.describe())


if __name__ == "__main__":
    unittest.main()
