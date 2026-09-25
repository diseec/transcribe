"""Counting errors, and separating the two ways of making them.

An engine that inserts words is inventing speech; one that deletes them is missing it.
A single rate cannot tell those apart, which is why the counts are kept separately and
tested separately.
"""

import unittest

from whisperx_local.compare.metrics import MAX_ALIGNMENT_CELLS, Edits, align, distance


class DistanceTest(unittest.TestCase):
    def test_identical_sequences_are_free(self):
        self.assertEqual(distance(["a", "b"], ["a", "b"]), 0)

    def test_an_empty_hypothesis_costs_a_deletion_each(self):
        self.assertEqual(distance(["a", "b"], []), 2)

    def test_an_empty_reference_costs_an_insertion_each(self):
        self.assertEqual(distance([], ["a", "b"]), 2)

    def test_a_substitution_costs_one(self):
        self.assertEqual(distance(["a"], ["b"]), 1)

    def test_both_empty_is_distance_zero(self):
        self.assertEqual(distance([], []), 0)

    def test_it_agrees_with_the_alignment_total(self):
        reference = "one two three four five".split()
        hypothesis = "one too three for five six".split()
        self.assertEqual(distance(reference, hypothesis), align(reference, hypothesis).errors)


class SplitTest(unittest.TestCase):
    def test_a_substitution_is_counted_as_one(self):
        edits = align(["a", "b", "c"], ["a", "x", "c"])
        self.assertEqual((edits.substitutions, edits.insertions, edits.deletions), (1, 0, 0))

    def test_a_missing_word_is_a_deletion(self):
        edits = align(["a", "b", "c"], ["a", "c"])
        self.assertEqual((edits.substitutions, edits.insertions, edits.deletions), (0, 0, 1))

    def test_an_extra_word_is_an_insertion(self):
        edits = align(["a", "c"], ["a", "b", "c"])
        self.assertEqual((edits.substitutions, edits.insertions, edits.deletions), (0, 1, 0))

    def test_an_invented_sentence_does_not_look_like_a_missed_one(self):
        invented = align(["a"], ["a", "b", "c", "d"])
        missed = align(["a", "b", "c", "d"], ["a"])
        self.assertEqual(invented.errors, missed.errors)
        self.assertGreater(invented.insertions, invented.deletions)
        self.assertGreater(missed.deletions, missed.insertions)

    def test_a_substitution_is_preferred_to_a_deletion_plus_an_insertion(self):
        # The cheaper explanation is the standard one, and it keeps the counts honest.
        edits = align(["a", "b"], ["a", "x"])
        self.assertEqual(edits.errors, 1)
        self.assertEqual(edits.substitutions, 1)

    def test_nothing_to_align_is_no_error(self):
        self.assertEqual(align([], []).errors, 0)


class RateTest(unittest.TestCase):
    def test_the_rate_is_errors_per_reference_word(self):
        edits = Edits(reference_length=100, substitutions=5, insertions=3, deletions=2)
        self.assertAlmostEqual(edits.rate, 0.10)
        self.assertAlmostEqual(edits.accuracy, 0.90)

    def test_no_reference_has_no_rate(self):
        # Dividing by nothing would produce a confident, meaningless number.
        self.assertIsNone(Edits(reference_length=0, insertions=4).rate)
        self.assertIsNone(Edits(reference_length=0).accuracy)

    def test_correct_counts_the_words_it_got_right(self):
        edits = Edits(reference_length=10, substitutions=2, deletions=1)
        self.assertEqual(edits.correct, 7)

    def test_correct_never_goes_negative(self):
        self.assertEqual(Edits(reference_length=1, substitutions=5).correct, 0)

    def test_merging_adds_every_counter(self):
        total = Edits(10, 1, 2, 3).merge(Edits(5, 4, 5, 6))
        self.assertEqual((total.reference_length, total.substitutions), (15, 5))
        self.assertEqual((total.insertions, total.deletions), (7, 9))

    def test_the_description_names_all_three_kinds(self):
        shown = Edits(100, 1, 2, 3).describe()
        self.assertIn("1 substituted", shown)
        self.assertIn("2 inserted", shown)
        self.assertIn("3 deleted", shown)

    def test_an_unscoreable_pair_says_so_rather_than_showing_zero(self):
        self.assertIn("no reference", Edits(reference_length=0).describe())


class GuardTest(unittest.TestCase):
    def test_an_enormous_pair_is_refused_rather_than_attempted(self):
        # A whole transcript must be scored in windows; this is the guard that says so
        # instead of quietly allocating gigabytes.
        size = int(MAX_ALIGNMENT_CELLS**0.5) + 2
        with self.assertRaises(ValueError):
            align(list(range(size)), list(range(size)))

    def test_a_large_but_reasonable_pair_is_scored(self):
        size = int(MAX_ALIGNMENT_CELLS**0.5) - 10
        self.assertEqual(align(list(range(size)), list(range(size))).errors, 0)


if __name__ == "__main__":
    unittest.main()
