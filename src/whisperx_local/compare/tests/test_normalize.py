"""Persian folding, where over-folding is as damaging as under-folding.

Two failures matter here and they point in opposite directions. Fold too little and two
engines look far more different than they are, because one wrote ``ی`` where the other
wrote ``ي``. Fold too much and two genuinely different words compare equal, which hides
the mistake the comparison exists to find. The rules that risk the second kind are
therefore off unless asked for, and those are the ones tested hardest.
"""

import unittest

from whisperx_local.compare.normalize import (
    ARABIC_DIGITS,
    PERSIAN_DIGITS,
    SAME_LETTER,
    fold,
    variants,
    words,
)


class SameLetterTest(unittest.TestCase):
    """The same letter on a different keyboard. Never a real difference."""

    def test_arabic_yeh_becomes_persian_yeh(self):
        self.assertEqual(fold("\u0645\u064a"), fold("\u0645\u06cc"))

    def test_arabic_kaf_becomes_persian_keh(self):
        self.assertEqual(fold("\u062f\u0648\u0631\u0643"), fold("\u062f\u0648\u0631\u06a9"))

    def test_every_mapped_letter_folds_to_its_persian_form(self):
        for source, target in SAME_LETTER.items():
            with self.subTest(codepoint=hex(ord(source))):
                self.assertEqual(fold(source), target)

    def test_a_word_written_two_ways_is_one_word(self):
        self.assertEqual(words("\u0633\u0627\u0628\u0633\u0643\u0631\u064a\u067e\u0634\u0646"), words("سابسکریپشن"))


class MeaninglessMarkTest(unittest.TestCase):
    def test_diacritics_are_removed(self):
        self.assertEqual(fold("\u0633\u064e\u0644\u0627\u0645"), "سلام")

    def test_the_decorative_kashida_is_removed(self):
        self.assertEqual(fold("\u0633\u0640\u0644\u0627\u0645"), "سلام")

    def test_direction_marks_are_removed(self):
        self.assertEqual(fold("\u200f\u0633\u0644\u0627\u0645\u200e"), "سلام")


class DigitTest(unittest.TestCase):
    def test_persian_digits_become_ascii(self):
        self.assertEqual(fold(PERSIAN_DIGITS), "0123456789")

    def test_arabic_digits_become_ascii(self):
        self.assertEqual(fold(ARABIC_DIGITS), "0123456789")

    def test_a_number_written_in_either_alphabet_is_the_same_number(self):
        self.assertEqual(fold("۲۰۲۶"), fold("2026"))


class ZwnjTest(unittest.TestCase):
    """The zero-width non-joiner, which is a spelling choice, not a spoken difference."""

    def test_it_is_removed_by_default(self):
        self.assertEqual(fold("\u0645\u06cc\u200c\u0631\u0648\u0645", zwnj=True), "میروم")
        self.assertEqual(fold("\u0645\u06cc\u200c\u0631\u0648\u0645"), "میروم")

    def test_keeping_it_is_possible(self):
        self.assertNotEqual(fold("\u0645\u06cc\u200c\u0631\u0648\u0645", zwnj=False), "میروم")

    def test_both_spellings_compare_equal_by_default(self):
        self.assertEqual(fold("\u0645\u06cc\u200c\u0631\u0648\u0645"), fold("\u0645\u06cc\u0631\u0648\u0645"))


class DistinctLetterTest(unittest.TestCase):
    """Folding these is a real decision, so it has to be asked for."""

    def test_alef_with_madda_is_kept_apart_by_default(self):
        # ``آب`` and ``اب`` are different words; merging them hides a real error.
        self.assertNotEqual(fold("\u0622\u0628"), fold("\u0627\u0628"))

    def test_alef_with_madda_can_be_folded_on_request(self):
        self.assertEqual(
            fold("\u0622\u0628", distinct_letters=True), fold("\u0627\u0628")
        )

    def test_a_hamza_carrier_is_kept_apart_by_default(self):
        self.assertNotEqual(fold("\u0624", distinct_letters=False), "و")

    def test_a_hamza_carrier_can_be_folded_on_request(self):
        self.assertEqual(fold("\u0624", distinct_letters=True), "و")


class PunctuationTest(unittest.TestCase):
    def test_punctuation_separates_rather_than_joins(self):
        # Replacing with a space, not deleting, keeps two words from becoming one.
        self.assertEqual(fold("سلام،خوبی"), "سلام خوبی")

    def test_it_can_be_kept(self):
        self.assertIn("،", fold("سلام،خوبی", punctuation=False))

    def test_latin_technical_terms_survive(self):
        # These are the words this domain actually gets wrong, so losing them would be
        # losing the evidence.
        for term in ("subscription", "database", "UI", "Pro"):
            with self.subTest(term=term):
                self.assertEqual(fold(term), term)


class ShapeTest(unittest.TestCase):
    def test_whitespace_is_collapsed_and_trimmed(self):
        self.assertEqual(fold("  سلام \n خوبی  "), "سلام خوبی")

    def test_a_tab_between_words_is_one_space(self):
        self.assertEqual(fold("سلام\tخوبی"), "سلام خوبی")

    def test_nothing_is_an_empty_string(self):
        self.assertEqual(fold(None), "")

    def test_folding_is_idempotent(self):
        once = fold("مي‌روم ۲۰۲۶، خوبی")
        self.assertEqual(fold(once), once)

    def test_words_splits_what_fold_produced(self):
        self.assertEqual(words("مي‌روم ۲۰۲۶"), ["میروم", "2026"])

    def test_variants_returns_the_foldings_of_one_term(self):
        self.assertGreaterEqual(len(variants("\u0645\u06cc\u200c\u0631\u0648\u0645")), 1)


if __name__ == "__main__":
    unittest.main()
