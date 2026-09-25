"""Persian-aware folding, so a comparison compares words and not keyboards.

Two transcripts of the same Persian audio differ in ways that are not disagreements at
all. Persian is written in the Arabic script, where the same word can be typed with either
codepoint of an identically-shaped letter; the zero-width non-joiner that holds ``می``
onto ``روم`` is invisible to a reader and a word boundary to a tokeniser; and digits
arrive in two alphabets. Scoring accuracy without folding those first produces a large
number that says nothing about recognition quality -- and larger than the difference
between the two engines anyone was trying to compare.

The rules are split into two groups on purpose, because they carry opposite risks.

**Applied always.** Diacritics, the decorative kashida, and the invisible direction marks
carry no meaning in this text. Digits are folded to one alphabet. And the Arabic letters
that are *the same letter* as their Persian counterparts -- ``ي``/``ی`` and ``ك``/``ک``
-- are folded, because that is the same word typed on a different keyboard.

**Off by default, because they can hide a real mistake.** ``آ`` is not ``ا`` in Persian,
and ``ؤ`` is not ``و``. Folding them makes two genuinely different spellings compare equal,
which is the opposite of the point. Turn them on when the question is "did it hear the
right word", not "did it spell it correctly".
"""

from __future__ import annotations

# The same letter, typed on a different keyboard. These are not stylistic choices: the
# Arabic and Persian codepoints render identically and mean the same thing, so treating
# them as different would report a spelling error where there is none.
SAME_LETTER = {
    "\u064a": "\u06cc",  # ARABIC YEH         -> PERSIAN YEH
    "\u0649": "\u06cc",  # ALEF MAKSURA       -> PERSIAN YEH
    "\u0643": "\u06a9",  # ARABIC KAF         -> PERSIAN KEH
    "\u0629": "\u0647",  # TEH MARBUTA        -> HEH
    "\u06c0": "\u0647",  # HEH WITH YEH ABOVE -> HEH
}

# Genuinely distinct letters that a careless fold would merge. Opt in to merge them.
DISTINCT_LETTERS = {
    "\u0622": "\u0627",  # ALEF WITH MADDA ABOVE -> ALEF
    "\u0623": "\u0627",  # ALEF WITH HAMZA ABOVE -> ALEF
    "\u0625": "\u0627",  # ALEF WITH HAMZA BELOW -> ALEF
    "\u0624": "\u0648",  # WAW WITH HAMZA        -> WAW
    "\u0626": "\u06cc",  # YEH WITH HAMZA        -> YEH
}

# Marks that never carry meaning in recognised speech: kashida is decoration, and the
# direction marks are invisible layout hints that some tools leave behind.
REMOVED_ALWAYS = (
    "\u0640",  # KASHIDA
    "\u200b",  # ZERO WIDTH SPACE
    "\u200e",  # LEFT-TO-RIGHT MARK
    "\u200f",  # RIGHT-TO-LEFT MARK
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",  # embedding overrides
    "\ufeff",  # BYTE ORDER MARK
    "\u200d",  # ZERO WIDTH JOINER
)

# Harakat and the superscript alef. Whisper does not usually emit them, but a Persian
# fine-tune or a hand-edited transcript might, and they change no word.
DIACRITICS = (
    "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0653\u0654\u0655\u0670"
)

ZWNJ = "\u200c"

# Both digit alphabets, folded to ASCII so ۱۲۳ and 123 count as the same number.
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"

# Replaced with a space rather than deleted, so ``سلام،خوبی`` becomes two words instead of
# one word that matches neither side.
PUNCTUATION = "،؛؟«»…٬٫\"'.,!?:;()[]{}<>|~`^*+=$%&@#\\/—_-"

_DIGIT_FOLD = str.maketrans(
    {**dict(zip(PERSIAN_DIGITS, "0123456789")), **dict(zip(ARABIC_DIGITS, "0123456789"))}
)
_PUNCTUATION_FOLD = str.maketrans({char: " " for char in PUNCTUATION})


def fold(
    text: object,
    *,
    zwnj: bool = True,
    distinct_letters: bool = False,
    punctuation: bool = True,
) -> str:
    """Normalise Persian text for comparison.

    ``zwnj`` removes the zero-width non-joiner, which makes ``می‌روم`` and ``میروم``
    one word. That is right when comparing what was *said*, and wrong if the question is
    how the writer chose to spell it, so it is a flag rather than a rule.
    """
    # A missing text folds to nothing. ``str(None)`` would produce the word "None", which
    # is a word no transcript contains.
    result = "" if text is None else str(text)
    for source, target in SAME_LETTER.items():
        result = result.replace(source, target)
    if distinct_letters:
        for source, target in DISTINCT_LETTERS.items():
            result = result.replace(source, target)
    for character in REMOVED_ALWAYS:
        result = result.replace(character, "")
    for character in DIACRITICS:
        result = result.replace(character, "")
    if zwnj:
        result = result.replace(ZWNJ, "")
    result = result.translate(_DIGIT_FOLD)
    if punctuation:
        result = result.translate(_PUNCTUATION_FOLD)
    # Whitespace last, so every fold above that produced a space is collapsed too.
    return " ".join(result.split())


def words(text: object, **options) -> list[str]:
    """The words of a folded string, which is the unit errors are counted in."""
    return fold(text, **options).split()


def variants(text: object, **options) -> set[str]:
    """Every folding of this text, for a quick membership check.

    Useful for term spotting, where a term may be written joined or not, or with either
    yeh, and any of those should count as a hit.
    """
    return {
        fold(text, **options),
        fold(text, zwnj=False, **options),
        fold(text, distinct_letters=True, **options),
    }
