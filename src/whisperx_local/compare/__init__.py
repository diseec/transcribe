"""Comparing two transcripts of the same recording, to decide which is better.

Built to answer one question -- is engine B worse than engine A, and where -- which needs
three things that are individually easy to get wrong:

* ``normalize``    Persian spelling folded, so a difference of keyboard or of ZWNJ does
                   not masquerade as a difference of recognition.
* ``metrics``      errors counted rather than eyeballed, with the split between
                   substitutions, insertions and deletions kept, because a run that
                   invents words fails differently from one that misses them.
* ``transcripts``  the two transcripts aligned by *time*, never by segment, because two
                   engines cut the same audio into different pieces and comparing segment
                   one with segment one compares unrelated speech.

Nothing here writes to a recording's workspace or reads its cache. Comparing is a
question about two files, and it should be answerable without touching either.
"""

from whisperx_local.compare.metrics import Edits, distance
from whisperx_local.compare.normalize import fold, words
from whisperx_local.compare.transcripts import (
    BinResult,
    Comparison,
    Utterance,
    compare,
    read,
)

__all__ = [
    "BinResult",
    "Comparison",
    "Edits",
    "Utterance",
    "compare",
    "distance",
    "fold",
    "read",
    "words",
]
