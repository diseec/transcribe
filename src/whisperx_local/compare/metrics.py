"""Counting how different two texts are, and in what way.

Accuracy is a number or it is an opinion. What follows is the standard measure, with one
addition that matters for choosing between engines: the split between substitutions,
insertions and deletions is kept, because the two failure modes are not equivalent. An
engine that inserts words is inventing speech; one that deletes them is missing it. A
single error rate hides which of those is happening.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Beyond this many cell comparisons a full alignment is refused rather than attempted.
# The scoring path sums per-bin results, so a whole transcript never reaches here; this
# guard is for a caller who passes two very long texts directly.
MAX_ALIGNMENT_CELLS = 4_000_000


@dataclass(frozen=True)
class Edits:
    """What it takes to turn the hypothesis into the reference.

    ``unattributed`` exists for the case where the total is known but its composition is
    not. Splitting errors into substitutions, insertions and deletions needs the alignment
    path, and the path needs a table proportional to both lengths; for two transcripts of
    a whole meeting that is tens of millions of cells. Rather than guess the split, or
    refuse to give a total, the total is given and the split is marked unknown.
    """

    reference_length: int
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    unattributed: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions + self.unattributed

    @property
    def split_known(self) -> bool:
        return self.unattributed == 0

    @property
    def correct(self) -> int:
        return max(0, self.reference_length - self.substitutions - self.deletions)

    @property
    def rate(self) -> float | None:
        """Errors per reference unit, or ``None`` when there is nothing to be wrong about."""
        if not self.reference_length:
            return None
        return self.errors / self.reference_length

    @property
    def accuracy(self) -> float | None:
        rate = self.rate
        return None if rate is None else max(0.0, 1.0 - rate)

    def merge(self, other: "Edits") -> "Edits":
        return Edits(
            reference_length=self.reference_length + other.reference_length,
            substitutions=self.substitutions + other.substitutions,
            insertions=self.insertions + other.insertions,
            deletions=self.deletions + other.deletions,
            unattributed=self.unattributed + other.unattributed,
        )

    def describe(self) -> str:
        if self.rate is None:
            return "no reference text to score against"
        if not self.split_known:
            return (
                f"{self.rate * 100:.1f}%  ({self.errors} errors; separating substituted "
                "from inserted and deleted needs word timings)"
            )
        return (
            f"{self.rate * 100:.1f}%  ({self.substitutions} substituted, "
            f"{self.insertions} inserted, {self.deletions} deleted)"
        )


def distance(reference: Sequence, hypothesis: Sequence) -> int:
    """Levenshtein distance, in two rows, so long inputs do not need a huge matrix."""
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for row, item in enumerate(reference, start=1):
        current = [row]
        for column, other in enumerate(hypothesis, start=1):
            cost = 0 if item == other else 1
            current.append(
                min(
                    previous[column] + 1,        # deletion
                    current[column - 1] + 1,     # insertion
                    previous[column - 1] + cost,  # substitution or match
                )
            )
        previous = current
    return previous[-1]


def total_only(reference: Sequence, hypothesis: Sequence) -> Edits:
    """The number of errors, without splitting them by kind.

    For pairs too large to align with a path. Levenshtein distance is enough for the
    total and needs only two rows of memory, so a whole meeting can be scored without
    allocating a table the size of its word count squared.
    """
    reference = list(reference)
    hypothesis = list(hypothesis)
    return Edits(
        reference_length=len(reference),
        unattributed=distance(reference, hypothesis),
    )


def align(reference: Sequence, hypothesis: Sequence) -> Edits:
    """Score one hypothesis against one reference, with the error types split out.

    Intended for the short spans a comparison works in. A long pair is refused rather
    than allowed to consume memory quietly: the caller can score it in pieces, which is
    what ``transcripts.compare`` does.
    """
    reference = list(reference)
    hypothesis = list(hypothesis)
    if len(reference) * len(hypothesis) > MAX_ALIGNMENT_CELLS:
        raise ValueError(
            "too long to align in one piece; score it in shorter spans"
        )
    rows, columns = len(reference), len(hypothesis)
    # cost[i][j] is the distance between the first i reference items and first j hypothesis
    # items. Kept in full because the counts need the path, not just the total.
    cost = [[0] * (columns + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        cost[i][0] = i
    for j in range(columns + 1):
        cost[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            same = reference[i - 1] == hypothesis[j - 1]
            cost[i][j] = min(
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
                cost[i - 1][j - 1] + (0 if same else 1),
            )

    substitutions = insertions = deletions = 0
    i, j = rows, columns
    while i > 0 or j > 0:
        if i and j:
            same = reference[i - 1] == hypothesis[j - 1]
            step = 0 if same else 1
            if cost[i][j] == cost[i - 1][j - 1] + step:
                if not same:
                    substitutions += 1
                i, j = i - 1, j - 1
                continue
        if i and cost[i][j] == cost[i - 1][j] + 1:
            deletions += 1
            i -= 1
            continue
        insertions += 1
        j -= 1

    return Edits(
        reference_length=rows,
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
    )
