"""Reading a transcript, and comparing two of them by time rather than by segment.

The alignment question decides whether a comparison means anything at all. Two engines
given the same audio cut it into different pieces: different segment counts, different
boundaries, sometimes one sentence where the other had two. Comparing segment one with
segment one therefore compares unrelated speech, and the error rate that comes out is a
measure of segmentation rather than of recognition.

So both transcripts are projected onto a shared clock before anything is scored: whatever
was said inside each fixed window is gathered up, and windows are compared with each
other. Scoring per window also keeps the work small, which is why a whole recording can be
measured without building a table the size of its vocabulary.

The price of that choice is real and stated plainly: a word falling on a window boundary
in one transcript and just past it in the other is scored as a deletion plus an insertion.
That is why the window is 30 seconds rather than something smaller, and why a disagreement
right at a boundary deserves a listen before it is believed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from whisperx_local.compare.metrics import Edits, align, total_only
from whisperx_local.compare.normalize import fold, words

# Wide enough that boundary effects stay small, short enough to localise a problem.
DEFAULT_WINDOW = 30.0

# Terms this domain gets wrong, in both scripts, because a meeting about subscriptions
# says the word in English and the transcript may hold it either way. Not a general
# vocabulary: it is the list that made someone read the output twice.
DEFAULT_TERMS = (
    "subscription",
    "سابسکریپشن",
    "database",
    "دیتابیس",
    "api",
    "ui",
    "pro",
    "پرو",
)

# Marked on utterances that were read from a file with no timing information at all, so a
# reader can tell "one span covering everything" from "a segment that really starts at 0".
UNTIMED_END = float("inf")

# Two timestamp forms, because both exist in the wild. This app writes HH:MM:SS.mmm; the
# earlier one wrote bare seconds. A transcript in the other form is still a transcript,
# and failing to recognise it is not a harmless miss: the file then reads as untimed, its
# ``#`` header comments are counted as speech, and the comparison silently becomes a
# whole-text one instead of saying where the two differ.
_TIMED_CLOCK = re.compile(
    r"^\[(\d+):(\d{2}):(\d{2}(?:\.\d+)?)\s*-->\s*"
    r"(\d+):(\d{2}):(\d{2}(?:\.\d+)?)\]\s*(.*)$"
)
_TIMED_SECONDS = re.compile(
    r"^\[(\d+(?:\.\d+)?)\s*-->\s*(\d+(?:\.\d+)?)\]\s*(.*)$"
)


@dataclass(frozen=True)
class Utterance:
    """One span of speech, as some engine chose to cut it."""

    start: float
    end: float
    text: str
    logprob: float | None = None


@dataclass(frozen=True)
class Transcript:
    """A transcript read from disk, with whether it carries any timing at all."""

    label: str
    utterances: list[Utterance] = field(default_factory=list)
    timed: bool = True

    @property
    def text(self) -> str:
        return " ".join(utterance.text for utterance in self.utterances)

    @property
    def words(self) -> int:
        return len(words(self.text))

    @property
    def end(self) -> float:
        timed = [
            utterance.end
            for utterance in self.utterances
            if utterance.end != UNTIMED_END
        ]
        return max(timed) if timed else 0.0

    def mean_logprob(self) -> float | None:
        values = [
            utterance.logprob
            for utterance in self.utterances
            if utterance.logprob is not None
        ]
        return sum(values) / len(values) if values else None


@dataclass(frozen=True)
class BinResult:
    """One window of the clock, and how the two transcripts differ inside it."""

    index: int
    start: float
    end: float
    reference: str
    other: str
    edits: Edits

    @property
    def rate(self) -> float | None:
        return self.edits.rate

    @property
    def identical(self) -> bool:
        return self.edits.errors == 0

    def describe(self) -> str:
        rate = self.rate
        shown = "—" if rate is None else f"{rate * 100:.0f}%"
        return f"{self.start:.0f}s–{self.end:.0f}s  {shown}"


@dataclass(frozen=True)
class TermHit:
    term: str
    reference: int
    other: int

    @property
    def missing(self) -> int:
        return max(0, self.reference - self.other)

    def describe(self) -> str:
        return f"{self.term} {self.other}/{self.reference}"


@dataclass
class Comparison:
    """Two transcripts measured against each other, ready to be reported."""

    reference: Transcript
    other: Transcript
    window: float
    bins: list[BinResult] = field(default_factory=list)
    terms: list[TermHit] = field(default_factory=list)
    windowed: bool = True

    @property
    def comparable_by_time(self) -> bool:
        """Whether both sides carry timing, and so can be compared window by window.

        Without it the comparison is of the whole text in one piece. That is a coarser
        question -- it cannot say *where* the two differ -- but it is the only honest one
        when a side has no timestamps, and it is the shape a plain ``.txt`` transcript has.
        """
        return self.reference.timed and self.other.timed

    # -- totals -----------------------------------------------------------

    @property
    def totals(self) -> Edits:
        """Errors summed over the windows, which is also how a whole WER is obtained."""
        total = Edits(reference_length=0)
        for entry in self.bins:
            total = total.merge(entry.edits)
        return total

    @property
    def reference_words(self) -> int:
        return len(words(self.reference.text))

    @property
    def other_words(self) -> int:
        return len(words(self.other.text))

    @property
    def agreement(self) -> float:
        """Share of windows where the two transcripts say exactly the same thing."""
        if not self.bins:
            return 1.0
        return sum(1 for entry in self.bins if entry.identical) / len(self.bins)

    @property
    def compared(self) -> bool:
        """Whether the two have enough in common to be worth a number at all."""
        return self.reference_words > 0 and self.other_words > 0

    def worst(self, limit: int = 5) -> list[BinResult]:
        """The windows carrying the most errors, worst first."""
        ranked = [entry for entry in self.bins if entry.edits.errors]
        ranked.sort(key=lambda entry: entry.edits.errors, reverse=True)
        return ranked[:limit]

    # -- reporting --------------------------------------------------------

    def rows(self) -> list[tuple[str, str]]:
        """Summary rows for a panel, most decision-relevant first."""
        total = self.totals
        rows: list[tuple[str, str]] = [
            ("Reference", f"{self.reference.label}  ·  {self.reference_words} words"),
            ("Compared", f"{self.other.label}  ·  {self.other_words} words"),
        ]
        if not self.compared:
            rows.append(("Result", "not comparable: one side has no recognised text"))
            return rows
        rows.append(
            ("Errors", f"{total.errors} of {total.reference_length} words  ·  {total.rate * 100:.1f}%")
        )
        if total.split_known:
            rows.append(
                (
                    "Breakdown",
                    f"{total.substitutions} substituted, "
                    f"{total.insertions} inserted, {total.deletions} deleted",
                )
            )
        else:
            # Said out loud rather than omitted: an absent row reads as a zero.
            rows.append(
                ("Breakdown", "unavailable: one side has no timings, so errors cannot be "
                              "attributed to a position")
            )
        if self.windowed:
            rows.append(
                ("Agreement", f"{self.agreement * 100:.0f}% of {len(self.bins)} windows identical")
            )
        else:
            rows.append(("Agreement", "scored as one span: no timings to place the differences"))
        if self.reference.timed and self.other.timed:
            rows.append(
                ("Covered", f"{self.reference.end:.0f}s vs {self.other.end:.0f}s")
            )
        else:
            rows.append(("Covered", "no timings in one side; scored as a single span"))
        confidence = self._confidence()
        if confidence:
            rows.append(("Confidence", confidence))
        if self.terms:
            rows.append(
                ("Terms", "  ·  ".join(hit.describe() for hit in self.terms if hit.reference or hit.other))
            )
        return rows

    def _confidence(self) -> str:
        reference = self.reference.mean_logprob()
        other = self.other.mean_logprob()
        if reference is None and other is None:
            return ""
        left = "n/a" if reference is None else f"{reference:.2f}"
        right = "n/a" if other is None else f"{other:.2f}"
        return f"mean logprob {left} vs {right}  (higher is more confident)"

    def worst_lines(self, limit: int = 5) -> list[str]:
        """The worst windows, with enough text to judge whether the errors are real."""
        lines: list[str] = []
        for entry in self.worst(limit):
            lines.append(entry.describe())
            lines.append(f"    reference  {entry.reference[:160] or '(nothing)'}")
            lines.append(f"    other      {entry.other[:160] or '(nothing)'}")
        return lines


def _seconds(hours: str, minutes: str, seconds: str) -> float:
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def read(path: str | Path) -> Transcript:
    """Read a transcript, preferring timing where the file has any.

    Accepts the canonical ``.raw.json``, a bare ``{"segments": [...]}``, or the
    ``.raw.txt`` timestamped text. A file with no timings at all is still read, as one
    span covering everything: comparing two such files gives a whole-text error rate,
    which is coarse but is better than refusing to answer.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"no transcript at {source}")
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        # Returned directly rather than falling through: a JSON transcript with no usable
        # segments is empty, and the untimed fallback below would otherwise read the JSON
        # *source* as though it were recognised speech.
        return Transcript(label=source.name, utterances=_read_json(text), timed=True)
    utterances = _read_timed_lines(text)
    if utterances:
        return Transcript(label=source.name, utterances=utterances, timed=True)
    if text.strip():
        return Transcript(
            label=source.name,
            utterances=[Utterance(start=0.0, end=UNTIMED_END, text=text)],
            timed=False,
        )
    return Transcript(label=source.name, utterances=[], timed=True)


def _read_json(text: str) -> list[Utterance]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        segments = payload.get("segments") or []
    elif isinstance(payload, list):
        segments = payload
    else:
        return []
    utterances: list[Utterance] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        body = str(segment.get("text", "")).strip()
        if not body or "start" not in segment:
            continue
        logprob = segment.get("avg_logprob")
        utterances.append(
            Utterance(
                start=float(segment["start"]),
                end=float(segment.get("end", segment["start"])),
                text=body,
                logprob=float(logprob) if isinstance(logprob, (int, float)) else None,
            )
        )
    return utterances


def _read_timed_lines(text: str) -> list[Utterance]:
    """Timed lines in either supported form, ignoring anything that is neither."""
    utterances: list[Utterance] = []
    for line in text.splitlines():
        stripped = line.strip()
        clock = _TIMED_CLOCK.match(stripped)
        if clock:
            start = _seconds(*clock.groups()[:3])
            end = _seconds(*clock.groups()[3:6])
            body = clock.group(7).strip()
        else:
            seconds = _TIMED_SECONDS.match(stripped)
            if not seconds:
                # Header comments and blank lines land here, which is what keeps a
                # recovery file's preamble out of the recognised text.
                continue
            start, end = float(seconds.group(1)), float(seconds.group(2))
            body = seconds.group(3).strip()
        if body:
            utterances.append(Utterance(start=start, end=end, text=body))
    return utterances


def bin_words(
    utterances: list[Utterance], window: float, **fold_options
) -> dict[int, list[str]]:
    """Words per window, each utterance placed by its middle.

    The middle rather than the start, because an utterance sits where it mostly is. A
    segment is never split across windows: that would invent a boundary inside a phrase
    and score the two halves separately.

    The fold options are threaded through rather than defaulted, so the words counted here
    are the same words the totals are divided by. Skipping that made the error count and
    the word count disagree as soon as a caller folded anything unusual.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    gathered: dict[int, list[str]] = {}
    for utterance in utterances:
        if utterance.end == UNTIMED_END:
            index = 0
        else:
            middle = (utterance.start + utterance.end) / 2.0
            index = int(middle // window)
        gathered.setdefault(index, []).extend(words(utterance.text, **fold_options))
    return gathered


def _count_sequence(haystack: list[str], needle: list[str]) -> int:
    """Occurrences of one word sequence inside another, counted without overlapping."""
    if not needle or len(needle) > len(haystack):
        return 0
    total = 0
    index = 0
    while index <= len(haystack) - len(needle):
        if haystack[index : index + len(needle)] == needle:
            total += 1
            index += len(needle)
        else:
            index += 1
    return total


def compare(
    reference: Transcript | str | Path,
    other: Transcript | str | Path,
    *,
    window: float = DEFAULT_WINDOW,
    terms: tuple[str, ...] = DEFAULT_TERMS,
    **fold_options,
) -> Comparison:
    """Score ``other`` against ``reference``, window by window."""
    if not isinstance(reference, Transcript):
        reference = read(reference)
    if not isinstance(other, Transcript):
        other = read(other)

    reference_words = words(reference.text, **fold_options)
    other_words = words(other.text, **fold_options)

    if not (reference.timed and other.timed):
        # One side has no clock, so there is nothing to line the two up against. Comparing
        # the texts whole is the only meaningful question left, and it is the right one for
        # "how much is missing". Scoring it window by window would pile every word of the
        # untimed side into the first window and measure the pile, not the difference.
        return Comparison(
            reference=reference,
            other=other,
            window=window,
            bins=[
                BinResult(
                    index=0,
                    start=0.0,
                    end=max(reference.end, other.end),
                    reference=" ".join(reference_words),
                    other=" ".join(other_words),
                    edits=total_only(reference_words, other_words),
                )
            ],
            terms=_terms(reference_words, other_words, terms, fold_options),
            windowed=False,
        )

    left = bin_words(reference.utterances, window, **fold_options)
    right = bin_words(other.utterances, window, **fold_options)
    bins: list[BinResult] = []
    for index in sorted(set(left) | set(right)):
        reference_text = " ".join(left.get(index, []))
        other_text = " ".join(right.get(index, []))
        bins.append(
            BinResult(
                index=index,
                start=index * window,
                end=(index + 1) * window,
                reference=reference_text,
                other=other_text,
                edits=align(reference_text.split(), other_text.split()),
            )
        )

    return Comparison(
        reference=reference,
        other=other,
        window=window,
        bins=bins,
        terms=_terms(reference_words, other_words, terms, fold_options),
        windowed=True,
    )


def _terms(
    reference_words: list[str],
    other_words: list[str],
    terms: tuple[str, ...],
    fold_options: dict,
) -> list[TermHit]:
    """How often each tracked term appears on either side."""
    hits: list[TermHit] = []
    for term in terms:
        needle = words(term, **fold_options)
        if not needle:
            continue
        hits.append(
            TermHit(
                term=term,
                reference=_count_sequence(reference_words, needle),
                other=_count_sequence(other_words, needle),
            )
        )
    return hits


def fold_all(text: object, **options) -> str:
    """Convenience re-export so a caller needs one import to normalise text."""
    return fold(text, **options)
