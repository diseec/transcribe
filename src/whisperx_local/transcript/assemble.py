"""Turning recognised words into speaker turns.

WhisperX's own text writer prints one line per Whisper segment and labels it with the
speaker dominating that whole span. On real conversations a segment usually contains
more than one voice, so that label drifts to whoever spoke most. Here the word-level
timings are kept, a speaker is assigned to every word, and only then are consecutive
same-speaker words grouped into readable turns.
"""

from __future__ import annotations

# A line breaks when the speaker changes, when a pause exceeds this, or when it would
# grow past the character budget. The budget keeps subtitle lines readable.
DEFAULT_MAX_GAP = 1.1
DEFAULT_MAX_CHARS = 110

# Flicker absorption, as two rules with two different limits because the two carry very
# different risks.
#
# A fragment that repeats the speaker before it costs nothing to absorb: the words keep
# their label and only their line changes.
FLICKER_MAX_WORDS = 2
FLICKER_MAX_DURATION = 1.0
# A fragment labelled as somebody other than the speakers on either side of it is a
# different matter. A real one-word reply looks identical -- ``بله``, ``آره`` and ``خب``
# all last well under a second -- so this rule is confined to fragments too short to be
# anyone's actual contribution. Erring wide here does not lose text, but it does hand one
# speaker's words to another, which is worse than leaving a label alone.
REPLY_MAX_DURATION = 0.5
FLICKER_JOIN_GAP = 1.1


def _segment_words(segment: dict) -> list[dict]:
    """The words of one segment, falling back to the segment itself as one word.

    WhisperX omits ``words`` when alignment was skipped, and a segment with no usable
    timing still carries text that must not be dropped.
    """
    words = [
        word
        for word in segment.get("words", []) or []
        if str(word.get("word", "")).strip() and "start" in word and "end" in word
    ]
    if words:
        return words
    text = str(segment.get("text", "")).strip()
    if not text:
        return []
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start))
    return [{"word": text, "start": start, "end": end}]


def build_lines(
    segments: list[dict],
    turns: list,
    *,
    max_gap: float = DEFAULT_MAX_GAP,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list:
    """Group words into speaker turns using word-level speaker labels."""
    from whisperx_local.transcript.turns import Line, speaker_for_span

    lines: list[Line] = []
    previous_speaker: str | None = None
    for segment in segments:
        for word in _segment_words(segment):
            start = float(word["start"])
            end = float(word.get("end", start))
            speaker = speaker_for_span(start, end, turns, fallback=previous_speaker)
            previous_speaker = speaker
            if lines:
                current = lines[-1]
                projected = len(current.text) + 1 + len(str(word["word"]).strip())
                if (
                    current.speaker == speaker
                    and start - current.end <= max_gap
                    and projected <= max_chars
                ):
                    current.words.append(word)
                    current.end = max(current.end, end)
                    continue
            lines.append(Line(speaker=speaker, start=start, end=end, words=[word]))
    return lines


def smooth_lines(
    lines: list,
    *,
    max_words: int = FLICKER_MAX_WORDS,
    max_duration: float = FLICKER_MAX_DURATION,
    reply_duration: float = REPLY_MAX_DURATION,
    gap: float = FLICKER_JOIN_GAP,
) -> list:
    """Fold flicker turns into their neighbours.

    Diarization sometimes flips speaker around a real change. A fragment is folded
    back when it simply repeats the previous speaker, or when it is so brief that it
    cannot be a contribution of its own between two turns by one other speaker.
    Genuine multi-word turns are never touched.

    Non-mutating: new ``Line`` objects are built, so a caller's input is never
    altered behind its back.
    """
    from whisperx_local.transcript.turns import Line

    if len(lines) < 2:
        return list(lines)
    result: list[Line] = []
    reply_limit = min(max_duration, reply_duration)
    for index, line in enumerate(lines):
        tiny = len(line.words) <= max_words and (line.end - line.start) <= max_duration
        previous = result[-1] if result else None
        following = lines[index + 1] if index + 1 < len(lines) else None
        surrounded = (
            previous is not None
            and following is not None
            and previous.speaker == following.speaker
            and previous.speaker != line.speaker
        )
        absorbable = tiny and (
            (previous is not None and previous.speaker == line.speaker)
            or (surrounded and (line.end - line.start) <= reply_limit)
        )
        if previous is not None and absorbable:
            result[-1] = Line(
                speaker=previous.speaker,
                start=previous.start,
                end=max(previous.end, line.end),
                words=previous.words + line.words,
            )
            continue
        if (
            previous is not None
            and previous.speaker == line.speaker
            and line.start - previous.end <= gap
        ):
            # Adjacent same-speaker lines still have to be rejoined, or a flicker
            # that was absorbed leaves two lines with the same label.
            result[-1] = Line(
                speaker=previous.speaker,
                start=previous.start,
                end=max(previous.end, line.end),
                words=previous.words + line.words,
            )
            continue
        result.append(line)
    return result


def rename_speakers(lines: list) -> list:
    """Turn ``SPEAKER_00`` into appearance-ordered names.

    Ordered by first appearance rather than by label, so "Speaker 1" is whoever spoke
    first instead of whoever pyannote happened to number first.
    """
    mapping: dict[str, str] = {}
    for line in lines:
        if line.speaker is not None and line.speaker not in mapping:
            mapping[line.speaker] = f"Speaker {len(mapping) + 1}"
    if mapping:
        for line in lines:
            if line.speaker is not None:
                line.speaker = mapping[line.speaker]
    return lines


def from_segments(segments: list[dict]) -> list:
    """One line per recognised segment, cut exactly where recognition cut it.

    Used when there are no speaker turns to justify regrouping. Regrouping is a
    presentation choice, and every regrouping risks looking like it deleted something:
    a tidy file that merged or dropped fragments is indistinguishable from a tidy file
    that lost content. So with nothing to gain, nothing is re-cut.
    """
    from whisperx_local.transcript.turns import Line

    lines: list[Line] = []
    for segment in segments:
        words = _segment_words(segment)
        if not words:
            continue
        lines.append(
            Line(
                speaker=None,
                start=float(words[0]["start"]),
                end=float(words[-1].get("end", words[-1]["start"])),
                words=list(words),
            )
        )
    return lines


def prepare_lines(segments: list[dict], turns: list) -> list:
    """The assembly path, from raw segments and turns to finished lines.

    With speaker turns, words are regrouped per speaker and flicker is absorbed.
    Without them there is nothing to regroup for, and the recognition segmentation is
    passed through untouched.
    """
    from whisperx_local.transcript.turns import clean_turns

    cleaned = clean_turns(turns)
    if not cleaned:
        return from_segments(segments)
    return rename_speakers(smooth_lines(build_lines(segments, cleaned)))
