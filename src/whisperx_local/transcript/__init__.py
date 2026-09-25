"""Speaker-aware transcript assembly and rendering.

WhisperX computes accurate word-level speaker data and then throws it away in its own
text writer, labelling each segment with whichever speaker dominated that span. The
whole point of this package is to keep that detail:

- ``turns``     the speaker model, cleaning raw diarization output, and labelling a span
- ``assemble``  grouping words into lines and absorbing diarization flicker
- ``render``    writing the result out in each supported format

``turns.diarize_turns`` imports whisperx lazily, so importing this package costs
nothing and does not require the models to be present.
"""

from whisperx_local.transcript.assemble import (
    build_lines,
    prepare_lines,
    rename_speakers,
    smooth_lines,
)
from whisperx_local.transcript.render import (
    FORMAT_ORDER,
    RENDERERS,
    render_aud,
    render_json,
    render_srt,
    render_tsv,
    render_txt,
    render_vtt,
    write_formats,
)
from whisperx_local.transcript.turns import (
    DIARIZATION_MODEL,
    Line,
    Turn,
    clean_turns,
    diarize_turns,
    drop_micro_turns,
    merge_adjacent,
    speaker_for_span,
)

__all__ = [
    "DIARIZATION_MODEL",
    "FORMAT_ORDER",
    "RENDERERS",
    "Line",
    "Turn",
    "build_lines",
    "clean_turns",
    "diarize_turns",
    "drop_micro_turns",
    "merge_adjacent",
    "prepare_lines",
    "rename_speakers",
    "render_aud",
    "render_json",
    "render_srt",
    "render_tsv",
    "render_txt",
    "render_vtt",
    "smooth_lines",
    "speaker_for_span",
    "write_formats",
]
