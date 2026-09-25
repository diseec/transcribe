"""Executing a plan.

One function per action, driven by the plan the planner produced. The order comes from
the plan rather than from this file, so a run that only needs to re-render subtitles
never reaches the transcription code at all.

Three guarantees are enforced here rather than hoped for:

- **A finished chunk is never recomputed.** Recognised text and alignment are stored
  per chunk, and either is reused when present.
- **The transcript is written before speakers are attempted.** Speaker separation is
  the slowest and most fragile stage, so a failure there costs speaker labels, never
  the text already paid for.
- **A plan can always reproduce its own chunks.** The boundaries are stored alongside
  the signature, because a run that starts mid-plan has to re-derive the same slicing,
  and a hash of the boundaries cannot be inverted.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from whisperx_local.actions.plan import Plan
from whisperx_local.chunking import (
    Chunk,
    atomic_write_json,
    chunk_paths,
    distribute_segments,
    group_chunks,
    is_contiguous,
    plan_chunks,
    plan_signature,
    quiet_cut_points,
)
from whisperx_local.engine import (
    align_chunk,
    diarize_voices,
    read_turns,
    transcribe_chunk,
    write_turns,
)
from whisperx_local.engine.commands import recognition_signature
from whisperx_local.media import (
    detect_silences,
    energy_windows,
    mean_volume_db,
    recognition_audio,
    recognition_path,
    silence_cut_points,
    slice_audio,
    speaker_audio,
)
from whisperx_local.services.artifacts import WorkSpace
from whisperx_local.services.jobs import (
    mark_complete,
    mark_core_ready,
    mark_diarization_ready,
)
from whisperx_local.services.report import Gap, RunReport, union_seconds
from whisperx_local.transcript import RENDERERS, clean_turns, write_formats
from whisperx_local.transcript.canonical import coverage_seconds, write_canonical

# Boundaries are stored at this precision so a re-read plan matches the one written.
BOUNDARY_PRECISION = 3


@dataclass
class Outcome:
    """What a run produced, including anything that went wrong without stopping it."""

    files: list[Path] = field(default_factory=list)
    words: int = 0
    speakers: int = 0
    chunks: int = 0
    missing: list[Chunk] = field(default_factory=list)
    unaligned: list[Chunk] = field(default_factory=list)
    silent: list[Chunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    report: str | None = None
    summary: RunReport | None = None
    canonical: list[Path] = field(default_factory=list)

    @property
    def incomplete(self) -> bool:
        """True when this run must not be presented as a finished transcript.

        ``silent`` is deliberately excluded. A stretch with no speech in it has no
        text to be missing, and recognition returns an empty result for silence quite
        legitimately. Whether that silence was expected is answered by the coverage
        figure in the report instead.
        """
        return bool(self.missing) or bool(self.unaligned)


@dataclass
class _Context:
    """Shared state handed to each action handler."""

    options: object
    workspace: WorkSpace
    studio: object
    ffmpeg: str
    env: dict
    duration: float
    network: str
    outcome: Outcome
    chunks: list[Chunk] = field(default_factory=list)
    cut_points: list[float] = field(default_factory=list)
    segments: list[dict] = field(default_factory=list)
    turns: list = field(default_factory=list)
    requested: set[str] = field(default_factory=set)

    @property
    def source(self) -> Path:
        return self.workspace.source

    def chunk_audio(self) -> Path:
        """The audio chunks are cut from: the levelled track, or the original when
        that stage was not part of this run."""
        derived = recognition_path(self.source)
        return derived if derived.is_file() else self.source


def describe_plan(plan: Plan) -> str:
    """A one-line summary of what will run, for the plan panel."""
    if not plan:
        return "nothing to do"
    text = " → ".join(action.label for action in plan.actions)
    if plan.added:
        text += f"   (added for you: {', '.join(plan.added)})"
    return text


def execute(
    plan: Plan,
    options,
    workspace: WorkSpace,
    studio,
    *,
    ffmpeg: str,
    env: dict,
    duration: float,
    network: str,
) -> Outcome:
    """Run the planned actions against one recording."""
    context = _Context(
        options=options,
        workspace=workspace,
        studio=studio,
        ffmpeg=ffmpeg,
        env=env,
        duration=duration,
        network=network,
        outcome=Outcome(),
        requested={action.key for action in plan.actions},
    )
    for action in plan.actions:
        _HANDLERS[action.key](context)
    _record_learning(plan, context)
    return context.outcome


def _record_learning(plan: Plan, context: _Context) -> None:
    """Note what is now cached, and mark the run complete only if nothing was lost.

    Every step here is bookkeeping for later runs -- which models may be used offline,
    which recordings are finished. None of it is the work that was asked for, so none of
    it may fail the run: if a marker cannot be written the worst case is a later run
    re-doing a check, while an exception here threw away a finished transcript's report
    and turned a success into a traceback.
    """
    keys = set(plan.keys())
    options = context.options
    try:
        if "transcribe" in keys and not context.outcome.missing:
            mark_core_ready(options.model, options.language)
        if context.outcome.speakers:
            mark_diarization_ready()
        if not context.outcome.incomplete and "export" in keys:
            # A run that fell short does not leave a completion marker behind, so the
            # recording stays visibly unfinished and a rerun does not skip it.
            summary = context.outcome.summary
            if summary is None or not summary.partial:
                mark_complete(context.source, options)
    except OSError as failure:
        context.outcome.warnings.append(f"could not record cached state: {failure}")


# -- individual actions ---------------------------------------------------


def _prepare(context: _Context) -> None:
    context.studio.begin_stage("Preparing audio")
    context.studio.set_status("Building the recognition track")
    recognition_audio(
        context.source,
        context.ffmpeg,
        context.options.normalize,
        duration=context.duration,
        on_progress=context.studio.progress_to,
        log=context.studio.log_handle(),
    )
    context.studio.end_stage()


def _boundaries(context: _Context) -> None:
    """Decide the cut points, then turn them into chunks."""
    options = context.options
    if not options.silence_split:
        context.studio.skip_stage("Finding speech")
        context.chunks = plan_chunks(
            context.duration,
            chunk_seconds=options.chunk_seconds,
            overlap_seconds=options.chunk_overlap,
        )
        context.outcome.chunks = len(context.chunks)
        _store_plan(context)
        return

    context.studio.begin_stage("Finding speech")
    context.studio.set_status("Locating pauses to cut on")
    spans = detect_silences(
        context.ffmpeg,
        context.source,
        env=context.env,
        min_silence=options.silence_min,
        duration=context.duration,
        on_progress=context.studio.progress_to,
        log=context.studio.log_handle(),
    )
    cuts = silence_cut_points(spans)
    context.studio.log_line(f"silence: {len(spans)} pauses, {len(cuts)} usable cuts\n")

    # Dense recordings contain no silence at all, and a plan built from nothing
    # degrades to fixed intervals without ever saying so.
    expected = max(0, math.ceil(context.duration / options.chunk_seconds) - 1)
    if len(cuts) < expected:
        context.studio.set_status("No usable pause · measuring the calmest moments")
        samples = energy_windows(
            context.ffmpeg,
            context.source,
            env=context.env,
            duration=context.duration,
            on_progress=context.studio.progress_to,
            log=context.studio.log_handle(),
        )
        quiet = quiet_cut_points(
            context.duration, samples, chunk_seconds=options.chunk_seconds
        )
        context.studio.log_line(f"energy: {len(samples)} windows, {len(quiet)} usable cuts\n")
        if quiet:
            cuts = sorted({*cuts, *quiet})

    context.cut_points = cuts
    context.chunks = plan_chunks(
        context.duration,
        chunk_seconds=options.chunk_seconds,
        overlap_seconds=options.chunk_overlap,
        cut_points=cuts,
    )
    context.outcome.chunks = len(context.chunks)
    kind = f"snapped to {len(cuts)} boundaries" if cuts else "evenly spaced with overlap"
    context.studio.log_line(f"plan: {len(context.chunks)} chunks, {kind}\n")
    context.studio.end_stage()
    _store_plan(context)


def _store_plan(context: _Context) -> None:
    """Fingerprint the slicing and the recogniser, discarding whatever moved.

    Two fingerprints, because they invalidate different things. Moving a boundary makes
    the sliced audio itself wrong, so the whole cache goes. Changing a recognition
    setting leaves the audio perfectly good and only makes the text read from it stale,
    so the slices are kept and re-read.

    The second one was missing. ``recognition_signature`` documents itself as a cache key
    and was only ever wired into the completion marker, so switching model, precision,
    beam, language, prompt or hotwords silently reused the previous text -- which would
    have made an A/B between two engines compare a transcript against itself and report
    that nothing changed.
    """
    options = context.options
    slicing = plan_signature(
        source=context.chunk_audio(),
        duration=context.duration,
        chunk_seconds=options.chunk_seconds,
        overlap_seconds=options.chunk_overlap,
        cut_points=context.cut_points,
    )
    slicing["boundaries"] = [round(point, BOUNDARY_PRECISION) for point in context.cut_points]

    stored = _read_plan(context)
    # Only the slicing is judged here. Stale *text* is cleared before the plan is even
    # resolved, in ``WorkSpace.discard_stale_recognition``, because by the time this runs
    # the planner has already been told what was available.
    if stored is not None and _slicing_of(stored) != slicing and context.workspace.chunks.is_dir():
        context.studio.log_line("slicing changed: cached chunks discarded\n")
        shutil.rmtree(context.workspace.chunks, ignore_errors=True)

    # The recognition record is carried over, never written here, and never dropped. It
    # means one specific thing: the settings that produced the text currently on disk.
    # Writing it now would claim text this run has not produced yet, so a run interrupted
    # between slicing and recognition would leave a record vouching for the previous
    # engine's words. ``WorkSpace.remember_recognition`` is the only writer, and it runs
    # once the text exists. A plan with no record therefore means "origin unknown", which
    # ``discard_stale_recognition`` treats as untrustworthy.
    document = dict(slicing)
    if stored is not None and "recognition" in stored:
        document["recognition"] = stored["recognition"]

    context.workspace.chunks.mkdir(parents=True, exist_ok=True)
    atomic_write_json(context.workspace.plan_path, document)


def _slicing_of(stored: dict) -> dict:
    """The slicing half of a stored plan, which is compared separately from the rest."""
    return {key: value for key, value in stored.items() if key != "recognition"}


def _read_plan(context: _Context) -> dict | None:
    path = context.workspace.plan_path
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _transcribe(context: _Context) -> None:
    """Recognise every chunk, grouped so one model load covers several of them.

    Grouping exists because each recogniser invocation loads the model from scratch, which
    measured at about 25 seconds. That cost is paid per chunk otherwise, and again on every
    retry, for no benefit.

    The grouping is only ever an optimisation. A group whose audio cannot be joined, or
    whose recognition fails, is recognised one chunk at a time -- the behaviour that
    existed before grouping did -- so the worst case is the old speed.
    """
    workspace, studio = context.workspace, context.studio
    chunks = context.chunks or _chunks_from_plan(context)
    source = context.chunk_audio()
    total = len(chunks)
    studio.begin_stage("Transcribing")

    pending = {
        chunk.index
        for chunk in chunks
        if not _already_recognised(workspace.chunks, chunk)
    }
    per_call = max(1, int(getattr(context.options, "chunks_per_call", 1) or 1))
    groups = group_chunks(chunks, per_call=per_call, pending=pending)

    for position, group in enumerate(groups, start=1):
        first, last = group[0], group[-1]
        scope = (
            f"Chunk {first.index + 1}/{total}"
            if len(group) == 1
            else f"Chunks {first.index + 1}-{last.index + 1}/{total}"
        )
        studio.set_chunks(position, len(groups), first.start)
        if not _slice_group(context, group, source, scope):
            context.outcome.missing.extend(group)
            continue
        studio.set_status(f"{scope} · recognising speech")
        produced = _recognise_group(context, group, position, len(groups), scope)
        if produced is None:
            context.outcome.missing.extend(group)
            continue
        for chunk in group:
            _record_chunk(context, chunk, produced.get(chunk.index) or [])

    context.chunks = chunks
    context.outcome.chunks = total
    context.workspace.remember_recognition(recognition_signature(context.options))
    _write_canonical(context, chunks)
    studio.end_stage()


def _already_recognised(chunk_dir: Path, chunk: Chunk) -> bool:
    paths = chunk_paths(chunk_dir, chunk)
    return paths["aligned"].is_file() or paths["recognised"].is_file()


def _slice_group(context: _Context, group: list[Chunk], source: Path, scope: str) -> bool:
    """Make sure every chunk in the group has its own audio slice.

    The per-chunk slices are kept rather than replaced by the joined file, because
    alignment later reads one chunk at a time and needs its own audio to do that.
    """
    studio = context.studio
    for chunk in group:
        paths = chunk_paths(context.workspace.chunks, chunk)
        if paths["audio"].is_file():
            continue
        studio.set_status(f"{scope} · slicing audio")
        try:
            slice_audio(
                context.ffmpeg, source, chunk.start, chunk.length, paths["audio"],
                env=context.env,
            )
        except Exception as error:  # noqa: BLE001 - reported as a missing chunk, not a crash
            studio.log_line(f"chunk {chunk.index}: could not slice audio: {error}\n")
            return False
    return True


def _recognise_group(
    context: _Context, group: list[Chunk], position: int, groups: int, scope: str
) -> dict[int, list[dict]] | None:
    """Recognise a group, returning each chunk's own segments keyed by chunk index.

    Falls back to one invocation per chunk whenever the group cannot be handled, so
    joining audio can cost time but never text.
    """
    workspace, options, studio = context.workspace, context.options, context.studio
    if len(group) == 1:
        chunk = group[0]
        raw = transcribe_chunk(
            options, studio, chunk_paths(workspace.chunks, chunk)["audio"],
            workspace.chunks, chunk.index, context.network, position, groups,
        )
        return None if raw is None else {chunk.index: raw}

    joined = _joined_audio(context, group, scope)
    if joined is None:
        return _recognise_one_by_one(context, group, position, groups, scope)

    raw = transcribe_chunk(
        options, studio, joined, workspace.chunks, group[0].index,
        context.network, position, groups,
    )
    if raw is None:
        studio.log_line(
            f"group {group[0].index}: joined recognition failed; retrying one chunk at a time\n"
        )
        return _recognise_one_by_one(context, group, position, groups, scope)
    return distribute_segments(group, raw)


def _recognise_one_by_one(
    context: _Context, group: list[Chunk], position: int, groups: int, scope: str
) -> dict[int, list[dict]] | None:
    """The ungrouped path, used when joining or the joined call did not work out."""
    workspace, options, studio = context.workspace, context.options, context.studio
    produced: dict[int, list[dict]] = {}
    for chunk in group:
        studio.set_status(f"Chunk {chunk.index + 1} · recognising speech")
        raw = transcribe_chunk(
            options, studio, chunk_paths(workspace.chunks, chunk)["audio"],
            workspace.chunks, chunk.index, context.network, position, groups,
        )
        if raw is None:
            return None
        produced[chunk.index] = raw
    return produced


def _joined_audio(context: _Context, group: list[Chunk], scope: str) -> Path | None:
    """One file holding the group's audio, or ``None`` if the chunks cannot be joined.

    A stream copy, not a re-encode: the slices came from one source with one filter chain,
    so concatenating the streams gives exactly the audio the chunks hold, and joining
    costs a fraction of a second rather than a second pass over the audio.
    """
    workspace, studio = context.workspace, context.studio
    slices = [chunk_paths(workspace.chunks, chunk)["audio"] for chunk in group]
    if not all(path.is_file() for path in slices):
        return None
    for earlier, later in zip(group, group[1:]):
        if not is_contiguous(earlier, later):
            # Joining overlapping slices would duplicate speech and joining gapped ones
            # would invent silence in the middle of a word.
            studio.log_line(
                f"group {group[0].index}: not contiguous after chunk {earlier.index}\n"
            )
            return None

    target = workspace.chunks / f"group-{group[0].index:04d}.wav"
    listing = target.with_suffix(".concat")
    listing.write_text(
        "".join(f"file '{path.name}'\n" for path in slices), encoding="utf-8"
    )
    try:
        result = subprocess.run(
            [
                context.ffmpeg, "-v", "error", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(listing), "-c", "copy", str(target),
            ],
            env=context.env,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        studio.log_line(f"group {group[0].index}: could not run ffmpeg: {error}\n")
        return None
    finally:
        listing.unlink(missing_ok=True)

    if result.returncode != 0 or not target.is_file():
        studio.log_line(
            f"group {group[0].index}: joining failed ({result.returncode}): "
            f"{result.stderr.strip()[:200]}\n"
        )
        return None
    studio.log_line(f"group {group[0].index}: joined {len(slices)} slices for {scope}\n")
    return target


def _record_chunk(context: _Context, chunk: Chunk, raw: list[dict]) -> None:
    """Store one chunk's recognised text, noting the two ways it can be empty."""
    paths = chunk_paths(context.workspace.chunks, chunk)
    if not raw:
        # Recognised, but nothing found in it. Legitimate for silence, and recorded rather
        # than passed over precisely because it is the undetectable case: if it happens
        # over speech, no stage has failed anywhere, and the only trace would be a
        # transcript that quietly stops early.
        context.outcome.silent.append(chunk)
    atomic_write_json(paths["recognised"], {"segments": raw})


def _write_canonical(context: _Context, chunks: list[Chunk]) -> None:
    """Put the recognised text on disk as soon as recognition has produced it.

    Recognition is the expensive part, and the only part that cannot be recomputed, so
    its result leaves the cache immediately rather than waiting for a later stage to
    decide to write something.

    This is also why ``--only transcribe`` produces a file to read. Before, it produced
    nothing: the plan for that request contains the recognition stages and no export, so
    no output was written -- while the run still reported "Transcript complete" over a
    path that did not exist, and exited zero. Whether recognised text reaches disk must
    not depend on which unrelated stages happened to be asked for.
    """
    segments = context.workspace.load_segments(chunks)
    if not segments:
        return
    written = write_canonical(
        context.workspace.output_dir,
        context.workspace.source.stem,
        segments,
        context.options.language,
    )
    context.segments = segments
    context.outcome.canonical = list(written)
    context.outcome.files = list(dict.fromkeys([*context.outcome.files, *written]))
    context.outcome.words = sum(
        len(str(segment.get("text", "")).split()) for segment in segments
    )


def _align(context: _Context) -> None:
    """Refine word timings, keeping the text when refinement is not possible."""
    workspace, studio = context.workspace, context.studio
    chunks = context.chunks or _chunks_from_plan(context)
    total = len(chunks)
    missing = {chunk.index for chunk in context.outcome.missing}
    studio.begin_stage("Aligning words")
    for position, chunk in enumerate(chunks, start=1):
        if chunk.index in missing:
            continue
        studio.set_chunks(position, total, chunk.start)
        paths = chunk_paths(workspace.chunks, chunk)
        if not paths["recognised"].is_file() or paths["aligned"].is_file():
            continue
        aligned = align_chunk(
            context.options, studio, paths["recognised"], paths["audio"],
            workspace.chunks, chunk.index, context.network, position, total,
        )
        if aligned is None:
            context.outcome.unaligned.append(chunk)
            continue
        atomic_write_json(paths["aligned"], {"segments": aligned})
    context.chunks = chunks
    studio.end_stage()


def _diarize(context: _Context) -> None:
    workspace, studio, options = context.workspace, context.studio, context.options
    studio.begin_stage("Identifying speakers")
    cached = workspace.turns_path(options.speakers)
    if cached.is_file():
        context.turns = read_turns(cached)
        studio.set_status("Reusing the previous speaker pass")
    else:
        studio.set_status("Building a speaker-faithful copy")
        source = (
            workspace.source
            if options.diarize_audio == "original"
            else speaker_audio(
                workspace.source,
                context.ffmpeg,
                duration=context.duration,
                on_progress=studio.progress_to,
                log=studio.log_handle(),
            )
        )
        context.turns = clean_turns(
            diarize_voices(options, studio, source, workspace.root, context.network),
            min_duration=options.min_turn,
        )
        if context.turns:
            write_turns(cached, context.turns)
    if context.turns:
        context.outcome.speakers = len({turn.speaker for turn in context.turns})
        studio.set_speakers(context.outcome.speakers)
    studio.end_stage()


def _export(context: _Context) -> None:
    """Write the canonical transcript first, then the readable one built from it."""
    workspace, studio, options = context.workspace, context.studio, context.options
    segments = _collected_segments(context)
    formats = list(RENDERERS) if options.output_format == "all" else [options.output_format]

    studio.begin_stage("Writing transcript")
    studio.set_status("Saving recognised text")
    # First, and unconditionally. The recognised text is the one product here that
    # cannot be reproduced cheaply, so it reaches disk before any formatting choice,
    # speaker pass, or later stage has a chance to go wrong.
    #
    # Written again here even when recognition already wrote it, because alignment may
    # have run since and this pass carries the word timings into the canonical copy.
    context.outcome.canonical = write_canonical(
        workspace.output_dir, workspace.source.stem, segments, options.language
    )
    context.outcome.files = list(context.outcome.canonical)
    studio.mark_partial()

    if context.turns:
        studio.set_status("Applying speaker labels")
    context.outcome.files += write_formats(
        workspace.output_dir, workspace.source.stem, segments, context.turns,
        language=options.language, formats=formats,
    )
    context.outcome.words = sum(
        len(str(segment.get("text", "")).split()) for segment in segments
    )
    context.outcome.summary = _summarise(context, segments)
    studio.set_words(context.outcome.words)
    studio.end_stage()


def _summarise(context: _Context, segments: list[dict]) -> RunReport:
    """Measure the run, holes included, so a partial result cannot pass as complete."""
    outcome, requested = context.outcome, context.requested
    gaps = [
        Gap(chunk.index, chunk.start, chunk.end, "not transcribed")
        for chunk in outcome.missing
    ]
    # A chunk that recognised nothing is named rather than shrugged off. It is the one
    # failure that raises no error anywhere: every stage reports success and the only
    # symptom is text that stops early. Calling it a gap costs a warning on a genuinely
    # quiet recording, and that is the cheaper mistake by a wide margin.
    gaps += [
        Gap(chunk.index, chunk.start, chunk.end, "no speech recognised")
        for chunk in outcome.silent
    ]
    gaps += [
        Gap(chunk.index, chunk.start, chunk.end, "words not timed")
        for chunk in outcome.unaligned
    ]
    covered, silent = _measured_coverage(context, segments)
    wanted_diarization = "diarize" in requested
    return RunReport(
        duration=context.duration,
        segments=len(segments),
        words=outcome.words,
        chunks=outcome.chunks or len(context.chunks),
        covered=covered,
        silent_seconds=silent,
        speakers=outcome.speakers,
        gaps=sorted(gaps, key=lambda gap: gap.start),
        alignment_requested="align" in requested,
        alignment_complete=not outcome.unaligned,
        diarization_requested=wanted_diarization,
        diarization_complete=bool(context.turns) if wanted_diarization else True,
        canonical_files=[path.name for path in outcome.canonical],
    )


def _measured_coverage(context: _Context, segments: list[dict]) -> tuple[float, float]:
    """Time with text, and time known to hold no speech, both in seconds.

    Measured from the chunks that produced text rather than from the last timestamp,
    because a hole in the middle of a recording is invisible to a maximum: whatever
    comes after it still moves the end time outward.

    A chunk that failed is excluded here as firmly as one that found no speech. It is
    easy to forget: counting a failed chunk as covered makes the one measurement that
    catches a silent shortfall claim full coverage while the text is missing.
    """
    chunks = context.chunks
    if not chunks:
        return coverage_seconds(segments), 0.0
    quiet = {chunk.index for chunk in context.outcome.silent}
    failed = {chunk.index for chunk in context.outcome.missing}
    productive = [
        (chunk.start, chunk.start + chunk.length)
        for chunk in chunks
        if chunk.index not in quiet and chunk.index not in failed
    ]
    silent = [
        (chunk.start, chunk.start + chunk.length)
        for chunk in chunks
        if chunk.index in quiet
    ]
    return union_seconds(productive), union_seconds(silent)


def _analyze(context: _Context) -> None:
    """A short factual report: level, coverage, and who spoke how much."""
    studio = context.studio
    studio.begin_stage("Measuring quality")
    segments = _collected_segments(context)
    chunks = context.chunks or _chunks_from_plan(context)
    level = mean_volume_db(context.ffmpeg, context.source, env=context.env)
    words = sum(len(str(segment.get("text", "")).split()) for segment in segments)
    level_text = f"{level:.1f} dB" if level is not None else "unavailable"

    lines = [
        f"source           {context.source}",
        f"duration         {context.duration:.1f}s",
        f"mean level       {level_text}",
        f"chunks           {len(chunks)}",
        f"segments         {len(segments)}",
        f"recovered words  {words}",
    ]
    if segments:
        covered = max(float(segment.get("end", 0.0)) for segment in segments)
        share = covered / context.duration * 100 if context.duration else 0.0
        lines.append(f"covered          {covered:.1f}s ({share:.1f}% of the recording)")
    if context.turns:
        spoken: dict[str, float] = {}
        for turn in context.turns:
            spoken[turn.speaker] = spoken.get(turn.speaker, 0.0) + turn.duration
        for speaker, seconds in sorted(spoken.items(), key=lambda item: -item[1]):
            share = seconds / context.duration * 100 if context.duration else 0.0
            lines.append(f"  {speaker:<14} {seconds:8.1f}s  {share:5.1f}%")

    context.outcome.report = "\n".join(lines) + "\n"
    studio.log_line(context.outcome.report)
    studio.end_stage()


def _collected_segments(context: _Context) -> list[dict]:
    """The merged transcript, read back from the cache when this run did not make it."""
    if context.segments:
        return context.segments
    chunks = context.chunks or _chunks_from_plan(context)
    context.segments = context.workspace.load_segments(chunks) if chunks else []
    return context.segments


def _chunks_from_plan(context: _Context) -> list[Chunk]:
    """Reproduce the slicing a previous run recorded."""
    options = context.options
    return context.workspace.stored_chunks(
        context.duration,
        chunk_seconds=options.chunk_seconds,
        overlap_seconds=options.chunk_overlap,
    )


_HANDLERS = {
    "prepare": _prepare,
    "boundaries": _boundaries,
    "transcribe": _transcribe,
    "align": _align,
    "diarize": _diarize,
    "export": _export,
    "analyze": _analyze,
}
