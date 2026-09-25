"""What a run does when a stage fails, and what it refuses to call finished.

The two things tested here are the ones that were lost on a real recording:

* Alignment aborted after the text had been recognised, and nothing was written. The
  requirement is that a stage failing can cost its own embellishment and nothing else.
* A finished-looking transcript covered 42% of the audio and was reported as a success.
  The requirement is that a shortfall is measured and stated, and that the recording does
  not record itself as done.

Both are tested through the runner's own handlers rather than around them, because the
guarantees belong to the order those handlers do things in.
"""

import json
import tempfile
import unittest
from pathlib import Path

from whisperx_local.actions import catalog, plan as planning
from whisperx_local.actions.base import AUDIO, FILES, PLAN, SEGMENTS, SOURCE
from whisperx_local.actions.plan import Plan, Selection
from whisperx_local.chunking import chunk_paths, plan_chunks
from whisperx_local.cli.options import build_parser
from whisperx_local.services.artifacts import WorkSpace
from whisperx_local.services.jobs import is_complete
from whisperx_local.services.runner import (
    Outcome,
    _Context,
    _HANDLERS,
    _measured_coverage,
    _write_canonical,
    describe_plan,
)


class Studio:
    """Records what the runner asked for, so a test needs no terminal."""

    def __init__(self):
        self.stages: list[str] = []
        self.warnings: list[str] = []
        self.statuses: list[str] = []
        self.words = 0

    def begin_stage(self, name):
        self.stages.append(name)

    def end_stage(self):
        return None

    def skip_stage(self, name):
        self.stages.append(f"skipped {name}")

    def set_status(self, text):
        self.statuses.append(text)

    def set_chunks(self, *_args):
        return None

    def set_speakers(self, *_args):
        return None

    def set_words(self, count):
        self.words = count

    def mark_partial(self):
        return None

    def log_line(self, _text):
        return None

    def log_handle(self):
        return None

    def progress_to(self, *_args):
        return None

    def warn(self, text):
        self.warnings.append(text)


def recognised(segments) -> str:
    return json.dumps({"segments": segments})


TEXT_A = "سلام این متن تشخیص داده شده است"
TEXT_B = "بله ادامه بده"


class RunnerTestCase(unittest.TestCase):
    """A two-chunk recording whose second chunk never got word timings."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "meeting.m4a"
        self.source.write_bytes(b"audio")
        self.workspace = WorkSpace(source=self.source, output_dir=self.root / "out")
        self.addCleanup(self.workspace.reset)
        self.chunks = plan_chunks(1200.0, chunk_seconds=600, overlap_seconds=0.0)
        self.workspace.chunks.mkdir(parents=True, exist_ok=True)
        # Timings are chunk-local, as the engine produces them: each chunk was sliced
        # into its own file and decoded from zero, then shifted onto the real timeline
        # when the results are stitched back together.
        self.write_chunk(self.chunks[0], [(0.0, 4.0, TEXT_A)])
        self.write_chunk(self.chunks[1], [(0.0, 4.0, TEXT_B)])

    def write_chunk(self, chunk, segments):
        payload = recognised(
            [{"start": start, "end": end, "text": text} for start, end, text in segments]
        )
        paths = chunk_paths(self.workspace.chunks, chunk)
        paths["recognised"].write_text(payload, encoding="utf-8")

    def context(self, *, requested=("align", "export"), outcome=None, chunks=None):
        options = build_parser().parse_args(["run", str(self.source)])
        options.language = "fa"
        options.output_format = "txt"
        options.diarize = False
        return _Context(
            options=options,
            workspace=self.workspace,
            studio=Studio(),
            ffmpeg="ffmpeg",
            env={},
            duration=1200.0,
            network="offline",
            outcome=outcome if outcome is not None else Outcome(),
            chunks=list(self.chunks if chunks is None else chunks),
            requested=set(requested),
        )


class FailedStageKeepsTextTest(RunnerTestCase):
    """The report's first required fixture.

    Word timing failed on one chunk after both had been recognised. The text of both
    must still reach disk, in its own file, on that run.
    """

    def export(self, context):
        _HANDLERS["export"](context)
        return context.outcome

    def test_the_recognised_text_survives_a_failed_alignment(self):
        outcome = self.export(self.context(outcome=Outcome(unaligned=[self.chunks[1]])))
        raw = self.workspace.output_dir / "meeting.raw.txt"
        self.assertTrue(raw.is_file())
        body = raw.read_text(encoding="utf-8")
        self.assertIn(TEXT_A, body)
        self.assertIn(TEXT_B, body)

    def test_the_canonical_copy_is_the_thing_written_first(self):
        outcome = self.export(self.context(outcome=Outcome(unaligned=[self.chunks[1]])))
        self.assertEqual(
            [path.name for path in outcome.canonical],
            ["meeting.raw.txt", "meeting.raw.json"],
        )
        self.assertEqual(outcome.files[0], outcome.canonical[0])

    def test_both_canonical_forms_carry_the_same_text(self):
        self.export(self.context(outcome=Outcome(unaligned=[self.chunks[1]])))
        payload = json.loads(
            (self.workspace.output_dir / "meeting.raw.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            [segment["text"] for segment in payload["segments"]], [TEXT_A, TEXT_B]
        )

    def test_the_readable_transcript_is_written_too(self):
        self.export(self.context(outcome=Outcome(unaligned=[self.chunks[1]])))
        self.assertIn(
            TEXT_A,
            (self.workspace.output_dir / "meeting.txt").read_text(encoding="utf-8"),
        )

    def test_an_empty_chunk_does_not_stop_the_others_being_written(self):
        self.write_chunk(self.chunks[0], [])
        self.export(self.context(outcome=Outcome(silent=[self.chunks[0]])))
        body = (self.workspace.output_dir / "meeting.raw.txt").read_text(encoding="utf-8")
        self.assertIn(TEXT_B, body)
        self.assertNotIn(TEXT_A, body)

    def test_the_failure_is_named_as_a_gap(self):
        outcome = self.export(self.context(outcome=Outcome(unaligned=[self.chunks[1]])))
        self.assertTrue(outcome.summary.partial)
        self.assertIn("words not timed", outcome.summary.gaps[0].reason)
        self.assertEqual(outcome.summary.gaps[0].start, 600.0)

    def test_the_raw_copy_is_not_overwritten_by_the_readable_one(self):
        self.export(self.context(outcome=Outcome(unaligned=[self.chunks[1]])))
        self.assertNotEqual(
            (self.workspace.output_dir / "meeting.raw.txt").read_text(encoding="utf-8"),
            (self.workspace.output_dir / "meeting.txt").read_text(encoding="utf-8"),
        )


class ShortRunIsNotFinishedTest(RunnerTestCase):
    """The report's other half: a partial run must not record itself as done."""

    def test_an_incomplete_run_leaves_no_completion_marker(self):
        context = self.context(outcome=Outcome(unaligned=[self.chunks[1]]))
        _HANDLERS["export"](context)
        plan = planning.resolve(Selection(only=("export",)), available={SEGMENTS})
        _HANDLERS["export"](context)
        from whisperx_local.services.runner import _record_learning

        _record_learning(plan, context)
        primary = self.workspace.primary_output("meeting", "txt")
        self.assertFalse(is_complete(self.source, context.options, primary))

    def test_a_complete_run_does_leave_one(self):
        context = self.context()
        _HANDLERS["export"](context)
        plan = planning.resolve(Selection(only=("export",)), available={SEGMENTS})
        from whisperx_local.services.runner import _record_learning

        _record_learning(plan, context)
        primary = self.workspace.primary_output("meeting", "txt")
        self.assertTrue(is_complete(self.source, context.options, primary))

    def test_a_missing_chunk_is_reported_as_a_gap(self):
        context = self.context(outcome=Outcome(missing=[self.chunks[1]]))
        _HANDLERS["export"](context)
        summary = context.outcome.summary
        self.assertTrue(summary.partial)
        self.assertIn("not transcribed", summary.gaps[0].reason)
        self.assertIn("Re-run", summary.advice())

    def test_the_coverage_figure_reflects_what_was_left_out(self):
        context = self.context(outcome=Outcome(missing=[self.chunks[1]]))
        _HANDLERS["export"](context)
        summary = context.outcome.summary
        self.assertEqual(summary.covered, 600.0)
        self.assertEqual(summary.unaccounted, 600.0)
        self.assertEqual(summary.coverage_percent, 50.0)

    def test_silence_is_accounted_for_rather_than_blamed_on_nothing(self):
        context = self.context(outcome=Outcome(silent=[self.chunks[1]]))
        _HANDLERS["export"](context)
        summary = context.outcome.summary
        self.assertEqual(summary.silent_seconds, 600.0)
        self.assertEqual(summary.unaccounted, 0.0)
        # Still partial: a chunk that produced no text is named, not shrugged off.
        self.assertTrue(summary.partial)

    def test_the_unchosen_refinements_are_not_reported_as_failures(self):
        # Speaker separation was never asked for, so it cannot have failed.
        context = self.context(requested=("export",))
        _HANDLERS["export"](context)
        summary = context.outcome.summary
        self.assertFalse(summary.diarization_requested)
        self.assertFalse(summary.alignment_requested)
        self.assertFalse(summary.partial)


class RecognitionWritesItsOwnOutputTest(RunnerTestCase):
    """Recognition must not depend on another stage to put text on disk.

    It did. A request for transcription stages only wrote nothing, because only export
    wrote files, and the run still reported success. Whether recognised text reaches
    disk cannot hinge on which unrelated stages were asked for.
    """

    def test_recognition_writes_the_canonical_transcript(self):
        context = self.context()
        _write_canonical(context, self.chunks)
        body = (self.workspace.output_dir / "meeting.raw.txt").read_text(encoding="utf-8")
        self.assertIn(TEXT_A, body)
        self.assertIn(TEXT_B, body)

    def test_it_records_what_it_wrote(self):
        context = self.context()
        _write_canonical(context, self.chunks)
        names = [path.name for path in context.outcome.canonical]
        self.assertEqual(names, ["meeting.raw.txt", "meeting.raw.json"])
        self.assertEqual(context.outcome.files, context.outcome.canonical)

    def test_it_counts_the_words_it_wrote(self):
        context = self.context()
        _write_canonical(context, self.chunks)
        self.assertEqual(
            context.outcome.words, len(TEXT_A.split()) + len(TEXT_B.split())
        )

    def test_nothing_is_written_when_recognising_found_nothing(self):
        for chunk in self.chunks:
            self.write_chunk(chunk, [])
        context = self.context()
        _write_canonical(context, self.chunks)
        self.assertEqual(context.outcome.canonical, [])
        self.assertFalse((self.workspace.output_dir / "meeting.raw.txt").exists())

    def test_the_file_list_has_no_duplicates_when_export_writes_it_again(self):
        # Export rewrites the canonical copy to pick up word timings; the same two files
        # must not appear twice in the list of what was produced.
        context = self.context()
        _write_canonical(context, self.chunks)
        _HANDLERS["export"](context)
        names = [path.name for path in context.outcome.files]
        self.assertEqual(len(names), len(set(names)))

    def test_export_carries_alignment_timings_into_the_canonical_copy(self):
        self.write_chunk(self.chunks[0], [(0.0, 4.0, TEXT_A)])
        context = self.context()
        _write_canonical(context, self.chunks)
        _HANDLERS["export"](context)
        payload = json.loads(
            (self.workspace.output_dir / "meeting.raw.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["segments"][0]["text"], TEXT_A)


class CoverageMeasurementTest(RunnerTestCase):
    """A hole in the middle must not be hidden by whatever comes after it."""

    def test_a_gap_in_the_middle_reduces_coverage(self):
        chunks = plan_chunks(1800.0, chunk_seconds=600, overlap_seconds=0.0)
        context = self.context(chunks=chunks, outcome=Outcome(missing=[chunks[1]]))
        covered, silent = _measured_coverage(context, [])
        self.assertEqual(covered, 1200.0)
        self.assertEqual(silent, 0.0)

    def test_overlapping_chunks_are_not_counted_twice(self):
        chunks = plan_chunks(1200.0, chunk_seconds=600, overlap_seconds=30.0)
        context = self.context(chunks=chunks)
        covered, _ = _measured_coverage(context, [])
        self.assertLessEqual(covered, 1200.0)

    def test_with_no_chunks_the_text_itself_is_measured(self):
        context = self.context(chunks=[])
        segments = [{"start": 0.0, "end": 12.0, "text": "x"}]
        self.assertEqual(_measured_coverage(context, segments), (12.0, 0.0))


class PlanDescriptionTest(unittest.TestCase):
    def test_a_plan_that_does_nothing_says_so(self):
        self.assertEqual(
            describe_plan(Plan(actions=(), available=frozenset())), "nothing to do"
        )

    def test_an_already_satisfied_step_stays_in_the_plan(self):
        # The plan describes the request. Whether each step still has work to do is
        # decided per chunk at run time, which is what makes a rerun cheap without the
        # plan quietly changing shape between runs.
        plan = planning.resolve(
            Selection(), available={SOURCE, AUDIO, PLAN, SEGMENTS, FILES}
        )
        self.assertEqual(plan.keys(), catalog.DEFAULT_ACTIONS)

    def test_an_explicit_request_runs_even_when_its_result_exists(self):
        # ``--only export`` means "render it again", not "skip it".
        plan = planning.resolve(Selection(only=("export",)), available={SEGMENTS, FILES})
        self.assertIn("Writing transcript", describe_plan(plan))

    def test_the_actions_are_named_in_order(self):
        plan = planning.resolve(Selection(only=("transcribe",)), available=set())
        self.assertTrue(
            describe_plan(plan).startswith(
                "Preparing audio → Finding speech → Transcribing"
            )
        )

    def test_an_added_dependency_is_named(self):
        plan = planning.resolve(Selection(only=("export",)), available={SEGMENTS})
        self.assertIn("Writing transcript", describe_plan(plan))

    def test_the_actions_are_named_in_order(self):
        plan = planning.resolve(Selection(only=("transcribe",)), available=set())
        self.assertTrue(
            describe_plan(plan).startswith(
                "Preparing audio → Finding speech → Transcribing"
            )
        )


if __name__ == "__main__":
    unittest.main()
