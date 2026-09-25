"""Knowing what is already cached, and recording it without breaking a finished run.

Both halves of this earned a test the same way.

The check: ``--model`` is either a size (``large-v3``) or a full repository id
(``owner/name``). Only the size form was understood, so a repository id -- the only way to
name a turbo model -- never looked cached, and the code that records it built a *path* out
of the name. It crashed with ``FileNotFoundError`` on the slash in ``owner/name``.

The timing of that crash is the part worth remembering: it happened after the transcript
had been written, in bookkeeping for future runs. A run that had fully succeeded ended in a
traceback and never printed what it had produced, which is indistinguishable from a run
that failed. Bookkeeping must not be able to do that.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from whisperx_local.services import jobs


class ModelDirTest(unittest.TestCase):
    """A throwaway models directory, standing in for the real Hugging Face cache."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.models = Path(temporary.name)
        patcher = mock.patch.object(jobs, "MODELS_DIR", self.models)
        patcher.start()
        self.addCleanup(patcher.stop)

    def persian(self) -> None:
        self.snapshot("models--jonatasgrosman--wav2vec2-large-xlsr-53-persian")

    def snapshot(self, directory: str) -> None:
        """A cached model: the cache directory, a revision, and a file inside it."""
        revision = self.models / directory / "snapshots" / "abc123"
        revision.mkdir(parents=True, exist_ok=True)
        (revision / "model.bin").write_bytes(b"weights")


class CacheLookupTest(ModelDirTest):
    """A model that is on disk has to be recognised as being on disk."""

    # -- a size, the form that always worked -------------------------------

    def test_a_size_is_found_under_its_owner_directory(self):
        self.snapshot("models--Systran--faster-whisper-large-v3")
        jobs.mark_core_ready("large-v3", "en")
        self.assertTrue(jobs.core_models_cached("large-v3", "en"))

    def test_a_size_with_no_owner_suffix_is_found_too(self):
        self.snapshot("models--Systran--faster-whisper-large-v3")
        jobs.mark_core_ready("faster-whisper-large-v3", "en")
        self.assertTrue(jobs.core_models_cached("faster-whisper-large-v3", "en"))

    def test_a_missing_model_is_not_cached_even_with_a_marker(self):
        # The marker only says a previous run got that far; the snapshot is the model.
        jobs.mark_core_ready("large-v3", "en")
        self.assertFalse(jobs.core_models_cached("large-v3", "en"))

    # -- a repository id, the form that crashed ----------------------------

    def test_a_repository_id_is_found(self):
        # `--model owner/name` is how a turbo model is named, and it is cached under
        # models--owner--name. It used to look uncached, which meant the run was treated
        # as a download it did not need.
        self.snapshot("models--deepdml--faster-whisper-large-v3-turbo-ct2")
        jobs.mark_core_ready("deepdml/faster-whisper-large-v3-turbo-ct2", "en")
        self.assertTrue(jobs.core_models_cached("deepdml/faster-whisper-large-v3-turbo-ct2", "en"))

    def test_the_owner_is_part_of_the_model(self):
        # Same name, different owner, and the marker is present: only the repository that
        # is actually on disk may count as cached.
        self.snapshot("models--deepdml--faster-whisper-large-v3-turbo-ct2")
        jobs.mark_core_ready("someoneelse/faster-whisper-large-v3-turbo-ct2", "en")
        self.assertFalse(
            jobs.core_models_cached("someoneelse/faster-whisper-large-v3-turbo-ct2", "en")
        )

    def test_an_absent_repository_id_is_not_claimed_as_cached(self):
        self.assertFalse(jobs.core_models_cached("deepdml/faster-whisper-large-v3-turbo-ct2", "en"))

    def test_only_the_snapshots_revision_counts_not_a_stray_directory(self):
        # An interrupted download leaves the directory behind with no revision in it,
        # which is exactly the state that made a model look ready when it was not.
        (self.models / "models--deepdml--faster-whisper-large-v3-turbo-ct2").mkdir()
        self.assertFalse(jobs.core_models_cached("deepdml/faster-whisper-large-v3-turbo-ct2", "en"))

    # -- the Persian aligner is required as well ---------------------------

    def test_a_persian_run_needs_the_aligner_as_well_as_the_model(self):
        self.snapshot("models--Systran--faster-whisper-large-v3")
        self.assertFalse(jobs.core_models_cached("large-v3", "fa"))
        self.persian()
        self.assertTrue(jobs.core_models_cached("large-v3", "fa"))


class MarkerTest(ModelDirTest):
    """Recording readiness, which must never depend on the shape of the model name."""

    def test_marking_a_repository_id_does_not_raise(self):
        # The crash: the slash in the name made this a path whose parent was missing.
        jobs.mark_core_ready("deepdml/faster-whisper-large-v3-turbo-ct2", "fa")

    def test_a_marked_repository_id_is_remembered(self):
        self.snapshot("models--deepdml--faster-whisper-large-v3-turbo-ct2")
        jobs.mark_core_ready("deepdml/faster-whisper-large-v3-turbo-ct2", "en")
        self.assertTrue(jobs.core_models_cached("deepdml/faster-whisper-large-v3-turbo-ct2", "en"))

    def test_marking_writes_something_inside_the_models_directory(self):
        # A marker that escaped the directory would survive a cache wipe.
        jobs.mark_core_ready("deepdml/faster-whisper-large-v3-turbo-ct2", "en")
        written = list(self.models.iterdir())
        self.assertTrue(written)

    def test_the_marker_name_carries_no_directory_separator(self):
        jobs.mark_core_ready("deepdml/faster-whisper-large-v3-turbo-ct2", "en")
        for entry in self.models.iterdir():
            with self.subTest(entry=entry.name):
                self.assertNotIn("/", entry.name)
                self.assertTrue(entry.is_file())

    def test_a_size_still_markers_under_its_plain_name(self):
        # The name that existing caches on disk already use, so it is asserted rather
        # than left to a rename.
        jobs.mark_core_ready("large-v3", "fa")
        self.assertTrue((self.models / ".core-ready-large-v3-fa").is_file())

    def test_a_full_rollup_name_markers_under_the_size(self):
        jobs.mark_core_ready("faster-whisper-large-v3", "fa")
        self.assertTrue((self.models / ".core-ready-large-v3-fa").is_file())

    def test_diarization_is_marked_separately(self):
        jobs.mark_diarization_ready()
        self.assertTrue(jobs.diarization_ready())

    def test_diarization_is_not_ready_until_it_is_marked(self):
        self.assertFalse(jobs.diarization_ready())


class BookkeepingCannotFailARunTest(unittest.TestCase):
    """The part that made the crash expensive rather than cosmetic.

    A finished transcript must still be reported as finished when a marker cannot be
    written, because the marker only affects what a *later* run may skip.
    """

    def context(self):
        from whisperx_local.services.runner import Outcome

        outcome = Outcome()
        context = mock.Mock()
        context.outcome = outcome
        context.options = mock.Mock(model="deepdml/faster-whisper-large-v3-turbo-ct2", language="fa")
        context.source = Path("/tmp/meeting.m4a")
        return context

    def plan(self):
        plan = mock.Mock()
        plan.keys.return_value = {"transcribe", "export"}
        return plan

    def test_a_marker_that_cannot_be_written_does_not_raise(self):
        from whisperx_local.services import runner

        context = self.context()
        with mock.patch.object(
            runner, "mark_core_ready", side_effect=FileNotFoundError("no such directory")
        ):
            runner._record_learning(self.plan(), context)

    def test_the_failure_is_recorded_as_a_warning_rather_than_swallowed(self):
        # Swallowing it silently would hide a real cache problem.
        from whisperx_local.services import runner

        context = self.context()
        with mock.patch.object(
            runner, "mark_core_ready", side_effect=FileNotFoundError("no such directory")
        ):
            runner._record_learning(self.plan(), context)
        self.assertTrue(context.outcome.warnings)
        self.assertIn("cached state", context.outcome.warnings[0])

    def test_a_running_out_of_disk_while_writing_a_marker_is_survivable(self):
        from whisperx_local.services import runner

        context = self.context()
        with mock.patch.object(runner, "mark_core_ready", side_effect=OSError("No space left")):
            runner._record_learning(self.plan(), context)
        self.assertTrue(context.outcome.warnings)


if __name__ == "__main__":
    unittest.main()
