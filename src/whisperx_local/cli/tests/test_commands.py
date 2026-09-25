"""What a run says at the end of itself, for each kind of ending there is.

This is the last thing between a user and a wrong belief, so the cases are tested one by
one rather than assumed. The one that prompted this file: ``--only transcribe`` plans
the recognition stages and no export, wrote no transcript, and still announced
"Transcript complete" over a path that did not exist -- exiting zero.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from whisperx_local.cli.commands import (
    INCOMPLETE_STATUS,
    NOTHING_WRITTEN_STATUS,
    SIMPLE_BEAM_SIZE as _SIMPLE_BEAM_SIZE,
    _announce_outcome,
    _copy_beside_input,
    _require_ready,
    _simple_options,
)
from whisperx_local.cli.options import build_parser
from whisperx_local.engine import backends
from whisperx_local.services.report import Gap, RunReport
from whisperx_local.services.runner import Outcome


def workspace(source: Path):
    return mock.Mock(source=source)


def report(**overrides) -> RunReport:
    base = dict(duration=20.0, segments=4, words=33, chunks=1, covered=20.0)
    base.update(overrides)
    return RunReport(**base)


class AnnounceTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "meeting.m4a"
        self.source.write_bytes(b"audio")
        self.primary = self.root / "meeting.txt"
        self.raw = self.root / "meeting.raw.txt"
        self.notices: list[tuple[str, list]] = []

    def announce(self, outcome: Outcome, *, primary: Path | None = None):
        """Run the announcement, capturing the panel instead of printing it."""
        with mock.patch(
            "whisperx_local.cli.commands.print_notice",
            side_effect=lambda rows, **kwargs: self.notices.append(
                (kwargs.get("title", ""), rows)
            ),
        ):
            _announce_outcome(
                outcome,
                workspace(self.source),
                primary if primary is not None else self.primary,
                self.root / "meeting.log",
                1.5,
            )

    def title(self) -> str:
        return self.notices[-1][0]

    def rows(self) -> dict[str, str]:
        return dict(self.notices[-1][1])

    # -- the transcript is there and whole ---------------------------------

    def test_a_written_transcript_that_covers_the_recording_is_complete(self):
        self.primary.write_text("text", encoding="utf-8")
        self.announce(Outcome(words=33, summary=report()))
        self.assertIn("complete", self.title())
        self.assertEqual(self.rows()["Covered"], "20s of 20s (100%)")

    def test_a_partial_transcript_is_not_called_complete_and_exits_non_zero(self):
        self.primary.write_text("text", encoding="utf-8")
        outcome = Outcome(words=10, summary=report(covered=8.0, gaps=[Gap(0, 12.0, 20.0, "not transcribed")]))
        with self.assertRaises(SystemExit) as raised:
            self.announce(outcome)
        self.assertEqual(raised.exception.code, INCOMPLETE_STATUS)
        self.assertIn("INCOMPLETE", self.title())

    # -- no transcript: the case that was wrong -----------------------------

    def test_nothing_written_is_never_called_complete(self):
        # The bug: no output file, no report, and the panel said "complete".
        self.raw.write_text("text", encoding="utf-8")
        self.announce(Outcome(words=33, canonical=[self.raw]))
        self.assertNotIn("complete", self.title().lower())

    def test_a_raw_copy_alone_says_what_was_written_and_how_to_render_it(self):
        self.raw.write_text("text", encoding="utf-8")
        self.announce(Outcome(words=33, canonical=[self.raw]))
        rows = self.rows()
        self.assertEqual(rows["Wrote"], "meeting.raw.txt")
        self.assertIn("--only export", rows["To write it"])
        self.assertEqual(rows["Recognised"], "33 words")

    def test_a_raw_copy_alone_is_not_a_failure_because_the_request_was_honoured(self):
        self.raw.write_text("text", encoding="utf-8")
        self.announce(Outcome(words=33, canonical=[self.raw]))

    def test_writing_no_file_at_all_fails(self):
        with self.assertRaises(SystemExit) as raised:
            self.announce(Outcome(words=0))
        self.assertEqual(raised.exception.code, NOTHING_WRITTEN_STATUS)
        self.assertIn("Nothing was written", self.title())

    def test_the_suggestion_names_a_command_that_exists(self):
        self.raw.write_text("text", encoding="utf-8")
        self.announce(Outcome(words=33, canonical=[self.raw]))
        self.assertNotIn("app.py", self.rows()["To write it"])

    # -- the transcript was already there ----------------------------------

    def test_an_untouched_existing_transcript_is_not_claimed_as_this_runs_work(self):
        self.primary.write_text("older text", encoding="utf-8")
        self.announce(Outcome(words=0, summary=None))
        self.assertIn("Existing", self.title())
        self.assertIn("left as it was", self.rows()["Note"])

    def test_a_missing_report_alone_is_not_taken_as_success(self):
        # No file and no report must take the failure path, not the happy one.
        with self.assertRaises(SystemExit):
            self.announce(Outcome(words=0, summary=None))


class StubEngine:
    """A recogniser whose readiness the test decides."""

    def __init__(self, *, available: bool = True, reason: str = "nothing is wrong", model=None):
        self.key = "stub"
        self.label = "Stub recogniser"
        self._available = available
        self._reason = reason
        self._model = model

    def available(self) -> bool:
        return self._available

    def missing_reason(self) -> str:
        return self._reason

    def notes(self, options):
        """Part of the backend contract the preflight reads before every run."""
        return []

    def model_reason(self) -> str:
        return "no model was found"

    def model_path(self, options):
        return self._model


def options_for(engine: str):
    return mock.Mock(engine=engine, model="large-v3", language="fa")


class ReadyBeforeStartingTest(unittest.TestCase):
    """Refuse work that cannot finish, before any of it is done.

    Both cases here cost real time when they are only discovered late: a recogniser that
    is not installed dies on its first chunk, and a model that has to be downloaded can be
    an hour of network. The second one is the reason this check exists -- an hour was spent
    watching a stalled progress bar that this reports in a single line.
    """

    def setUp(self):
        self.prints: list[str] = []
        patcher = mock.patch("builtins.print", side_effect=lambda *a, **k: self.prints.append(" ".join(map(str, a))))
        patcher.start()
        self.addCleanup(patcher.stop)

    def warned(self) -> str:
        return " ".join(self.prints)

    # -- the recogniser itself ---------------------------------------------

    def test_an_unavailable_engine_is_refused_before_the_run(self):
        stub = StubEngine(available=False, reason="the binary is not on PATH")
        with mock.patch.dict(backends.BACKENDS, {"stub": stub}):
            with self.assertRaises(SystemExit) as raised:
                _require_ready(options_for("stub"), "offline")
        self.assertIn("the binary is not on PATH", str(raised.exception))

    def test_the_refusal_names_the_engine_that_cannot_run(self):
        stub = StubEngine(available=False, reason="no binary")
        with mock.patch.dict(backends.BACKENDS, {"stub": stub}):
            with self.assertRaises(SystemExit) as raised:
                _require_ready(options_for("stub"), "offline")
        self.assertIn("Stub recogniser", str(raised.exception))

    def test_an_available_engine_with_its_model_is_allowed_through(self):
        stub = StubEngine(model=Path("/tmp/ggml-large-v3.bin"))
        with mock.patch.dict(backends.BACKENDS, {"stub": stub}):
            _require_ready(options_for("stub"), "offline")

    def test_an_available_engine_without_a_model_is_refused(self):
        # Knowing the binary exists is not enough, and the complaint has to be about the
        # model: reporting a missing model as a missing binary -- which this did -- sends
        # the reader looking for the wrong thing.
        stub = StubEngine(model=None, reason="the binary would be fine")
        with mock.patch.dict(backends.BACKENDS, {"stub": stub}):
            with self.assertRaises(SystemExit) as raised:
                _require_ready(options_for("stub"), "offline")
        message = str(raised.exception)
        self.assertIn("no model", message)
        self.assertNotIn("the binary would be fine", message)

    # -- the model the default engine needs --------------------------------

    def test_a_cached_model_starts_silently(self):
        with mock.patch("whisperx_local.cli.commands.core_models_cached", return_value=True):
            _require_ready(options_for("whisperx"), "offline")
        self.assertEqual(self.warned(), "")

    def test_a_model_that_must_be_downloaded_is_announced_first(self):
        with mock.patch("whisperx_local.cli.commands.core_models_cached", return_value=False):
            _require_ready(options_for("whisperx"), "online")
        self.assertIn("download", self.warned())

    def test_a_download_is_warned_about_rather_than_refused(self):
        # Fetching a model is legitimate; being surprised by it is not.
        with mock.patch("whisperx_local.cli.commands.core_models_cached", return_value=False):
            self.assertIsNone(_require_ready(options_for("whisperx"), "online"))
        self.assertIn("interrupted", self.warned())

    def test_an_uncached_model_with_the_network_offline_is_refused(self):
        # Without this the run cannot begin, and saying so is the only useful outcome.
        with mock.patch("whisperx_local.cli.commands.core_models_cached", return_value=False):
            with self.assertRaises(SystemExit) as raised:
                _require_ready(options_for("whisperx"), "offline")
        message = str(raised.exception)
        self.assertIn("large-v3", message)
        self.assertIn("offline", message)

    def test_the_refusal_says_how_to_proceed(self):
        with mock.patch("whisperx_local.cli.commands.core_models_cached", return_value=False):
            with self.assertRaises(SystemExit) as raised:
                _require_ready(options_for("whisperx"), "offline")
        self.assertIn("--network online", str(raised.exception))


class SimpleCommandTest(unittest.TestCase):
    """The one-shot command: a path in, text out, and no saved settings involved.

    The promise is predictability, so these assert the pinned values and, more
    importantly, that nothing is read from the settings file at all.
    """

    def options(self, language: str = "fa"):
        return _simple_options(Path("/tmp/meeting.m4a"), language=language)

    def test_it_never_reads_the_saved_settings(self):
        # This is the whole point: the same path behaves the same for an agent as for the
        # person who ran it last, whatever is in the settings file.
        with mock.patch("whisperx_local.config.Preferences.load") as load:
            self.options()
        load.assert_not_called()

    def test_the_language_is_the_only_thing_that_varies(self):
        self.assertEqual(self.options().language, "fa")
        self.assertEqual(self.options("en").language, "en")

    def test_it_writes_a_readable_transcript_not_only_the_raw_copy(self):
        # An agent asking for text should get text, not a file it has to convert.
        self.assertIn("export", self.options().only)

    def test_it_pins_the_choices_that_shape_the_text(self):
        options = self.options()
        self.assertIn(options.engine, {"whispercpp", "whisperx"})
        self.assertEqual(options.model, "automatic")
        self.assertIsNone(options.preset)
        self.assertEqual(options.beam_size, _SIMPLE_BEAM_SIZE)

    def test_it_needs_no_huggingface_token_to_run(self):
        # Silero rather than pyannote, so a machine with no token can still run it.
        options = self.options()
        self.assertEqual(options.vad_method, "silero")
        self.assertFalse(options.diarize)
        self.assertIsNone(options.hotwords)

    def test_the_output_lands_under_the_output_directory(self):
        from whisperx_local.paths import OUTPUT_DIR

        self.assertEqual(self.options().output_dir, str(OUTPUT_DIR))

    def test_it_asks_for_no_analysis_it_was_not_asked_for(self):
        options = self.options()
        self.assertFalse(options.diarize)
        self.assertFalse(options.analyze)
        self.assertFalse(options.dry_run)

    def test_the_command_is_reachable_from_the_parser(self):
        args = build_parser().parse_args(["transcribe", "a.m4a", "b.m4a"])
        self.assertEqual(args.handler, "transcribe")
        self.assertEqual(args.files, ["a.m4a", "b.m4a"])
        self.assertEqual(args.language, "fa")

    def test_several_recordings_can_be_dropped_in_at_once(self):
        args = build_parser().parse_args(["transcribe", "a.m4a", "b.m4a", "c.m4a"])
        self.assertEqual(len(args.files), 3)

    def test_a_bare_path_is_routed_here_by_the_launcher(self):
        # The routing lives in a shell script, so it is asserted rather than trusted.
        launcher = Path(__file__).resolve().parents[4] / "cli"
        if not launcher.is_file():
            self.skipTest("no launcher in this layout")
        text = launcher.read_text(encoding="utf-8")
        self.assertIn('set -- transcribe "$@"', text)

    def test_the_command_ignores_the_ambient_preferences_file(self):
        # Building the options must not consult the settings on disk, even indirectly
        # through the parser's defaults.
        with mock.patch(
            "whisperx_local.config.Preferences.load",
            side_effect=AssertionError("the one-shot command read the saved settings"),
        ):
            self.options("en")


class CopyBesideInputTest(unittest.TestCase):
    """The copy left next to the recording: convenient, and never a way to lose a run.

    A transcript is easier to find beside its recording than inside a directory named after
    it, so one is left there too. It is a convenience on top of work that is already saved,
    which is what decides the awkward cases: it must never overwrite the recording it is
    reading, and it must never turn a finished run into a failed one.
    """

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.recordings = self.root / "Downloads"
        self.recordings.mkdir()
        self.source = self.recordings / "talk.m4a"
        self.source.write_bytes(b"audio")
        self.primary = self.output / "talk.txt"
        self.primary.write_text("سلام", encoding="utf-8")

    def beside(self) -> Path:
        return self.recordings / "talk.txt"

    def copy(self, *, exported: bool = True, setting: bool = True, suffix: str = ".txt"):
        primary = self.primary.with_suffix(suffix)
        if suffix != self.primary.suffix:
            primary.write_text("سلام", encoding="utf-8")
        options = mock.Mock(copy_beside_input=setting)
        return _copy_beside_input(options, self.source, primary, exported=exported)

    # -- when it happens ---------------------------------------------------

    def test_a_transcript_is_left_beside_the_recording(self):
        self.assertEqual(self.copy(), self.beside())
        self.assertTrue(self.beside().is_file())

    def test_the_copy_holds_the_transcript(self):
        self.copy()
        self.assertEqual(self.beside().read_text(encoding="utf-8"), "سلام")

    def test_the_copy_carries_the_exported_suffix_not_the_recordings(self):
        # A subtitle request must leave a .srt beside the recording, not a .txt.
        destination = self.copy(suffix=".srt")
        self.assertEqual(destination.suffix, ".srt")
        self.assertEqual(destination.stem, self.source.stem)

    # -- when it must not --------------------------------------------------

    def test_a_recording_already_in_input_is_excepted(self):
        # They were collected there to be transcribed, so a copy is clutter. This is the
        # one exception the behaviour was asked for by name.
        with mock.patch("whisperx_local.paths.INPUT_DIR", self.recordings):
            self.assertIsNone(self.copy())
        self.assertFalse(self.beside().exists())

    def test_a_recording_inside_a_subdirectory_of_input_is_excepted_too(self):
        deeper = self.recordings / "2026" / "march"
        deeper.mkdir(parents=True)
        source = deeper / "talk.m4a"
        source.write_bytes(b"audio")
        with mock.patch("whisperx_local.paths.INPUT_DIR", self.recordings):
            self.assertIsNone(
                _copy_beside_input(
                    mock.Mock(copy_beside_input=True), source, self.primary, exported=True
                )
            )

    def test_nothing_is_copied_when_the_setting_is_off(self):
        self.assertIsNone(self.copy(setting=False))
        self.assertFalse(self.beside().exists())

    def test_nothing_is_copied_when_no_readable_transcript_was_written(self):
        # A raw-only run has no ready-to-use file to place, so there is nothing to copy.
        self.assertIsNone(self.copy(exported=False))
        self.assertFalse(self.beside().exists())

    def test_nothing_is_copied_when_there_is_no_transcript_to_copy(self):
        self.primary.unlink()
        self.assertIsNone(self.copy())

    def test_the_recording_itself_is_never_overwritten(self):
        # Transcribing ``talk.txt`` must not replace its own input with the transcript.
        source = self.recordings / "talk.txt"
        source.write_text("the original", encoding="utf-8")
        self.assertIsNone(
            _copy_beside_input(
                mock.Mock(copy_beside_input=True), source, self.primary, exported=True
            )
        )
        self.assertEqual(source.read_text(encoding="utf-8"), "the original")

    # -- it can never cost the run -----------------------------------------

    def test_a_folder_that_cannot_be_written_to_does_not_raise(self):
        # The transcript is already safe in the output directory by this point, and a
        # read-only folder must not turn a finished run into a failed one. The warning it
        # prints is checked by the next test, so it is silenced here.
        with mock.patch("shutil.copyfile", side_effect=OSError("Read-only file system")):
            with mock.patch("builtins.print"):
                self.assertIsNone(self.copy())

    def test_that_failure_is_reported_rather_than_swallowed(self):
        with mock.patch("shutil.copyfile", side_effect=OSError("Read-only file system")):
            with mock.patch("builtins.print") as printed:
                self.copy()
        self.assertIn("beside", " ".join(str(call) for call in printed.call_args_list))

    def test_a_copy_that_succeeds_says_nothing_extra_on_stdout(self):
        # The panel names the path; the copy itself does not need to print.
        with mock.patch("builtins.print") as printed:
            self.copy()
        printed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
