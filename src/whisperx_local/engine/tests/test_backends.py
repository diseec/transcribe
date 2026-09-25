"""The recogniser seam: which engine runs, and what a second one must get right.

Two things are being protected here.

**The default must not move.** Every measurement and every transcript so far came from one
recogniser, and introducing a seam is exactly the kind of change that quietly alters the
command line or the file that gets read. So the default backend's command path is asserted
directly, including the output filename, because that name is the contract between the
command and the reader.

**A second engine must fail loudly, not quietly.** The whisper.cpp adapter cannot be run
here -- it needs a build and a multi-gigabyte model -- so what is tested is the part that
decides whether it is allowed to run at all, and the part that reads its output. An adapter
that misreads a payload as an empty transcript would look exactly like a recording with no
speech in it, which is the failure this codebase spends most of its effort preventing.
"""

import json
import os
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from whisperx_local.chunking import read_segments
from whisperx_local.cli.options import build_parser
from whisperx_local.engine import backends
from whisperx_local.engine.backends import (
    DEFAULT_BACKEND,
    WHISPERCPP_BINARIES,
    WhisperCpp,
    WhisperXCt2,
    preferred_backend,
    read_whispercpp_json,
)
from whisperx_local.engine.commands import recognition_signature
from whisperx_local.engine.stages import transcribe_chunk


class RegistryTest(unittest.TestCase):
    def test_the_default_is_the_fast_engine(self):
        # whisper.cpp on Metal measured ~9x faster than CTranslate2 on the same 300s.
        self.assertEqual(DEFAULT_BACKEND, "whispercpp")

    def test_linux_defaults_to_the_backend_installed_by_the_app(self):
        for architecture in ("x86_64", "aarch64"):
            with self.subTest(architecture=architecture), \
                 mock.patch.object(backends.platform, "machine", return_value=architecture):
                self.assertEqual(preferred_backend("Linux"), "whisperx")

    def test_macos_keeps_the_native_accelerated_default(self):
        self.assertEqual(preferred_backend("Darwin"), "whispercpp")

    def test_the_engine_that_owns_the_huggingface_cache_is_separate_from_the_default(self):
        # These were one constant, which meant "the engine that runs by default" and "the
        # engine whose models are checked by cache marker" could only ever agree.
        self.assertEqual(backends.HF_CACHED_BACKEND, "whisperx")
        self.assertNotEqual(backends.HF_CACHED_BACKEND, DEFAULT_BACKEND)

    def test_both_engines_are_offered(self):
        self.assertEqual(set(backends.keys()), {"whisperx", "whispercpp"})

    def test_nothing_chosen_means_the_default(self):
        self.assertEqual(backends.get(None).key, DEFAULT_BACKEND)

    def test_an_engine_is_found_by_name(self):
        self.assertEqual(backends.get("whispercpp").key, "whispercpp")

    def test_an_unknown_engine_is_refused_with_the_valid_names(self):
        # Falling back to the default would attribute a transcript to the wrong tool.
        with self.assertRaises(KeyError) as raised:
            backends.get("whisperx.cpp")
        self.assertIn("whisperx", str(raised.exception))
        self.assertIn("whispercpp", str(raised.exception))

    def test_a_backend_is_chosen_from_the_options(self):
        self.assertEqual(
            backends.for_options(SimpleNamespace(engine="whispercpp")).key, "whispercpp"
        )

    def test_options_without_an_engine_still_resolve(self):
        self.assertEqual(
            backends.for_options(SimpleNamespace()).key, DEFAULT_BACKEND
        )

    def test_every_backend_labels_itself(self):
        for key, backend in backends.BACKENDS.items():
            with self.subTest(engine=key):
                self.assertTrue(backend.label.strip())


class DefaultEngineUnchangedTest(unittest.TestCase):
    """The seam must be invisible in the default path."""

    def test_the_output_filename_is_the_one_the_command_writes(self):
        # This is the contract between the command and the reader. If it moved, every run
        # would find no output and report a missing chunk.
        backend = WhisperXCt2()
        self.assertEqual(
            backend.output_path(Path("/tmp/part-0000.wav"), Path("/tmp/asr-0000")),
            Path("/tmp/asr-0000/part-0000.json"),
        )

    def test_the_reader_is_the_same_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "part-0000.json"
            path.write_text(json.dumps({"segments": [{"start": 0, "end": 1, "text": "x"}]}), encoding="utf-8")
            self.assertEqual(WhisperXCt2().read(path), read_segments(path))

    def test_the_command_is_the_one_that_was_always_built(self):
        from whisperx_local.engine.commands import whisperx_command

        options = build_parser().parse_args(["run", "a.wav"])
        options.model = "large-v3"
        audio, output = Path("/tmp/a.wav"), Path("/tmp/out")
        self.assertEqual(
            WhisperXCt2().command(options, audio, output, "offline", 8),
            whisperx_command(options, audio, output, "offline", 8),
        )

    def test_the_default_engine_is_available_here(self):
        # If this fails, nothing in the project can run, so it is asserted rather than
        # discovered.
        self.assertTrue(WhisperXCt2().available())


class UnavailableEngineTest(unittest.TestCase):
    """A missing engine is reported before a run, not during one."""

    def test_without_the_binary_it_refuses_to_run(self):
        with mock.patch("shutil.which", return_value=None):
            self.assertFalse(WhisperCpp().available())

    def test_the_reason_names_what_is_missing_and_what_to_do(self):
        with mock.patch.object(backends.platform, "system", return_value="Darwin"):
            reason = WhisperCpp().missing_reason()
        self.assertIn("whisper-cli", reason)
        self.assertIn("ggml-large-v3.bin", reason)

    def test_the_linux_reason_does_not_recommend_homebrew(self):
        with mock.patch.object(backends.platform, "system", return_value="Linux"):
            reason = WhisperCpp().missing_reason()
        self.assertIn("build whisper.cpp", reason)
        self.assertIn("PATH", reason)
        self.assertNotIn("brew", reason)

    def test_with_the_binary_it_offers_itself(self):
        with mock.patch("shutil.which", return_value="/opt/homebrew/bin/whisper-cli"):
            self.assertTrue(WhisperCpp().available())

    def test_the_binary_is_looked_for_under_both_its_names(self):
        for name in ("whisper-cli", "whisper-cpp"):
            with self.subTest(binary=name):
                with mock.patch("shutil.which", side_effect=lambda c, path=None: f"/usr/bin/{c}" if c == name else None):
                    self.assertEqual(WhisperCpp().binary(), f"/usr/bin/{name}")

    def test_the_binary_is_found_when_the_shell_has_no_homebrew_on_its_path(self):
        # The failure that started this: a terminal without a login shell has no
        # /opt/homebrew/bin, and the app reported an installed engine as missing. The
        # search uses the managed PATH, exactly as the ffmpeg lookup already did.
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        brew = Path(temporary.name) / "bin"
        brew.mkdir()
        binary = brew / "whisper-cli"
        binary.write_text("#!/bin/sh\n", encoding="utf-8")
        binary.chmod(0o755)
        with mock.patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}):
            with mock.patch.object(backends, "managed_env", return_value={"PATH": str(brew)}):
                self.assertEqual(WhisperCpp().binary(), str(binary))

    def test_the_search_asks_the_managed_path_not_the_ambient_one(self):
        with mock.patch.object(backends, "managed_env", return_value={"PATH": "/nowhere"}):
            with mock.patch("shutil.which", return_value=None) as which:
                WhisperCpp().binary()
        which.assert_called_with(WHISPERCPP_BINARIES[-1], path="/nowhere")


class WhisperCppCommandTest(unittest.TestCase):
    """The command line, built from the documented flags."""

    def setUp(self):
        # An empty models directory, so a test cannot pass or fail because of whatever
        # happens to be installed on the machine running it.
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        patcher = mock.patch.object(backends, "WHISPERCPP_DIR", Path(temporary.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def options(self, **overrides) -> SimpleNamespace:
        base = dict(language="fa", model="large-v3", prompt=None)
        base.update(overrides)
        return SimpleNamespace(**base)

    def with_model(self, **overrides):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        model = Path(temporary.name) / "ggml-large-v3.bin"
        model.write_bytes(b"")
        return model, self.options(model=str(model), **overrides)

    def test_the_command_makes_the_directory_it_writes_into(self):
        # whisper.cpp does not create it, and does not fail when it is missing either: it
        # exits zero, writes nothing, and the pipeline reports a chunk it could not
        # transcribe -- after four retries that each reload the whole model.
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        model = Path(temporary.name) / "ggml-large-v3-turbo-q8_0.bin"
        model.write_bytes(b"")
        output_dir = Path(temporary.name) / "asr-0000"
        self.assertFalse(output_dir.exists())
        options = SimpleNamespace(language="fa", model=str(model), prompt=None, beam_size=8)
        with mock.patch("shutil.which", return_value="/opt/homebrew/bin/whisper-cli"):
            WhisperCpp().command(options, Path("/tmp/a.wav"), output_dir, "offline", 8)
        self.assertTrue(output_dir.is_dir())

    def test_a_missing_model_is_refused_by_name(self):
        with mock.patch("shutil.which", return_value="/usr/bin/whisper-cli"):
            with self.assertRaises(ValueError) as raised:
                WhisperCpp().command(self.options(), Path("/tmp/a.wav"), Path("/tmp/o"), "offline", 4)
        self.assertIn("ggml-large-v3.bin", str(raised.exception))

    def test_the_command_names_the_model_the_audio_and_the_language(self):
        model, options = self.with_model()
        with mock.patch("shutil.which", return_value="/usr/bin/whisper-cli"):
            command, env = WhisperCpp().command(options, Path("/tmp/a.wav"), Path("/tmp/o"), "offline", 6)
        self.assertIn(str(model), command)
        self.assertIn("/tmp/a.wav", command)
        self.assertIn("fa", command)
        self.assertIn("6", command)

    def test_it_asks_for_json_output(self):
        _, options = self.with_model()
        with mock.patch("shutil.which", return_value="/usr/bin/whisper-cli"):
            command, _ = WhisperCpp().command(options, Path("/tmp/a.wav"), Path("/tmp/o"), "offline", 4)
        self.assertIn("--output-json", command)
    def test_it_asks_the_engine_to_report_progress(self):
        # Without this nothing arrives until the run is over, and the progress bar sits
        # still for minutes, which reads as a hang.
        _, options = self.with_model()
        with mock.patch("shutil.which", return_value="/opt/homebrew/bin/whisper-cli"):
            command, _ = WhisperCpp().command(options, Path("/tmp/a.wav"), Path("/tmp/o"), "offline", 4)
        self.assertIn("--print-progress", command)

    def test_silence_is_trimmed_when_the_vad_model_is_installed(self):
        # Measured: it brought the two engines' words closer together, 35.8% apart against
        # 39.9% without, at no time cost.
        _, options = self.with_model()
        (backends.WHISPERCPP_DIR / "ggml-silero-v5.1.2.bin").write_bytes(b"vad")
        with mock.patch("shutil.which", return_value="/opt/homebrew/bin/whisper-cli"):
            command, _ = WhisperCpp().command(options, Path("/tmp/a.wav"), Path("/tmp/o"), "offline", 4)
        self.assertIn("--vad", command)
        self.assertIn("--vad-model", command)

    def test_the_newer_vad_model_name_is_accepted_too(self):
        _, options = self.with_model()
        (backends.WHISPERCPP_DIR / "ggml-silero-v6.2.0.bin").write_bytes(b"vad")
        with mock.patch("shutil.which", return_value="/opt/homebrew/bin/whisper-cli"):
            command, _ = WhisperCpp().command(options, Path("/tmp/a.wav"), Path("/tmp/o"), "offline", 4)
        self.assertIn("--vad", command)

    def test_no_vad_flag_when_its_model_is_absent(self):
        # Asked for a flag whose model is missing, whisper.cpp would refuse to run at all.
        _, options = self.with_model()
        with mock.patch("shutil.which", return_value="/opt/homebrew/bin/whisper-cli"):
            command, _ = WhisperCpp().command(options, Path("/tmp/a.wav"), Path("/tmp/o"), "offline", 4)
        self.assertNotIn("--vad", command)

    def test_the_absent_vad_model_is_reported_rather_than_silent(self):
        _, options = self.with_model()
        notes = WhisperCpp().notes(options)
        self.assertTrue(any("silence" in note for note in notes))

    def test_no_vad_note_once_the_model_is_there(self):
        _, options = self.with_model()
        (backends.WHISPERCPP_DIR / "ggml-silero-v5.1.2.bin").write_bytes(b"vad")
        notes = WhisperCpp().notes(options)
        self.assertFalse(any("silence" in note for note in notes))
    def test_a_context_prompt_is_passed_through_when_given(self):
        _, options = self.with_model(prompt="اشتراک")
        with mock.patch("shutil.which", return_value="/usr/bin/whisper-cli"):
            command, _ = WhisperCpp().command(options, Path("/tmp/a.wav"), Path("/tmp/o"), "offline", 4)
        self.assertIn("--prompt", command)
        self.assertIn("اشتراک", command)

    def test_no_prompt_means_no_prompt_flag(self):
        _, options = self.with_model()
        with mock.patch("shutil.which", return_value="/usr/bin/whisper-cli"):
            command, _ = WhisperCpp().command(options, Path("/tmp/a.wav"), Path("/tmp/o"), "offline", 4)
        self.assertNotIn("--prompt", command)

    def test_the_output_filename_is_what_is_read_back(self):
        # The two have to agree or a successful run reports a missing chunk.
        backend = WhisperCpp()
        self.assertEqual(
            backend.output_path(Path("/tmp/part.wav"), Path("/tmp/o")), Path("/tmp/o/part.json")
        )


class WhisperCppPayloadTest(unittest.TestCase):
    """Reading its output, from the documented shape and defensively around it."""

    def read(self, payload) -> list[dict]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.json"
            path.write_text(
                payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
            )
            return read_whispercpp_json(path)

    def test_offsets_in_milliseconds_become_seconds(self):
        segments = self.read(
            {"transcription": [{"offsets": {"from": 1500, "to": 4000}, "text": " سلام"}]}
        )
        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0]["start"], 1.5)
        self.assertAlmostEqual(segments[0]["end"], 4.0)

    def test_surrounding_whitespace_is_trimmed(self):
        segments = self.read(
            {"transcription": [{"offsets": {"from": 0, "to": 1000}, "text": "  سلام  "}]}
        )
        self.assertEqual(segments[0]["text"], "سلام")

    def test_the_clock_form_is_understood_with_a_comma(self):
        segments = self.read(
            {"transcription": [{"timestamps": {"from": "00:00:01,500", "to": "00:00:04,000"}, "text": "x"}]}
        )
        self.assertAlmostEqual(segments[0]["start"], 1.5)

    def test_the_clock_form_is_understood_with_a_dot(self):
        segments = self.read(
            {"transcription": [{"timestamps": {"from": "01:02:03.000", "to": "01:02:04.000"}, "text": "x"}]}
        )
        self.assertAlmostEqual(segments[0]["start"], 3723.0)

    def test_offsets_are_preferred_over_the_clock_when_both_are_present(self):
        segments = self.read(
            {
                "transcription": [
                    {
                        "offsets": {"from": 2000, "to": 3000},
                        "timestamps": {"from": "00:00:09,000", "to": "00:00:10,000"},
                        "text": "x",
                    }
                ]
            }
        )
        self.assertAlmostEqual(segments[0]["start"], 2.0)

    def test_an_empty_transcription_is_silence_not_a_failure(self):
        self.assertEqual(self.read({"transcription": []}), [])

    def test_entries_with_no_text_are_skipped(self):
        self.assertEqual(self.read({"transcription": [{"offsets": {"from": 0, "to": 1}, "text": "  "}]}), [])

    def test_an_entry_with_no_usable_time_is_skipped(self):
        self.assertEqual(self.read({"transcription": [{"text": "x"}]}), [])

    def test_a_shape_that_is_not_understood_is_refused_rather_than_read_as_silence(self):
        # The whole point: an adapter that cannot read its input must not look like a
        # recording with no speech in it.
        with self.assertRaises(ValueError) as raised:
            self.read({"result": {"language": "fa"}})
        self.assertIn("transcription", str(raised.exception))

    def test_a_payload_that_is_not_an_object_is_refused(self):
        with self.assertRaises(ValueError):
            self.read([{"start": 0}])

    def test_several_entries_keep_their_order(self):
        segments = self.read(
            {
                "transcription": [
                    {"offsets": {"from": 0, "to": 1000}, "text": "یک"},
                    {"offsets": {"from": 1000, "to": 2000}, "text": "دو"},
                ]
            }
        )
        self.assertEqual([entry["text"] for entry in segments], ["یک", "دو"])

    def test_a_non_numeric_offset_is_ignored_rather_than_crashing(self):
        self.assertEqual(
            self.read({"transcription": [{"offsets": {"from": "soon", "to": "later"}, "text": "x"}]}),
            [],
        )


class DispatchTest(unittest.TestCase):
    """The chosen engine is the one that actually runs.

    Asserted through the real ``transcribe_chunk`` rather than around it, because the point
    is that the seam is wired into the pipeline, not merely that it exists.
    """

    @dataclass
    class Recording:
        key: str = "fake"
        label: str = "Recording recogniser"
        segments: list = field(default_factory=list)
        commands: list = field(default_factory=list)
        reads: list = field(default_factory=list)

        def available(self):
            return True

        def missing_reason(self):
            return "never missing"

        def command(self, options, audio, output_dir, network, threads):
            self.commands.append({"audio": audio, "threads": threads})
            return ["fake-recogniser", str(audio)], {}

        def output_path(self, audio, output_dir):
            return output_dir / f"{audio.stem}.fake.json"

        def read(self, path):
            self.reads.append(path)
            return self.segments

    def run_stage(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        chunk_dir = Path(temporary.name)
        slice_path = chunk_dir / "part-0000.wav"
        slice_path.write_text("", encoding="utf-8")

        backend = self.Recording(segments=[{"start": 0.0, "end": 1.0, "text": "سلام"}])
        produced = backend.output_path(slice_path, chunk_dir / "asr-0000")
        produced.parent.mkdir(parents=True, exist_ok=True)
        produced.write_text("{}", encoding="utf-8")

        options = SimpleNamespace(engine="fake", retries=1, threads=8)
        studio = mock.Mock()
        studio.stream.return_value = 0

        with mock.patch.dict(backends.BACKENDS, {"fake": backend}):
            result = transcribe_chunk(options, studio, slice_path, chunk_dir, 0, "offline", 1, 1)
        return backend, studio, result

    def test_the_chosen_engine_builds_the_command(self):
        backend, _, _ = self.run_stage()
        self.assertEqual(len(backend.commands), 1)

    def test_the_chosen_engine_reads_the_output(self):
        backend, _, result = self.run_stage()
        self.assertEqual(len(backend.reads), 1)
        self.assertEqual(result, backend.segments)

    def test_the_command_that_ran_is_written_to_the_log(self):
        # The log is how a bad transcription is explained after the fact, so the exact
        # command has to be in it.
        _, studio, _ = self.run_stage()
        logged = "".join(call.args[0] for call in studio.log_line.call_args_list)
        self.assertIn("fake-recogniser", logged)

    def test_a_chunk_that_produces_nothing_is_reported_as_missing(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        chunk_dir = Path(temporary.name)
        slice_path = chunk_dir / "part-0000.wav"
        slice_path.write_text("", encoding="utf-8")

        backend = self.Recording()
        options = SimpleNamespace(engine="fake", retries=1, threads=8)
        studio = mock.Mock()
        studio.stream.return_value = 1  # the engine failed

        with mock.patch.dict(backends.BACKENDS, {"fake": backend}):
            result = transcribe_chunk(options, studio, slice_path, chunk_dir, 0, "offline", 1, 1)
        self.assertIsNone(result)


class SignatureTest(unittest.TestCase):
    """The engine is part of what produced the text, so it is part of the cache key."""

    def signature(self, *arguments):
        return recognition_signature(build_parser().parse_args(["run", "a.wav", *arguments]))

    def test_the_engine_is_part_of_the_signature(self):
        self.assertNotEqual(self.signature("--engine", "whispercpp"), self.signature("--engine", "whisperx"))

    def test_choosing_the_default_explicitly_is_the_same_as_not_choosing_it(self):
        # Two spellings of one behaviour must not throw away cached text.
        self.assertEqual(self.signature(), self.signature("--engine", "whispercpp"))

    def test_batching_has_one_spelling_for_its_default(self):
        # Same trap as the engine: None and 1 mean the same thing and must compare equal.
        self.assertEqual(self.signature(), self.signature("--chunks-per-call", "1"))

    def test_a_different_batch_size_is_a_different_signature(self):
        self.assertNotEqual(self.signature(), self.signature("--chunks-per-call", "2"))


class WhisperCppEnvironmentTest(unittest.TestCase):
    """What whisper.cpp needs around it to run at all.

    Both of these were found by running the real binary, and both would have made the
    backend fail on this machine while reading as correct in the source.
    """

    def options(self, **overrides):
        base = dict(language="fa", model="large-v3-turbo-q8_0", prompt=None, beam_size=8)
        base.update(overrides)
        return SimpleNamespace(**base)

    def command(self, **overrides):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        model = Path(temporary.name) / "ggml-large-v3-turbo-q8_0.bin"
        model.write_bytes(b"")
        options = self.options(model=str(model), **overrides)
        with mock.patch("shutil.which", return_value="/opt/homebrew/bin/whisper-cli"):
            return WhisperCpp().command(options, Path("/tmp/a.wav"), Path("/tmp/o"), "offline", 8)

    def test_the_environment_carries_the_variable_this_build_needs(self):
        # Without it, ggml asserts in its Metal residency-set teardown and the run dies
        # with SIGABRT half a second after loading the model.
        with mock.patch.object(backends.platform, "system", return_value="Darwin"):
            _, env = self.command()
        self.assertEqual(env[backends.NO_RESIDENCY_VARIABLE], "1")

    def test_linux_does_not_receive_the_metal_workaround(self):
        with mock.patch.object(backends.platform, "system", return_value="Linux"):
            env = WhisperCpp().environment("online")
        self.assertNotIn(backends.NO_RESIDENCY_VARIABLE, env)

    def test_the_environment_is_complete_rather_than_bare(self):
        # It is handed to Popen as the child's whole environment, so an empty dict would
        # strip PATH and HOME from the recogniser.
        _, env = self.command()
        self.assertTrue(env)
        self.assertIn("PATH", env)

    def test_an_offline_run_carries_the_cache_only_variables(self):
        _, env = self.command()
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")

    def test_an_online_run_does_not_pretend_to_be_offline(self):
        env = WhisperCpp().environment("online")
        self.assertNotIn("HF_HUB_OFFLINE", env)
        self.assertIn("PATH", env)

    def test_the_default_engine_takes_its_environment_from_its_own_command(self):
        # The other backend returns its environment alongside the command, so this file
        # must not assume both engines have an ``environment`` method.
        options = build_parser().parse_args(["run", "a.wav"])
        command, env = WhisperXCt2().command(options, Path("/tmp/a.wav"), Path("/tmp/o"), "offline", 8)
        self.assertTrue(env)
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")


class BeamCapTest(unittest.TestCase):
    """whisper.cpp allocates one decoder per beam and refuses to allocate ten."""

    def options(self, beam):
        return SimpleNamespace(
            language="fa", model="large-v3-turbo-q8_0", prompt=None, beam_size=beam
        )

    def test_the_cap_is_eight(self):
        self.assertEqual(backends.MAX_BEAM_SIZE, 8)

    def test_a_beam_above_the_cap_is_brought_down_to_it(self):
        self.assertEqual(WhisperCpp().beam_size(self.options(10)), 8)

    def test_a_beam_below_the_cap_is_passed_through_untouched(self):
        self.assertEqual(WhisperCpp().beam_size(self.options(5)), 5)

    def test_a_missing_beam_uses_the_cap(self):
        self.assertEqual(WhisperCpp().beam_size(SimpleNamespace()), 8)

    def test_the_cap_is_reported_rather_than_dropped_silently(self):
        notes = WhisperCpp().notes(self.options(10))
        self.assertTrue(notes)
        self.assertIn("beam 10", notes[0])

    def test_the_report_says_the_comparison_is_not_like_for_like(self):
        self.assertIn("not like for like", WhisperCpp().notes(self.options(10))[0])

    def test_no_note_when_the_beam_fits(self):
        self.assertEqual(WhisperCpp().notes(self.options(5)), [])

    def test_the_engine_with_no_cap_says_nothing(self):
        self.assertEqual(WhisperXCt2().notes(self.options(10)), [])

    def test_the_command_asks_for_the_capped_beam(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        model = Path(temporary.name) / "ggml-large-v3-turbo-q8_0.bin"
        model.write_bytes(b"")
        options = SimpleNamespace(language="fa", model=str(model), prompt=None, beam_size=10)
        with mock.patch("shutil.which", return_value="/opt/homebrew/bin/whisper-cli"):
            command, _ = WhisperCpp().command(
                options, Path("/tmp/a.wav"), Path("/tmp/o"), "offline", 8
            )
        self.assertIn("--beam-size", command)
        self.assertEqual(command[command.index("--beam-size") + 1], "8")
        self.assertEqual(command[command.index("--best-of") + 1], "8")


class AutomaticModelTest(unittest.TestCase):
    """Which GGML file ``automatic`` settles on.

    Picked from what is on disk, and quality first, so that changing the *engine* does
    not also quietly change the *model* -- the default transcript has always come from
    large-v3 weights.
    """

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        patcher = mock.patch.object(backends, "WHISPERCPP_DIR", self.directory)
        patcher.start()
        self.addCleanup(patcher.stop)

    def install(self, name: str) -> None:
        (self.directory / f"ggml-{name}.bin").write_bytes(b"weights")

    def options(self, model=None):
        return SimpleNamespace(model=model)

    def chosen(self, model=None) -> str | None:
        path = WhisperCpp().model_path(self.options(model))
        return path.name if path else None

    def test_with_nothing_installed_there_is_no_model(self):
        self.assertIsNone(self.chosen())

    def test_each_model_resolves_when_it_is_the_only_one_installed(self):
        for name in backends.MODEL_PREFERENCE:
            with self.subTest(model=name):
                # Each case starts from an empty directory, or an earlier model with a
                # higher preference would win and the case would prove nothing.
                for existing in self.directory.glob("*.bin"):
                    existing.unlink()
                self.install(name)
                self.assertEqual(self.chosen(), f"ggml-{name}.bin")

    def test_automatic_takes_the_only_model_that_is_there(self):
        self.install("large-v3-turbo-q8_0")
        self.assertEqual(self.chosen("automatic"), "ggml-large-v3-turbo-q8_0.bin")

    def test_quality_wins_when_both_are_installed(self):
        self.install("large-v3-turbo-q8_0")
        self.install("large-v3")
        self.assertEqual(self.chosen("automatic"), "ggml-large-v3.bin")

    def test_a_named_model_is_honoured_over_the_preference(self):
        self.install("large-v3")
        self.install("large-v3-turbo-q8_0")
        self.assertEqual(self.chosen("large-v3-turbo-q8_0"), "ggml-large-v3-turbo-q8_0.bin")

    def test_a_named_model_that_is_not_installed_falls_back_and_says_so(self):
        # Every profile names large-v3, so refusing would make the default engine
        # unusable on a machine that has only a turbo file. The substitute is reported.
        self.install("large-v3-turbo-q8_0")
        self.assertEqual(self.chosen("medium"), "ggml-large-v3-turbo-q8_0.bin")
        note = WhisperCpp().substituted(SimpleNamespace(model="medium"))
        self.assertIsNotNone(note)
        self.assertIn("medium", note)

    def test_the_note_says_the_words_will_differ(self):
        self.install("large-v3-turbo-q8_0")
        note = WhisperCpp().substituted(SimpleNamespace(model="large-v3"))
        self.assertIn("not be the ones", note)

    def test_the_exact_model_is_used_when_it_is_installed(self):
        self.install("large-v3")
        self.install("large-v3-turbo-q8_0")
        self.assertEqual(self.chosen("large-v3"), "ggml-large-v3.bin")
        self.assertIsNone(WhisperCpp().substituted(SimpleNamespace(model="large-v3")))

    def test_no_substitution_note_when_nothing_was_named(self):
        self.install("large-v3-turbo-q8_0")
        self.assertIsNone(WhisperCpp().substituted(SimpleNamespace(model=None)))
        self.assertIsNone(WhisperCpp().substituted(SimpleNamespace(model="automatic")))

    def test_the_substitution_reaches_the_run_as_a_note(self):
        # It has to arrive where the preflight prints it, or it is silent after all.
        # Asserted among the notes rather than as the only one: the missing-VAD note is
        # also expected here, and each is a separate disclosure.
        self.install("large-v3-turbo-q8_0")
        notes = WhisperCpp().notes(SimpleNamespace(model="large-v3", beam_size=8))
        self.assertTrue(any("large-v3" in note and "not installed" in note for note in notes))

    def test_the_model_reason_names_what_is_installed(self):
        self.install("large-v3-turbo-q8_0")
        self.assertIn("ggml-large-v3-turbo-q8_0.bin", WhisperCpp().model_reason())

    def test_a_path_to_a_model_wins_over_everything(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        explicit = Path(temporary.name) / "my-own.bin"
        explicit.write_bytes(b"weights")
        self.assertEqual(self.chosen(str(explicit)), "my-own.bin")

    def test_the_installed_list_names_what_is_actually_there(self):
        self.install("large-v3")
        self.install("large-v3-turbo-q8_0")
        self.assertEqual(
            [path.name for path in WhisperCpp().installed()],
            ["ggml-large-v3-turbo-q8_0.bin", "ggml-large-v3.bin"],
        )

    def test_an_empty_directory_lists_nothing(self):
        self.assertEqual(WhisperCpp().installed(), [])


if __name__ == "__main__":
    unittest.main()
