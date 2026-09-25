"""Tool discovery, environment building, and layout arithmetic.

The ffmpeg failures matter most: they are the first thing a new user hits, and a
bare "bad CPU type" error from an Intel build on Apple Silicon is genuinely
unhelpful, so each path has to produce an explanation rather than a traceback.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from whisperx_local.paths import environment, layout, tools


class LayoutTest(unittest.TestCase):
    def test_the_app_root_is_the_directory_holding_this_package(self):
        # The folder may be called anything: it was renamed from ``whisperx`` to
        # ``Whisper`` and the app carried on, because the root is derived from where the
        # package sits rather than from a name. So this asserts the relationship.
        package = Path(layout.__file__).resolve().parents[1]
        self.assertEqual(package.name, "whisperx_local")
        self.assertEqual(layout.APP_DIR, package.parent.parent)
        self.assertIn(layout.APP_DIR / "src" / "whisperx_local", [package])

    def test_every_runtime_directory_sits_under_the_app(self):
        for directory in (
            layout.MODELS_DIR,
            layout.INPUT_DIR,
            layout.OUTPUT_DIR,
            layout.WORK_DIR,
            layout.LOG_DIR,
            layout.STATE_DIR,
        ):
            with self.subTest(directory=directory.name):
                self.assertIn(layout.APP_DIR, directory.parents)

    def test_the_interpreter_is_the_apps_own_venv(self):
        self.assertEqual(layout.PYTHON, layout.VENV_DIR / "bin" / "python")
        self.assertEqual(layout.WHISPERX, layout.VENV_DIR / "bin" / "whisperx")

    def test_preferences_live_outside_the_app(self):
        # So a reinstall, or a second checkout, keeps the user's saved defaults.
        self.assertNotIn(layout.APP_DIR, layout.SETTINGS_FILE.parents)


class ManagedEnvTest(unittest.TestCase):
    def test_caches_are_pinned_inside_the_app(self):
        env = environment.managed_env()
        self.assertEqual(env["HF_HOME"], str(layout.HUGGINGFACE_DIR))
        self.assertEqual(env["TORCH_HOME"], str(layout.TORCH_DIR))
        self.assertEqual(env["NLTK_DATA"], str(layout.NLTK_DATA_DIR))

    def test_output_is_unbuffered_so_progress_arrives_live(self):
        self.assertEqual(environment.managed_env()["PYTHONUNBUFFERED"], "1")

    def test_homebrew_is_prepended_when_arm_homebrew_exists(self):
        with mock.patch.object(environment.os.path, "isdir", return_value=True):
            env = environment.managed_env()
        self.assertTrue(env["PATH"].startswith("/opt/homebrew/bin" + os.pathsep))


class SupportedHostTest(unittest.TestCase):
    def test_ubuntu_supports_x86_64_and_aarch64_aliases(self):
        for machine, canonical in (
            ("x86_64", "x86_64"),
            ("amd64", "x86_64"),
            ("aarch64", "aarch64"),
            ("arm64", "aarch64"),
        ):
            with self.subTest(machine=machine), \
                 mock.patch.object(environment.platform, "system", return_value="Linux"), \
                 mock.patch.object(environment.platform, "machine", return_value=machine):
                self.assertEqual(environment.host_architecture(), canonical)
                self.assertTrue(environment.supported_host())

    def test_ubuntu_rejects_unsupported_architectures(self):
        with mock.patch.object(environment.platform, "system", return_value="Linux"), \
             mock.patch.object(environment.platform, "machine", return_value="i686"):
            self.assertFalse(environment.supported_host())


class LoadLocalEnvTest(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.env_file = Path(self._temporary.name) / ".env"
        self._previous = os.environ.pop("HF_TOKEN", None)

    def tearDown(self):
        os.environ.pop("HF_TOKEN", None)
        if self._previous is not None:
            os.environ["HF_TOKEN"] = self._previous
        self._temporary.cleanup()

    def load(self, text: str) -> None:
        self.env_file.write_text(text, encoding="utf-8")
        with mock.patch.object(environment, "APP_DIR", Path(self._temporary.name)):
            environment.load_local_env()

    def test_the_token_is_read(self):
        self.load("HF_TOKEN=hf_abc123\n")
        self.assertEqual(os.environ.get("HF_TOKEN"), "hf_abc123")

    def test_quotes_are_stripped(self):
        self.load('HF_TOKEN="hf_quoted"\n')
        self.assertEqual(os.environ.get("HF_TOKEN"), "hf_quoted")

    def test_comments_and_blank_lines_are_ignored(self):
        self.load("# a comment\n\nHF_TOKEN=hf_ok\n")
        self.assertEqual(os.environ.get("HF_TOKEN"), "hf_ok")

    def test_other_names_are_refused(self):
        # A stray file in the working tree must not be able to redirect model
        # downloads or change how another program behaves.
        self.load("HF_HOME=/tmp/elsewhere\nPATH=/tmp/evil\nHF_TOKEN=hf_ok\n")
        self.assertEqual(os.environ.get("HF_TOKEN"), "hf_ok")
        self.assertNotEqual(os.environ.get("HF_HOME"), "/tmp/elsewhere")

    def test_an_existing_variable_is_not_overwritten(self):
        os.environ["HF_TOKEN"] = "already-set"
        self.load("HF_TOKEN=from-file\n")
        self.assertEqual(os.environ.get("HF_TOKEN"), "already-set")

    def test_a_missing_file_is_harmless(self):
        with mock.patch.object(environment, "APP_DIR", Path(self._temporary.name)):
            environment.load_local_env()


class WorkingFfmpegTest(unittest.TestCase):
    def test_a_missing_ffmpeg_on_macos_explains_the_fix(self):
        with mock.patch.object(tools.shutil, "which", return_value=None), \
             mock.patch.object(tools.platform, "system", return_value="Darwin"):
            with self.assertRaises(SystemExit) as caught:
                tools.working_ffmpeg()
        self.assertIn("brew install ffmpeg", str(caught.exception))

    def test_a_missing_ffmpeg_on_ubuntu_explains_the_fix_on_both_architectures(self):
        for architecture in ("x86_64", "aarch64"):
            with self.subTest(architecture=architecture), \
                 mock.patch.object(tools.shutil, "which", return_value=None), \
                 mock.patch.object(tools.platform, "system", return_value="Linux"), \
                 mock.patch.object(tools.platform, "machine", return_value=architecture):
                with self.assertRaises(SystemExit) as caught:
                    tools.working_ffmpeg()
            self.assertIn("apt-get install ffmpeg", str(caught.exception))

    def test_a_usable_ffmpeg_is_returned(self):
        with mock.patch.object(tools.shutil, "which", return_value="/opt/homebrew/bin/ffmpeg"), \
             mock.patch.object(tools.subprocess, "run") as run:
            self.assertEqual(tools.working_ffmpeg(), "/opt/homebrew/bin/ffmpeg")
        run.assert_called_once()

    def test_an_intel_binary_gets_a_processor_specific_message(self):
        error = OSError()
        error.errno = 86  # EBADARCH
        with mock.patch.object(tools.shutil, "which", return_value="/usr/local/bin/ffmpeg"), \
             mock.patch.object(tools.subprocess, "run", side_effect=error):
            with self.assertRaises(SystemExit) as caught:
                tools.working_ffmpeg()
        message = str(caught.exception)
        self.assertIn("Intel (x86_64)", message)
        self.assertIn("Apple Silicon build", message)

    def test_a_broken_ffmpeg_reports_its_path(self):
        with mock.patch.object(tools.shutil, "which", return_value="/opt/homebrew/bin/ffmpeg"), \
             mock.patch.object(
                 tools.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")
             ):
            with self.assertRaises(SystemExit) as caught:
                tools.working_ffmpeg()
        self.assertIn("/opt/homebrew/bin/ffmpeg", str(caught.exception))


class WorkingFfprobeTest(unittest.TestCase):
    def test_the_sibling_of_ffmpeg_is_preferred(self):
        with tempfile.TemporaryDirectory() as directory:
            ffmpeg = Path(directory) / "ffmpeg"
            ffmpeg.write_text("", encoding="utf-8")
            probe = Path(directory) / "ffprobe"
            probe.write_text("", encoding="utf-8")
            self.assertEqual(tools.working_ffprobe(str(ffmpeg)), str(probe))

    def test_a_missing_ffprobe_is_reported(self):
        with mock.patch.object(tools.shutil, "which", return_value=None):
            with self.assertRaises(SystemExit) as caught:
                tools.working_ffprobe("/nowhere/ffmpeg")
        self.assertIn("ffprobe", str(caught.exception))


class PerformanceCoresTest(unittest.TestCase):
    def test_the_platform_count_is_used(self):
        probe = mock.Mock(stdout="8\n")
        with mock.patch.object(environment.subprocess, "run", return_value=probe), \
             mock.patch.object(environment.platform, "system", return_value="Darwin"):
            self.assertEqual(environment.performance_cores(), 8)

    def test_a_failure_falls_back_to_the_cpu_count(self):
        with mock.patch.object(
            environment.subprocess, "run", side_effect=OSError("no sysctl")
        ), mock.patch.object(environment.platform, "system", return_value="Darwin"):
            self.assertGreaterEqual(environment.performance_cores(), 1)


class AvailableMemoryTest(unittest.TestCase):
    def test_a_value_is_positive_when_available(self):
        value = environment.available_memory_bytes()
        if value is not None:
            self.assertGreater(value, 0)


if __name__ == "__main__":
    unittest.main()
