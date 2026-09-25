"""Command construction and the cache signature.

The signature is a cache key: a field missing from it means a changed setting silently
reuses chunks produced by the old one, and a field wrongly added to it means needless
recomputation. Both directions are checked.
"""

import argparse
import unittest
from pathlib import Path

from whisperx_local.engine import commands, profiles
from whisperx_local.engine.tests.test_profiles import options
from whisperx_local.paths import GIB


class SignatureTest(unittest.TestCase):
    def test_the_signature_covers_settings_that_change_the_text(self):
        signature = commands.recognition_signature(options(model="large-v3", hotwords="a,b"))
        for name in ("model", "language", "beam_size", "hotwords", "prompt", "normalize"):
            self.assertIn(name, signature)

    def test_a_changed_setting_changes_the_signature(self):
        first = commands.recognition_signature(options(beam_size=1))
        second = commands.recognition_signature(options(beam_size=5))
        self.assertNotEqual(first, second)

    def test_threads_are_deliberately_absent(self):
        # Thread count cannot affect the text, so including it would throw away good
        # cached chunks for no reason.
        self.assertNotIn("threads", commands.recognition_signature(options(threads=8)))

    def test_an_option_the_namespace_lacks_does_not_raise(self):
        signature = commands.recognition_signature(argparse.Namespace(language="fa"))
        self.assertIsNone(signature["model"])


class WhisperxCommandTest(unittest.TestCase):
    def build(self, network="auto", **overrides):
        # Profile first, exactly as the app does, so the command is built from filled
        # options rather than from the raw "nothing chosen yet" state.
        tuned = options(**overrides)
        profiles.apply(tuned, available=8 * GIB)
        return commands.whisperx_command(
            tuned, Path("chunk.wav"), Path("/tmp/out"), network, threads=tuned.threads
        )

    def test_alignment_is_disabled_in_the_recognition_pass(self):
        # Alignment is a separate retryable stage; doing it here would put the text at
        # risk of the abort that motivated splitting them.
        command, _ = self.build()
        self.assertIn("--no_align", command)

    def test_json_output_is_requested(self):
        command, _ = self.build()
        self.assertIn("--output_format", command)
        self.assertIn("json", command)

    def test_batching_is_batched(self):
        command, _ = self.build()
        index = command.index("--batch_size")
        self.assertGreater(int(command[index + 1]), 1)

    def test_progress_printing_is_requested(self):
        # Without this the parent sees no Progress lines and the bar cannot move.
        command, _ = self.build()
        index = command.index("--print_progress")
        self.assertEqual(command[index + 1], "True")

    def test_hotwords_and_prompt_are_forwarded_when_given(self):
        command, _ = self.build(hotwords="apollo,revops", prompt="a sales call")
        self.assertIn("--hotwords", command)
        self.assertIn("apollo,revops", command)
        self.assertIn("--initial_prompt", command)

    def test_they_are_omitted_when_empty(self):
        command, _ = self.build(hotwords=None, prompt=None)
        self.assertNotIn("--hotwords", command)
        self.assertNotIn("--initial_prompt", command)

    def test_offline_mode_forbids_downloads(self):
        command, env = self.build(network="offline")
        self.assertIn("--model_cache_only", command)
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")

    def test_online_mode_leaves_the_cache_policy_alone(self):
        _, env = self.build(network="auto")
        self.assertNotIn("HF_HUB_OFFLINE", env)


class WorkerCommandTest(unittest.TestCase):
    def test_the_worker_is_reached_as_a_module(self):
        # A module path keeps the worker importable and testable, unlike a script path.
        command = commands.managed_worker_command("align", "--segments", "a.json")
        self.assertIn("-m", command)
        self.assertIn(commands.WORKER_MODULE, command)

    def test_the_model_directory_is_always_pinned(self):
        command = commands.managed_worker_command("diarize", "--audio", "a.wav")
        self.assertIn("--model-dir", command)

    def test_offline_is_forwarded_as_a_flag(self):
        command = commands.managed_worker_command("align", network="offline")
        self.assertIn("--cache-only", command)

    def test_arguments_are_stringified(self):
        command = commands.managed_worker_command("diarize", "--min-speakers", 2)
        self.assertIn("2", command)


if __name__ == "__main__":
    unittest.main()
