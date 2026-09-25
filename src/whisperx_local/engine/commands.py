"""Building the commands handed to the recogniser and its stages.

Alignment and diarization are deliberately excluded from the recognition command.
Both run as their own retryable child processes so a failure in either cannot destroy
an expensive recognition result that has already been paid for.
"""

from __future__ import annotations

from pathlib import Path

from whisperx_local.paths import MODELS_DIR, PYTHON, WHISPERX, managed_env

# Settings that change the transcript text. Used as a cache key, so anything listed
# here invalidates cached chunks when it changes, and anything left out deliberately
# does not.
SIGNATURE_FIELDS = (
    "model",
    "language",
    "compute_type",
    "beam_size",
    "best_of",
    "patience",
    "batch_size",
    "vad_method",
    "vad_onset",
    "vad_offset",
    "chunk_size",
    "compression_ratio_threshold",
    "logprob_threshold",
    "no_speech_threshold",
    "hotwords",
    "prompt",
    "normalize",
    # A different recogniser is unambiguously a different transcript, so it invalidates
    # cached text like any other recognition setting.
    "engine",
    # Batching chunks into one recogniser call is meant to be text-neutral, and mostly is:
    # the model still reads the same audio in the same 30-second windows. It is listed
    # anyway because a longer input can move where the voice-activity detector decides
    # speech begins, and a setting that *might* change the text has to invalidate cached
    # text when it changes. The alternative is reusing words produced under a different
    # arrangement and reporting them as current.
    "chunks_per_call",
)


def recognition_signature(options) -> dict[str, object]:
    """The subset of options that changes the recognised text."""
    return {name: getattr(options, name, None) for name in SIGNATURE_FIELDS}


def offline_environment(network: str) -> dict[str, str]:
    """Environment for a child process, restricted to cache when asked."""
    env = managed_env()
    if network == "offline":
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    return env


# The child-process stages. Kept as a module path rather than a script path so the
# worker is importable, testable and versioned with the package it belongs to.
WORKER_MODULE = "whisperx_local.engine.worker"


def managed_worker_command(stage: str, *arguments: object, network: str = "auto") -> list[str]:
    """A child-process stage command, with the model directory always pinned."""
    command = [str(PYTHON), "-m", WORKER_MODULE, stage]
    command.extend(str(argument) for argument in arguments)
    command.extend(["--model-dir", str(MODELS_DIR)])
    if network == "offline":
        command.append("--cache-only")
    return command


def whisperx_command(
    options,
    source_audio: Path,
    output_dir: Path,
    network: str,
    threads: int,
) -> tuple[list[str], dict[str, str]]:
    """The recognition command for a single chunk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        str(WHISPERX),
        str(source_audio),
        "--model", options.model,
        "--language", options.language,
        "--device", "cpu",
        "--compute_type", options.compute_type,
        "--batch_size", str(options.batch_size),
        "--threads", str(threads),
        "--vad_method", options.vad_method,
        "--model_dir", str(MODELS_DIR),
        "--output_dir", str(output_dir),
        "--output_format", "json",
        "--vad_onset", str(options.vad_onset),
        "--vad_offset", str(options.vad_offset),
        "--chunk_size", str(options.chunk_size),
        "--beam_size", str(options.beam_size),
        "--best_of", str(options.best_of),
        "--patience", str(options.patience),
        "--compression_ratio_threshold", str(options.compression_ratio_threshold),
        "--logprob_threshold", str(options.logprob_threshold),
        "--no_speech_threshold", str(options.no_speech_threshold),
        # Alignment is a separate stage: a native abort there must not cost the text.
        "--no_align",
        "--print_progress", "True",
        "--verbose", "True",
    ]
    if options.hotwords:
        command.extend(["--hotwords", options.hotwords])
    if options.prompt:
        command.extend(["--initial_prompt", options.prompt])
    env = offline_environment(network)
    if network == "offline":
        command.extend(["--model_cache_only", "True"])
    return command, env
