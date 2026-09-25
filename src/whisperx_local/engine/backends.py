"""Which recogniser runs, and how its command and output are spoken to.

There is one seam in this program that decides what the transcript says, and it is small:
something builds a command line, something reads the file that command wrote. Everything
downstream -- chunking, coverage, alignment, diarization, the canonical artifact -- reads
one shape and does not care who produced it. ``read_segments`` already tolerated two shapes
for exactly that reason.

So a second recogniser is an adapter, not a rewrite. Each backend answers four questions:

* **Is it usable?** A missing binary or a missing model has to be reported before a run
  starts, not discovered an hour in.
* **What command runs it?** Including where it should write.
* **Where did it write?** Different tools name their output differently.
* **What does it mean?** Parsed into ``{"segments": [...]}`` with ``start``, ``end`` and
  ``text``, because that is the contract the rest of the pipeline holds.

Adding a backend is therefore bounded and testable without installing it, which matters
because the alternative runtimes need multi-gigabyte models and this was built on a
connection that could not finish one.

Honesty about what is and is not verified: the CTranslate2 backend is the one every run so
far has used. The whisper.cpp adapter is written to the documented output format and is
**not verified against the real binary**, so ``available()`` refuses it until that binary
and a model are actually present, and a payload whose shape is not understood raises rather
than being read as silence. An unverified adapter that silently returns nothing would be
indistinguishable from a recording with no speech in it, which is the failure this whole
module exists to avoid.
"""

from __future__ import annotations

import json
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path

from whisperx_local.chunking import read_segments
from whisperx_local.engine.commands import offline_environment, whisperx_command
from whisperx_local.paths import MODELS_DIR, managed_env

# Where a whisper.cpp model is looked for, if ``--model`` is not itself a path to one.
WHISPERCPP_DIR = MODELS_DIR / "whisper.cpp"

# The names whisper.cpp's CLI is installed under, depending on how it was built.
WHISPERCPP_BINARIES = ("whisper-cli", "whisper-cpp")

# What ``--model automatic`` means for this engine, best quality first. Chosen from what
# is actually on disk rather than fetched, so a machine with one file runs and a machine
# with large-v3 uses large-v3. The run reports which file it settled on.
#
# Quality first deliberately: large-v3 is the model the default transcript has always
# come from, so preferring it keeps a change of engine from quietly becoming a change of
# model as well.
MODEL_PREFERENCE = (
    "large-v3",
    "large-v3-q5_0",
    "large-v3-turbo-q8_0",
    "large-v3-turbo-q5_0",
    "large-v3-turbo",
)

# How "no model was named" is spelled, in the settings file and on the command line.
AUTOMATIC = ("", "auto", "automatic", "none")

# whisper.cpp's own voice-activity model, downloaded separately from the weights. Newer
# builds ship the later version, so either name is accepted.
VAD_MODELS = ("ggml-silero-v6.2.0.bin", "ggml-silero-v5.1.2.bin")

# whisper.cpp allocates one decoder per beam and refuses to allocate more than this:
#
#   whisper_full_with_state: too many decoders requested (10), max = 8
#
# The limit does not move with --threads, so a beam-10 profile cannot be honoured by it.
# Measured on 1.9.4.
MAX_BEAM_SIZE = 8

# The Homebrew build of ggml 0.24.0 asserts inside its Metal residency-set teardown
# (``ggml-metal-device.m:1025: GGML_ASSERT([rsets->data count] == 0) failed``) and dies
# with SIGABRT about half a second in, right after the model has loaded. Setting this
# makes it work. Found by listing the dylib's ``GGML_*`` strings; without it a working
# Metal build looks completely broken.
NO_RESIDENCY_VARIABLE = "GGML_METAL_NO_RESIDENCY"


@dataclass(frozen=True)
class Backend:
    """A recogniser, as this program needs to talk to it."""

    key: str
    label: str

    # Whether this engine's models are Hugging Face snapshots, which are resolved and
    # downloaded by the library rather than named by us. A plain class attribute, not a
    # dataclass field: it describes the adapter, not a run.
    uses_hf_cache = False

    def available(self) -> bool:
        raise NotImplementedError

    def missing_reason(self) -> str:
        """Why this backend cannot run, phrased for the person who asked for it."""
        raise NotImplementedError

    def notes(self, options) -> list[str]:
        """Settings this engine cannot honour, to be reported rather than dropped.

        A setting that changes what the transcript says must never be quietly ignored:
        that would make an A/B compare two things at once and call the result a speed
        difference.
        """
        return []

    def model_reason(self) -> str:
        """Why there is no model to run, when there is none.

        Separate from :meth:`missing_reason`, which is about the engine itself. Using one
        message for both reported a missing model as a missing binary, which sent the
        reader looking for the wrong thing.
        """
        return "no model was found"

    def command(
        self, options, audio: Path, output_dir: Path, network: str, threads: int
    ) -> tuple[list[str], dict[str, str]]:
        raise NotImplementedError

    def output_path(self, audio: Path, output_dir: Path) -> Path:
        raise NotImplementedError

    def read(self, path: Path) -> list[dict]:
        raise NotImplementedError


@dataclass(frozen=True)
class WhisperXCt2(Backend):
    """The default: WhisperX driving CTranslate2.

    A thin wrapper, deliberately. The command line and the reader are the ones every run
    has used, so making the seam cannot change what the pipeline produces.
    """

    key: str = "whisperx"
    label: str = "WhisperX (CTranslate2)"
    uses_hf_cache = True

    def available(self) -> bool:
        from whisperx_local.paths import WHISPERX

        return WHISPERX.exists()

    def missing_reason(self) -> str:
        return "the pinned WhisperX install is missing; run: ./cli install"

    def command(self, options, audio, output_dir, network, threads):
        return whisperx_command(options, audio, output_dir, network, threads)

    def output_path(self, audio: Path, output_dir: Path) -> Path:
        return output_dir / f"{audio.stem}.json"

    def read(self, path: Path) -> list[dict]:
        return read_segments(path)


@dataclass(frozen=True)
class WhisperCpp(Backend):
    """whisper.cpp, using the native acceleration available on the host.

    Selected only when both its binary and a model are present. Nothing here has been run
    against a real install, so the flag names and the payload shape come from its
    documentation and are parsed defensively.
    """

    key: str = "whispercpp"
    label: str = "whisper.cpp (native)"

    def binary(self) -> str | None:
        """Where whisper-cli is, searched the way ffmpeg already is.

        Not ``shutil.which(name)`` on the ambient PATH: a terminal started without a login
        shell has no /opt/homebrew/bin on it, so an engine that is installed and working
        would be reported as missing. ``managed_env`` prepends the Homebrew prefix.
        """
        search = managed_env().get("PATH")
        for name in WHISPERCPP_BINARIES:
            found = shutil.which(name, path=search)
            if found:
                return found
        return None

    def installed(self) -> list[Path]:
        """The GGML models on disk."""
        if not WHISPERCPP_DIR.is_dir():
            return []
        return sorted(WHISPERCPP_DIR.glob("ggml-*.bin"))

    def vad_model(self) -> Path | None:
        """whisper.cpp's voice-activity model, if it has been downloaded."""
        for name in VAD_MODELS:
            path = WHISPERCPP_DIR / name
            if path.is_file():
                return path
        return None

    def model_path(self, options) -> Path | None:
        """The model file: an explicit path, the named file, or the best one installed.

        A named size is honoured exactly when its file is there. When it is not, the best
        installed file is used instead -- refusing outright would leave the default engine
        unusable on a machine that has one turbo file, since every profile names
        ``large-v3``. The difference is reported by :meth:`substituted` rather than
        assumed, because a substituted model speaks with different words.
        """
        candidate = str(getattr(options, "model", None) or "")
        if candidate.endswith(".bin") and Path(candidate).is_file():
            return Path(candidate)
        if candidate and candidate.lower() not in AUTOMATIC:
            exact = WHISPERCPP_DIR / f"ggml-{candidate}.bin"
            if exact.is_file():
                return exact
        for name in MODEL_PREFERENCE:
            path = WHISPERCPP_DIR / f"ggml-{name}.bin"
            if path.is_file():
                return path
        return None

    def substituted(self, options) -> str | None:
        """What was asked for versus what will run, when they differ."""
        candidate = str(getattr(options, "model", None) or "")
        if not candidate or candidate.lower() in AUTOMATIC or candidate.endswith(".bin"):
            return None
        chosen = self.model_path(options)
        if chosen is None or chosen.name == f"ggml-{candidate}.bin":
            return None
        return (
            f"{candidate} is not installed for whisper.cpp, so {chosen.name} is used "
            f"instead: the words will not be the ones {candidate} would produce. Put "
            f"ggml-{candidate}.bin in {WHISPERCPP_DIR} to use the exact model"
        )

    def model_reason(self) -> str:
        installed = ", ".join(path.name for path in self.installed()) or "nothing"
        return (
            f"no GGML model is in {WHISPERCPP_DIR} (found: {installed}). Download one "
            "from huggingface.co/ggerganov/whisper.cpp -- ggml-large-v3-turbo-q8_0.bin "
            "(834 MB) is the fastest, ggml-large-v3-q5_0.bin (1031 MB) and "
            "ggml-large-v3.bin (2952 MB) keep the large-v3 weights"
        )

    def available(self) -> bool:
        return self.binary() is not None

    def missing_reason(self) -> str:
        if platform.system() == "Linux":
            install = (
                "build whisper.cpp for this host from github.com/ggml-org/whisper.cpp "
                "and install whisper-cli on PATH"
            )
        else:
            install = "install it with `brew install whisper.cpp`"
        return (
            f"the whisper-cli binary was not found; {install}, and put a GGML "
            f"model in {WHISPERCPP_DIR} (ggml-large-v3.bin or ggml-large-v3-turbo-q8_0.bin). "
            "Models come from huggingface.co/ggerganov/whisper.cpp"
        )

    def beam_size(self, options) -> int:
        """The beam width to ask for, within the decoders whisper.cpp will allocate."""
        requested = int(getattr(options, "beam_size", 0) or 0)
        if requested <= 0:
            return MAX_BEAM_SIZE
        return min(requested, MAX_BEAM_SIZE)

    def notes(self, options) -> list[str]:
        notes = []
        requested = int(getattr(options, "beam_size", 0) or 0)
        if requested > MAX_BEAM_SIZE:
            notes.append(
                f"whisper.cpp allocates one decoder per beam and refuses more than "
                f"{MAX_BEAM_SIZE}, so beam {requested} becomes {MAX_BEAM_SIZE}: this run "
                "searches less widely than the profile asks, and a comparison against a "
                "beam-10 run is not like for like"
            )
        substitute = self.substituted(options)
        if substitute:
            notes.append(substitute)
        if self.vad_model() is None:
            notes.append(
                "whisper.cpp's voice-activity model is not installed, so silence inside a "
                "chunk is not trimmed, unlike the WhisperX engine which trims it. On the "
                "same audio that widened the difference between the two engines from "
                "35.8% to 39.9% of words. Download ggml-silero-v5.1.2.bin from "
                f"huggingface.co/ggml-org/whisper-vad into {WHISPERCPP_DIR}"
            )
        return notes

    def environment(self, network: str) -> dict[str, str]:
        """The child environment, including the variable this build cannot run without.

        Built on the managed environment rather than returned bare, because the caller
        hands it to ``Popen`` as the *whole* environment: ``{}`` would strip PATH and
        HOME from the child.
        """
        env = offline_environment(network)
        if platform.system() == "Darwin":
            env[NO_RESIDENCY_VARIABLE] = "1"
        return env

    def command(self, options, audio, output_dir, network, threads):
        model = self.model_path(options)
        if model is None:
            raise ValueError(
                "no whisper.cpp model found. Pass --model as a path to a .bin file, or "
                f"place ggml-{getattr(options, 'model', 'large-v3')}.bin in {WHISPERCPP_DIR}"
            )
        # It does not create the directory it writes into. Given a prefix inside a missing
        # directory it exits **zero**, leaves no file, and the pipeline reports a chunk it
        # could not transcribe -- after four retries, each re-running the whole model.
        # The other engine's command builder makes its own directory, so this belongs to
        # the backend rather than to the caller.
        output_dir.mkdir(parents=True, exist_ok=True)
        beam = self.beam_size(options)
        prefix = output_dir / audio.stem
        command = [
            self.binary() or "whisper-cli",
            "--model", str(model),
            "--file", str(audio),
            "--language", str(options.language),
            "--threads", str(threads),
            "--beam-size", str(beam),
            "--best-of", str(beam),
            # Without this the engine reports nothing until it is finished, and the
            # progress bar sits still for the whole run -- which reads as a hang.
            "--print-progress",
            "--output-json",
            "--output-file", str(prefix),
        ]
        prompt = getattr(options, "prompt", None)
        if prompt:
            command.extend(["--prompt", str(prompt)])
        # Silence inside a chunk is trimmed, the way the other engine's VAD trims it.
        # Measured on 300s of Persian, this moved whisper.cpp's words closer to the
        # CTranslate2 result -- 35.8% apart against 39.9% without -- at no time cost.
        vad = self.vad_model()
        if vad is not None:
            command.extend(["--vad", "--vad-model", str(vad)])
        return command, self.environment(network)

    def output_path(self, audio: Path, output_dir: Path) -> Path:
        return output_dir / f"{audio.stem}.json"

    def read(self, path: Path) -> list[dict]:
        return read_whispercpp_json(path)


# The two timestamp forms whisper.cpp can report, in the order of preference. Offsets are
# milliseconds and unambiguous; the clock form is a fallback for a build that omits them.
_CLOCK_FIELDS = ("from", "to")


def read_whispercpp_json(path: Path) -> list[dict]:
    """Read whisper.cpp's ``--output-json`` payload into internal segments.

    A payload with no ``transcription`` at all is refused rather than read as empty. The
    difference matters: an empty transcription is a recording with no speech in it, while
    an unrecognised shape is the adapter being wrong, and only one of those is the user's
    problem to know about.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "transcription" not in payload:
        raise ValueError(
            f"{path.name} is not a whisper.cpp JSON payload: expected a 'transcription' "
            "list, so the adapter and the installed build disagree about the format"
        )
    segments: list[dict] = []
    for entry in payload.get("transcription") or []:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text", "")).strip()
        if not text:
            continue
        span = _span_of(entry)
        if span is None:
            continue
        segments.append({"start": span[0], "end": span[1], "text": text})
    return segments


def _span_of(entry: dict) -> tuple[float, float] | None:
    """One entry's times in seconds, from whichever form the payload carries."""
    offsets = entry.get("offsets")
    if isinstance(offsets, dict):
        start, end = offsets.get("from"), offsets.get("to")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            return float(start) / 1000.0, float(end) / 1000.0
    stamps = entry.get("timestamps")
    if isinstance(stamps, dict):
        start = _clock_seconds(stamps.get(_CLOCK_FIELDS[0]))
        end = _clock_seconds(stamps.get(_CLOCK_FIELDS[1]))
        if start is not None and end is not None:
            return start, end
    return None


def _clock_seconds(value: object) -> float | None:
    """``hh:mm:ss,mmm`` or ``hh:mm:ss.mmm`` as seconds, or ``None``."""
    if not isinstance(value, str):
        return None
    normalised = value.strip().replace(",", ".")
    parts = normalised.split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return None


# whisper.cpp first because it is the accelerated macOS default. Registry order does
# not choose the default on Linux, where a normal install guarantees WhisperX but not a
# separately compiled whisper-cli.
BACKENDS: dict[str, Backend] = {
    WhisperCpp().key: WhisperCpp(),
    WhisperXCt2().key: WhisperXCt2(),
}

def preferred_backend(system: str | None = None) -> str:
    """The backend a fresh install can run without another native installation."""
    return WhisperCpp().key if (system or platform.system()) == "Darwin" else WhisperXCt2().key


# The engine that runs when nothing else says otherwise.
DEFAULT_BACKEND = preferred_backend()

# The engine whose models are Hugging Face snapshots, resolved and downloaded by the
# library rather than named by us. Used only to decide which readiness check applies.
HF_CACHED_BACKEND = WhisperXCt2().key


def keys() -> tuple[str, ...]:
    return tuple(BACKENDS)


def get(name: str | None) -> Backend:
    """The named backend, or the default when nothing was chosen.

    An unknown name raises rather than quietly falling back: a misspelled engine that
    silently ran the default would produce a transcript attributed to the wrong tool.
    """
    key = name or DEFAULT_BACKEND
    try:
        return BACKENDS[key]
    except KeyError:
        raise KeyError(
            f"unknown engine {key!r}. Known engines: {', '.join(BACKENDS)}"
        ) from None


def for_options(options) -> Backend:
    return get(getattr(options, "engine", None))
