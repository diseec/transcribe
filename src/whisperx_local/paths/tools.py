"""External tool discovery.

A wrong ffmpeg is the single most confusing failure this app can produce -- an Intel
build on Apple Silicon fails with a bare "bad CPU type", which says nothing about
the cause -- so each failure mode here carries its own explanation.
"""

from __future__ import annotations

import errno
import platform
import shutil
import subprocess
from pathlib import Path

from whisperx_local.paths.environment import managed_env
from whisperx_local.paths.layout import PYTHON

INTEL_BINARY_HINT = (
    "ffmpeg is an Intel (x86_64) binary but this Mac is {machine}. "
    "Install the Apple Silicon build with /opt/homebrew/bin/brew install ffmpeg."
)


def ffmpeg_install_hint() -> str:
    """The native package command for the current operating system."""
    if platform.system() == "Linux":
        return "install it with: sudo apt-get update && sudo apt-get install ffmpeg"
    if platform.system() == "Darwin":
        return "install it with: brew install ffmpeg"
    return "install ffmpeg and make sure it is available on PATH"


def working_ffmpeg() -> str:
    """The ffmpeg to use, or a SystemExit explaining how to fix it."""
    path = shutil.which("ffmpeg", path=managed_env().get("PATH"))
    if path is None:
        raise SystemExit(f"ffmpeg is not installed; {ffmpeg_install_hint()}.")
    try:
        subprocess.run(
            [path, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except OSError as error:
        if error.errno == errno.EBADARCH:
            raise SystemExit(
                INTEL_BINARY_HINT.format(machine=platform.machine())
            ) from error
        raise SystemExit(f"ffmpeg at {path} cannot run: {error}") from error
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"ffmpeg at {path} failed its startup check.") from error
    return path


def working_ffprobe(ffmpeg: str) -> str:
    """ffprobe from the same install as ffmpeg, so the two cannot disagree."""
    sibling = Path(ffmpeg).with_name("ffprobe")
    if sibling.is_file():
        return str(sibling)
    found = shutil.which("ffprobe", path=managed_env().get("PATH"))
    if found is None:
        raise SystemExit("ffprobe is required to measure audio length but was not found.")
    return found


def nltk_resource_available() -> bool:
    """Whether the tokenizer used during alignment is present.

    Checked in a child process because importing nltk into the parent would load
    several hundred megabytes for a yes/no answer.
    """
    if not PYTHON.exists():
        return False
    check = subprocess.run(
        [str(PYTHON), "-c", "import nltk; nltk.data.find('tokenizers/punkt_tab')"],
        env=managed_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return check.returncode == 0
