"""The environment child processes run in, plus host facts that affect tuning.

Everything here answers "what does this machine look like" or "what should a child
process see". Both are guesswork-prone, so both are isolated from the pipeline and
covered by tests.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys

from whisperx_local.paths.layout import (
    APP_DIR,
    HUGGINGFACE_DIR,
    NLTK_DATA_DIR,
    TORCH_DIR,
    VENV_DIR,
    WHISPERX,
)

SUPPORTED_PYTHON = (3, 10) <= sys.version_info[:2] < (3, 14)

ARCHITECTURE_ALIASES = {
    "amd64": "x86_64",
    "x86_64": "x86_64",
    "arm64": "aarch64",
    "aarch64": "aarch64",
}
SUPPORTED_LINUX_ARCHITECTURES = frozenset({"x86_64", "aarch64"})

# Only this name may come from .env. Reading arbitrary variables out of a file in
# the working tree would let a stray file redirect model downloads or worse.
ALLOWED_ENV_NAMES = frozenset({"HF_TOKEN"})


def host_architecture() -> str:
    """The canonical architecture name used by support checks and status output."""
    machine = platform.machine().strip().lower()
    return ARCHITECTURE_ALIASES.get(machine, machine or "unknown")


def supported_host() -> bool:
    """Whether this operating system and CPU architecture are supported."""
    system = platform.system()
    if system == "Linux":
        return host_architecture() in SUPPORTED_LINUX_ARCHITECTURES
    return system == "Darwin"


def load_local_env() -> None:
    """Read the ignored .env file, allowing only the Hugging Face token."""
    env_file = APP_DIR / ".env"
    if not env_file.is_file():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if name not in ALLOWED_ENV_NAMES or name in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[name] = value


def managed_env() -> dict[str, str]:
    """Environment for every child process, with caches and CA certs pinned.

    Pinning the caches keeps model downloads inside the app directory, so a purge is
    a directory removal rather than a hunt through the home directory.
    """
    env = os.environ.copy()
    arm_homebrew = "/opt/homebrew/bin"
    if os.path.isdir(arm_homebrew):
        env["PATH"] = f"{arm_homebrew}{os.pathsep}{env.get('PATH', '')}"
    certificate = next(VENV_DIR.glob("lib/python*/site-packages/certifi/cacert.pem"), None)
    env.update(
        {
            "HF_HOME": str(HUGGINGFACE_DIR),
            "HF_HUB_DISABLE_PROGRESS_BARS": "0",
            "NLTK_DATA": str(NLTK_DATA_DIR),
            "TORCH_HOME": str(TORCH_DIR),
            "PYTHONUNBUFFERED": "1",
        }
    )
    if certificate:
        env["SSL_CERT_FILE"] = str(certificate)
        env["REQUESTS_CA_BUNDLE"] = str(certificate)
    return env


def require_install() -> None:
    if not WHISPERX.exists():
        raise SystemExit(f"WhisperX is not installed. Run: {APP_DIR / 'cli'} install")


def available_memory_bytes() -> int | None:
    """Physical memory the OS could hand over without heavy paging.

    Returns None when it cannot be determined, which callers must treat as "unknown"
    rather than "none": guessing zero memory would disable every optimisation.
    """
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_AVPHYS_PAGES")
        if page > 0 and pages > 0:
            return int(page) * int(pages)
    except (AttributeError, OSError, ValueError):
        pass
    if platform.system() != "Darwin":
        return None
    try:
        probe = subprocess.run(
            ["/usr/bin/vm_stat"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    # macOS counts reclaimable pages separately, and free pages alone badly
    # under-reports what is actually available.
    page_match = re.search(r"page size of (\d+) bytes", probe.stdout)
    free_match = re.search(r"Pages free:\s+(\d+)\.", probe.stdout)
    inactive_match = re.search(r"Pages inactive:\s+(\d+)\.", probe.stdout)
    if not (page_match and free_match):
        return None
    size = int(page_match.group(1))
    free = int(free_match.group(1))
    inactive = int(inactive_match.group(1)) if inactive_match else 0
    return size * (free + inactive)


def performance_cores() -> int:
    """Performance cores only; efficiency cores slow batched inference down.

    Handing torch all cores makes it schedule work onto the slow ones, which
    measures slower than using the fast cores alone.
    """
    if platform.system() == "Darwin":
        try:
            probe = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "hw.perflevel0.logicalcpu"],
                capture_output=True,
                text=True,
                check=True,
            )
            value = int(probe.stdout.strip())
            if value > 0:
                return value
        except (OSError, subprocess.CalledProcessError, ValueError):
            pass
    return max(1, os.cpu_count() or 1)
