"""Running ffmpeg with machine-readable progress.

ffmpeg writes progress to stdout and diagnostics to stderr. Both streams must be
consumed continuously: if either fills its pipe buffer, ffmpeg blocks forever
mid-run, which looks exactly like a hang. stderr is therefore drained on its own
thread rather than being read after the process exits.
"""

from __future__ import annotations

import re
import subprocess
import threading

_OUT_TIME = re.compile(r"out_time=(\d+):(\d\d):(\d\d)(?:\.(\d+))?")


def parse_progress_line(line: str, duration: float) -> float | None:
    """Turn one ``-progress`` line into a percentage, or None if it carries none.

    ``out_time`` is parsed as HH:MM:SS.ffffff rather than ``out_time_ms``, which
    despite its name is reported in microseconds on some builds.
    """
    match = _OUT_TIME.search(line)
    if match is None or duration <= 0:
        return None
    hours, minutes, seconds, fraction = match.groups()
    elapsed = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    if fraction:
        elapsed += float(f"0.{fraction}")
    return max(0.0, min(100.0, elapsed / duration * 100.0))


def run_ffmpeg_progress(
    ffmpeg: str,
    arguments: list[str],
    *,
    env: dict[str, str],
    duration: float | None = None,
    on_progress=None,
    on_line=None,
    log=None,
) -> str:
    """Run ffmpeg with progress reporting, returning its stderr.

    ``on_progress`` receives percentages when ``duration`` is known. ``on_line`` sees
    every stdout line, which is where filter metadata such as the loudness scan
    arrives. stderr is returned because silence detection has to parse it.
    """
    command = [ffmpeg, "-hide_banner", "-nostats", "-progress", "pipe:1", *arguments]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    captured: list[str] = []

    def drain() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            captured.append(line)

    reader = threading.Thread(target=drain, daemon=True, name="ffmpeg-stderr")
    reader.start()

    assert process.stdout is not None
    for line in process.stdout:
        if on_line is not None:
            on_line(line)
        if on_progress is not None and duration:
            percent = parse_progress_line(line, duration)
            if percent is not None:
                on_progress(percent)

    return_code = process.wait()
    reader.join(timeout=10)
    stderr_text = "".join(captured)
    if log is not None:
        log.write(stderr_text)
        log.flush()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return stderr_text
