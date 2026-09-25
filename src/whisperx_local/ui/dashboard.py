"""The live dashboard.

Two rules make the display trustworthy, and both exist because the first version
broke them:

* Progress is owned by :class:`whisperx_local.progress.model.ProgressModel`, which
  gives each stage a slice of one global bar. The dashboard used to give every stage
  its own 0-100% sweep, so the bar filled and snapped back six times per run.
* The display is repainted by a ticker thread, not only when work reports. Without
  it the bar froze for minutes during model loading and native alignment.

Rich is confined to this module: everything else deals in percentages and text.
"""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text

from whisperx_local.progress.model import ProgressModel
from whisperx_local.progress.timing import TimingStore
from whisperx_local.ui.console import clock

# Both engines' progress lines. WhisperX and the workers print ``Progress: 42.5%``;
# whisper.cpp prints ``whisper_print_progress_callback: progress =  42%`` -- padded, and
# only every 5%, so the bar moves in visible steps rather than continuously.
PROGRESS_RE = re.compile(r"progress\s*[:=]\s*([0-9.]+)%", re.IGNORECASE)
TRANSCRIPT_RE = re.compile(r"Transcript:\s*(.*)")
# whisper.cpp prints segments as ``[00:00:00.000 --> 00:00:03.560]  text``, with no
# ``Transcript:`` prefix at all, so without this the live excerpt stays empty for the
# whole run even though the words are there.
SEGMENT_RE = re.compile(r"^\s*\[([0-9:.]+)\s*-->\s*([0-9:.]+)\]\s*(\S.*)$")
TIMING_RE = re.compile(r"\[([0-9.]+)\s*-->\s*([0-9.]+)\]")
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# How long a stretch with no report before the activity line admits it. Silence is
# not a fault, but unexplained silence is what makes a run feel hung.
IDLE_NOTICE_SECONDS = 6.0

# How often a non-interactive run prints a progress line. Redirected output cannot
# animate a bar, and a run with no output at all is indistinguishable from a hang.
PLAIN_INTERVAL_SECONDS = 10.0


class AccuracyStudio:
    """Renders one run: a global bar, a stage checklist, and a live excerpt."""

    # Repaint interval. Fast enough to look continuous, slow enough not to fight the
    # child processes for the terminal.
    TICK = 0.1

    def __init__(
        self,
        *,
        output: Path,
        log_path: Path,
        expected: dict[str, float] | None = None,
        timing: TimingStore | None = None,
        context: str | None = None,
        audio_seconds: float = 0.0,
    ) -> None:
        self.console = Console()
        self.output = output
        self.log_path = log_path
        # Redirected output gets plain lines instead of a bar: a bar cannot animate
        # into a file, and an unobservable run is the thing users complain about.
        self._plain = not self.console.is_terminal
        self._last_announce = 0.0
        self.model = ProgressModel(expected=expected)
        self.timing = timing
        self.context = context
        self.audio_seconds = audio_seconds
        self.current_stage = self.model.stages[0].label
        self.excerpts: deque[str] = deque(maxlen=4)
        self.words = 0
        self.speakers = 0
        self.status = ""
        self.chunk_position = 1
        self.chunk_total = 1
        self.chunk_offset = 0.0
        self.started = time.monotonic()
        self._labels = {stage.label: stage.key for stage in self.model.stages}
        self._keys = {stage.key for stage in self.model.stages}
        self._started_stage: str | None = None
        self._stage_started = self.started
        self._last_advance = self.started
        # Reentrant on purpose. ``_refresh`` paints while holding the lock and
        # ``render`` takes it again to update the bar, so a plain Lock deadlocks the
        # ticker thread and freezes the run at the first repaint.
        self._lock = threading.RLock()
        self._stopped = threading.Event()
        self._ticker: threading.Thread | None = None
        self._live: Live | None = None
        self._log = None
        # Notices are queued rather than printed: anything written while the live
        # region is active can be erased by the next repaint, which is how the
        # memory-tuning notes used to disappear without ever being read.
        self._pending: list[str] = []
        self._failure_code: int | None = None
        self.partial_ready = False
        self.progress = Progress(
            TextColumn("  [bold cyan]{task.description:<26}"),
            BarColumn(bar_width=None, complete_style="cyan", finished_style="green"),
            TaskProgressColumn(),
            TextColumn("[dim]{task.fields[eta]}[/dim]"),
            expand=True,
        )
        self.task = self.progress.add_task(self.current_stage, total=100, eta="")

    # -- lifecycle --------------------------------------------------------
    def __enter__(self) -> "AccuracyStudio":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w", encoding="utf-8")
        self._live = Live(self.render(), console=self.console, transient=True)
        self._live.__enter__()
        self._start_ticker()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_ticker()
        if self._live is not None:
            self._live.__exit__(exc_type, exc, tb)
            self._live = None
        if self._log is not None:
            self._log.close()
            self._log = None
        self._flush_notices()

    def _flush_notices(self) -> None:
        """Show queued warnings and the failure panel now the live region is gone."""
        for text in self._pending:
            self.console.print(f"[yellow]![/] {text}")
        self._pending.clear()
        if self._failure_code is not None:
            self.console.print(self._failure_panel(self._failure_code))

    def _start_ticker(self) -> None:
        self._stopped.clear()
        self._ticker = threading.Thread(
            target=self._tick, name="studio-ticker", daemon=True
        )
        self._ticker.start()

    def _stop_ticker(self) -> None:
        self._stopped.set()
        if self._ticker is not None:
            self._ticker.join(timeout=1.0)
            self._ticker = None

    def _tick(self) -> None:
        """Repaint on a timer, so time-based progress shows while work is silent."""
        while not self._stopped.wait(self.TICK):
            try:
                if self._plain:
                    self._announce()
                else:
                    self._refresh()
            except Exception:  # noqa: BLE001 - a paint failure must not kill the run
                return

    def _refresh(self) -> None:
        if self._live is None or self._plain:
            return
        with self._lock:
            try:
                self._live.update(self.render())
            except Exception:  # noqa: BLE001 - see _tick
                pass

    def _announce(self, *, force: bool = False) -> None:
        """Print one progress line, rate-limited unless a stage just changed."""
        now = time.monotonic()
        if not force and now - self._last_announce < PLAIN_INTERVAL_SECONDS:
            return
        self._last_announce = now
        eta = self.model.eta()
        row = f"  [{clock(self.model.elapsed())}] {self.model.percent():5.1f}%  {self._label()}"
        if eta and eta >= 12.0:
            row += f"  ·  about {clock(self._round_eta(eta))} left"
        self.console.print(row, markup=False)

    def silence(self) -> "_Silence":
        """Divert in-process library noise into the diagnostic log."""
        return _Silence(self)

    def log_handle(self):
        """Open diagnostic log, for child processes that should write there."""
        return self._log

    def log_line(self, line: str) -> None:
        """Append a diagnostic line without showing it to the user."""
        if not line.endswith("\n"):
            line += "\n"
        if self._log is not None:
            self._log.write(line)
            self._log.flush()

    # -- stage control ----------------------------------------------------
    def begin_stage(self, name: str) -> None:
        """Enter a stage once, closing the previous one.

        Call this around a *phase*, not around each unit of work inside it. The
        per-chunk loop used to re-enter stages, which made the bar jump between
        "Transcribing" and "Aligning words" on every chunk.
        """
        key = self._labels.get(name, name)
        if key not in self._keys:
            return
        if (
            self._started_stage is not None
            and self._started_stage != key
            and not self.model.is_done(self._started_stage)
        ):
            self.model.complete()
        self._started_stage = key
        self.model.begin(key)
        self.current_stage = name
        self.status = ""
        # The previous stage's chunk count is meaningless here. Leaving it made the
        # label read "Aligning words 3/3" before that stage had started a single part.
        self.chunk_position = 1
        self.chunk_total = 1
        self._stage_started = time.monotonic()
        self._last_advance = self._stage_started
        self._refresh()
        if self._plain:
            self._announce(force=True)

    def stage(self, name: str) -> None:
        """Alias for :meth:`begin_stage`, so older call sites keep working."""
        self.begin_stage(name)

    def end_stage(self) -> None:
        """Close the running stage, so its duration can be learned from."""
        if self._started_stage is not None and not self.model.is_done(self._started_stage):
            self.model.complete()
        self._started_stage = None
        self._refresh()

    def skip_stage(self, name: str) -> None:
        """Mark a stage as not requested, so the bar never waits on it."""
        key = self._labels.get(name, name)
        if key in self._keys:
            self.model.skip(key)
            self._refresh()

    def set_chunks(self, position: int, total: int, offset: float = 0.0) -> None:
        """Declare which unit of the running stage is in flight."""
        self.chunk_position = position
        self.chunk_total = total
        self.chunk_offset = offset
        self.model.set_units(position, total)
        self._refresh()

    def progress_to(self, percent: float) -> None:
        """Report progress within the running stage, or within its running unit."""
        self.model.report(percent)
        self._last_advance = time.monotonic()
        self._refresh()

    def set_status(self, text: str) -> None:
        self.status = text
        self._refresh()

    def set_speakers(self, speakers: int) -> None:
        self.speakers = speakers
        self._refresh()

    def set_words(self, count: int) -> None:
        """Set the recovered-word total, e.g. after a resumed run that streamed nothing."""
        self.words = count
        self._refresh()

    def warn(self, text: str) -> None:
        """Queue a warning for display once the dashboard has closed."""
        self._pending.append(text)

    def mark_partial(self, ready: bool = True) -> None:
        """Record that the transcript on disk already holds the recognised text."""
        self.partial_ready = ready

    def finish(self) -> None:
        """Close the run: end the last stage, then release anything left over."""
        self.end_stage()
        for stage in self.model.stages:
            if not self.model.is_done(stage.key):
                self.model.skip(stage.key)
        self.chunk_total = 1
        self._learn_timings()
        self._refresh()
        if self._plain:
            self._announce(force=True)

    def _learn_timings(self) -> None:
        """Persist measured stage durations so the next run can predict them."""
        if self.timing is None or not self.context or self.audio_seconds <= 0:
            return
        measured = self.model.measured()
        if measured:
            self.timing.record(
                self.context, audio_seconds=self.audio_seconds, measured=measured
            )

    # -- rendering --------------------------------------------------------
    def _label(self) -> str:
        if self.chunk_total > 1:
            return f"{self.current_stage} {self.chunk_position}/{self.chunk_total}"
        return self.current_stage

    @staticmethod
    def _round_eta(seconds: float) -> float:
        """Coarsen an estimate so the shown value does not jitter every second."""
        if seconds < 90.0:
            return max(15.0, round(seconds / 15.0) * 15.0)
        return max(120.0, round(seconds / 60.0) * 60.0)

    def _eta_text(self) -> str:
        eta = self.model.eta()
        # Withheld until it means something: an estimate of a few seconds at the very
        # start is noise, and showing it makes the tool look unreliable.
        if not eta or eta < 12.0:
            return ""
        return f"~{clock(self._round_eta(eta))} left"

    def _absolute(self, excerpt: str) -> str:
        """Rewrite chunk-local timings onto the recording's own timeline."""
        if not self.chunk_offset:
            return excerpt

        def replace(match: re.Match[str]) -> str:
            start = float(match.group(1)) + self.chunk_offset
            end = float(match.group(2)) + self.chunk_offset
            return f"[{start:.2f} --> {end:.2f}]"

        return TIMING_RE.sub(replace, excerpt, count=1)

    def _activity_line(self) -> Text:
        """A live line proving work is happening even without a percentage."""
        now = time.monotonic()
        frame = SPINNER[int(now * 8) % len(SPINNER)]
        step_seconds = now - self._stage_started
        idle = now - self._last_advance
        detail = self.status or self.current_stage
        message = f"  {frame} {detail}  ·  {step_seconds:.0f}s on this step"
        if idle > IDLE_NOTICE_SECONDS:
            message += f"  ·  quiet for {idle:.0f}s"
        return Text(message, style="dim")

    def _stage_table(self) -> Table:
        table = Table.grid(expand=True)
        for _ in range(3):
            table.add_column(ratio=1)
        running = self.model.current.key
        for start in range(0, len(self.model.stages), 3):
            cells = []
            for stage in self.model.stages[start:start + 3]:
                if self.model.is_done(stage.key):
                    cells.append(f"[green]✓[/] [dim]{stage.label}[/]")
                elif stage.key == running:
                    cells.append(f"[bold cyan]● {stage.label}[/]")
                else:
                    cells.append(f"[dim]○ {stage.label}[/]")
            table.add_row(*cells)
        return table

    def _header(self) -> Text:
        parts: list[tuple[str, str]] = [
            (" WHISPERX ", "bold black on cyan"),
            ("  ACCURACY STUDIO", "bold white"),
            (f"    {clock(self.model.elapsed())}", "dim"),
        ]
        total = self.model.predicted_total()
        if total:
            parts.append((f" of ~{clock(total)}", "dim"))
        return Text.assemble(*parts)

    def render(self) -> Group:
        with self._lock:
            self.progress.update(
                self.task,
                description=self._label(),
                completed=self.model.percent(),
                eta=self._eta_text(),
            )
        preview = "\n".join(self.excerpts) if self.excerpts else (self.status or "Working…")
        title = "[bold]Recovered speech[/]"
        if self.words:
            title += f"  [dim]{self.words} words[/]"
        return Group(
            self._header(),
            Panel(self._stage_table(), border_style="bright_black", padding=(0, 1)),
            self.progress,
            self._activity_line(),
            Panel(preview, title=title, border_style="cyan", padding=(0, 1)),
            Text(f"  Output  {self.output}", style="dim"),
        )

    # -- engine output ----------------------------------------------------
    def absorb(self, line: str) -> None:
        """Fold one line of engine output into progress and the live excerpt.

        Stage changes are deliberately *not* inferred from the text here: the
        pipeline states them explicitly, and guessing made the bar hop between
        stages as soon as an unrelated line happened to mention one.
        """
        if match := PROGRESS_RE.search(line):
            self.model.report(float(match.group(1)))
            self._last_advance = time.monotonic()
        excerpt = self._excerpt_of(line)
        if excerpt:
            self.excerpts.append(self._absolute(excerpt))
            text = re.sub(r"^\[[^]]+\]\s*", "", excerpt)
            self.words += len(text.split())
        self._refresh()

    @staticmethod
    def _excerpt_of(line: str) -> str | None:
        """The recognised text a line carries, if it carries any.

        Two shapes, because the engines differ: WhisperX and the workers prefix it with
        ``Transcript:``, and whisper.cpp prints a bare timestamped line.
        """
        if match := TRANSCRIPT_RE.search(line):
            return match.group(1).strip() or None
        if match := SEGMENT_RE.match(line):
            return f"[{match.group(1)} --> {match.group(2)}] {match.group(3).strip()}"
        return None

    def stream(self, command: list[str], *, env: dict[str, str], check: bool = True) -> int:
        """Run a child process, hiding its noise behind the dashboard.

        With ``check=False`` a failure is returned instead of raised, so the
        caller can retry a chunk without alarming the user.
        """
        self.log_line("$ " + " ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            if self._log is not None:
                self._log.write(line)
                self._log.flush()
            self.absorb(line)
        return_code = process.wait()
        if return_code and check:
            self.failure(return_code)
            raise subprocess.CalledProcessError(return_code, command)
        return return_code

    def print_result(self) -> None:
        elapsed = time.monotonic() - self.started
        result = Table.grid(padding=(0, 2))
        result.add_column(style="dim")
        result.add_column(style="bold")
        result.add_row("Transcript", str(self.output))
        result.add_row("Recovered", f"{self.words} words")
        if self.speakers:
            result.add_row("Speakers", str(self.speakers))
        result.add_row("Elapsed", f"{elapsed:.1f} seconds")
        result.add_row("Diagnostics", str(self.log_path))
        self.console.print(
            Panel(result, title="[bold green]✓ Transcript ready[/]", border_style="green")
        )

    def failure(self, return_code: int) -> None:
        """Record a failure, to be shown once the dashboard has closed."""
        self._failure_code = return_code

    def _failure_panel(self, return_code: int) -> Panel:
        parts: list[tuple[str, str]] = [
            ("The transcription engine stopped before the run was complete.\n", "bold red"),
            (f"It exited with code {return_code}.\n", "white"),
        ]
        if self.partial_ready:
            parts.extend(
                [
                    ("The recognised text is already saved at:\n", "white"),
                    (f"{self.output}\n", "bold"),
                    (
                        "It was written before speaker separation, so it holds the text "
                        "without speaker labels. Re-run the same command to finish the rest.\n",
                        "white",
                    ),
                ]
            )
        else:
            parts.append(
                ("Partial work is saved; run the same command again to resume.\n", "white")
            )
        parts.extend([("Technical details: ", "dim"), (str(self.log_path), "bold")])
        return Panel(
            Text.assemble(*parts),
            title="[bold red]Run incomplete[/]",
            border_style="red",
        )


class _Silence:
    """Redirect stdout and stderr into the studio's diagnostic log."""

    def __init__(self, studio: "AccuracyStudio") -> None:
        self._studio = studio
        self._stdout = None
        self._stderr = None

    def __enter__(self) -> "_Silence":
        self._stdout, self._stderr = sys.stdout, sys.stderr
        target = self._studio._log
        if target is not None:
            sys.stdout = target
            sys.stderr = target
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._stdout is not None:
            sys.stdout = self._stdout
        if self._stderr is not None:
            sys.stderr = self._stderr
        return False
