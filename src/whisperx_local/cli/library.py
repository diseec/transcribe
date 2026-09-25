"""Finding recordings: turning what a user typed into media files."""

from __future__ import annotations

from pathlib import Path

from whisperx_local.paths import DOWNLOAD_DIR, INPUT_DIR

# Containers ffmpeg can pull audio out of. Video is included on purpose: screen
# recordings are a normal input for this tool, not a special case.
MEDIA_SUFFIXES = frozenset(
    {
        ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".3gp", ".mpg", ".mpeg",
        ".m4a", ".mp3", ".wav", ".flac", ".ogg", ".oga", ".opus", ".aac", ".wma",
    }
)


def is_media(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES


def discover(directory: Path, *, recursive: bool = True) -> list[Path]:
    """Media files in a directory, newest first, ignoring dotfiles."""
    if not directory.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    found = [
        path
        for path in directory.glob(pattern)
        if is_media(path) and not path.name.startswith(".")
    ]
    return sorted(found, key=lambda path: path.stat().st_mtime, reverse=True)


def expand_paths(arguments) -> list[Path]:
    """Expand typed arguments into media files.

    A file is taken as given even if its suffix is unfamiliar, because the user named
    it explicitly; a directory is searched. Order is preserved so a batch runs in the
    order the user listed it.
    """
    found: list[Path] = []
    for argument in arguments:
        path = Path(argument).expanduser()
        if path.is_dir():
            found.extend(discover(path))
        elif path.is_file():
            found.append(path.resolve())
    return found


def default_locations() -> list[Path]:
    """Where the picker looks without being told: the app's input and Downloads."""
    return [directory for directory in (INPUT_DIR, DOWNLOAD_DIR) if directory.is_dir()]


def candidates() -> list[Path]:
    """Everything worth offering in the picker, deduplicated and newest first."""
    seen: dict[Path, float] = {}
    for directory in default_locations():
        for path in discover(directory):
            seen[path] = path.stat().st_mtime
    return sorted(seen, key=lambda path: seen[path], reverse=True)


def presentable(path: Path, *, seconds: float | None = None) -> str:
    """A one-line description of a file for a list, with its length when known."""
    size = path.stat().st_size / 1024**2
    stamp = ""
    if seconds:
        minutes, remainder = divmod(int(seconds), 60)
        stamp = f"  {minutes}m{remainder:02d}s"
    return f"{path.name}{stamp}  ({size:.1f} MB)"
