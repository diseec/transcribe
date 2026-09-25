"""Local, observable transcription for Persian and English meetings.

Everything runs on this machine. The package is layered so that pure decisions
(chunk planning, progress maths, transcript assembly, settings resolution) never
touch a subprocess or a terminal, and can therefore be tested without either.
"""

__all__ = ["__version__"]

__version__ = "0.2.0"
