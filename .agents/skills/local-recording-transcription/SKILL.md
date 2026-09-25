---
name: local-recording-transcription
description: "Use when: transcribing one explicitly selected audio or video file with the local WhisperX application and validating its canonical transcript."
metadata:
  version: "1.0.0"
---

# Local Recording Transcription

## Required Input

Require an exact source path or an unambiguous user-selected file. Record source path, media kind, run time, command, canonical output path, requested stages, and result status.

## Procedure

1. Confirm the exact source exists and is readable.
2. Inspect current `./cli --help` or the application README before constructing a command.
3. Choose the least expensive plan that satisfies the request. For extraction work that does not need speakers or word alignment, request transcription only.
4. Preserve existing source and output artifacts. Use a new output destination when comparison or recovery requires it.
5. Run the process to completion without issuing another command into its terminal.
6. Verify the canonical transcript file exists, is non-empty, and covers the expected recording duration. Do not trust exit code or a completion banner alone.
7. Record gaps, partial coverage, missing speakers, and recognition uncertainty explicitly.
8. Hand the verified canonical transcript to the workspace recording-intake workflow; do not perform destination-specific publication here.

## Faithfulness

- Do not silently switch recordings, models, profiles, or existing transcripts.
- Do not use another transcript to fill gaps unless the user explicitly requests comparison or recovery.
- Do not guess unclear names, numbers, products, or requirements.
- Keep raw recognition separate from cleaned or summarized derivatives.
- Preserve timestamps when available.

## Failure Handling

Retain partial recognition when later optional stages fail. A missing or incomplete canonical transcript is not success even if the process exits zero. Report the verified coverage boundary and resumable state.
