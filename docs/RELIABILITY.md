# Reliability and Recovery

Transcribe is designed around long-running work where a partial result must never look like
a complete transcript.

## Durable Stages

Recordings are divided at pauses when possible. Boundaries and completed chunks are stored
as they are produced, so rerunning the same request reuses finished work. Recognition,
alignment, speaker labeling, and rendering remain separate stages; a later failure does not
erase canonical recognized text.

Changing a recognition-sensitive option invalidates text that can no longer be trusted while
retaining reusable prepared audio. The signature includes the engine, model, language,
decoding, prompt, hotwords, VAD, normalization, and batching settings.

## Failure Isolation

- failed recognition chunks retry with reduced resource pressure;
- alignment and diarization run in child processes;
- raw transcript files are written before optional presentation stages;
- unsupported backend output raises instead of being interpreted as silence;
- impossible action plans are rejected before execution.

## Coverage Integrity

Coverage is computed from productive chunk intervals rather than the last timestamp. Missing
ranges are named explicitly. A partial result is marked `INCOMPLETE`, has no completion
marker, and exits with status `2`; a run with no usable output exits with status `3`.

That contract matters for both people and automation: a clean-looking but shortened
transcript cannot silently pass as success.

## Observable Progress

The terminal dashboard reports stages and monotonic progress. Timing estimates learn from
completed runs on the current machine and profile. Redirected output uses plain progress
lines instead of terminal control sequences, and child commands are logged for diagnosis.

## Validation

The repository contains 852 automated tests across planning, chunking, configuration,
engines, media, paths, progress, services, rendering, comparison, UI, and integration
boundaries. Run the complete offline suite with:

```bash
./cli selftest
```

The suite validates contracts such as resume, targeted cache invalidation, output coverage,
completion markers, backend parsing, renderer consistency, and cross-module planning.
