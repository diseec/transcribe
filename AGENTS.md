# WhisperX Application Agent Instructions

This application owns local transcription implementation, commands, output contracts, tests, and validated runtime lessons.

## Read Before Work

1. [`../../AGENTS.md`](../../AGENTS.md)
2. [`README.md`](README.md)
3. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) when changing architecture
4. the applicable skill under [`.agents/skills/`](.agents/skills/)

## Boundaries

- This app produces transcription artifacts; it does not decide whether content becomes a Todo, project Task, CRM note, HRM record, or another destination.
- Preserve canonical raw transcripts. Derived formatting, summaries, and domain extraction must never overwrite them.
- Verify output existence, non-zero content, and expected duration coverage independently of exit status.
- Never choose an input recording or alternate transcript merely because it is newest or easier to process.
- Do not interrupt a running transcription by reusing its terminal.
- Reusable app operation belongs in this scope. Cross-domain recording interpretation belongs to the workspace recording-intake skill and the destination domain.

## Validation

Run the narrowest relevant test. The full local suite is `./cli selftest`; it does not require network or model downloads.
