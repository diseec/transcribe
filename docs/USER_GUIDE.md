# User Guide

This guide covers the everyday Transcribe workflows. For implementation boundaries, see
[Architecture](ARCHITECTURE.md). For runtimes and model setup, see
[Engines and Models](ENGINES_AND_MODELS.md).

## Setup

Transcribe requires Python 3.10-3.13 and FFmpeg. Create the environment on the machine
where it will run; native packages in `.venv` cannot be copied between operating systems
or CPU architectures.

```bash
./cli install
./cli resources
./cli status
```

The first command installs the application dependencies. `resources` downloads the
alignment tokenizer. Recognition models are acquired separately or on first use, depending
on the selected engine and network mode.

Speaker labels require a Hugging Face read token and access to the gated diarization model.
Copy `.env.example` to `.env`, add the token, and never commit that file.

## Choose a Workflow

### Guided menu

Run `./cli` with no arguments. The menu finds recordings, offers a workload and quality
profile, previews the plan, and asks for confirmation. It supports multiple files and falls
back to numbered prompts when output is redirected.

### One path in, one transcript out

```bash
./cli /path/to/recording.m4a
./cli transcribe recording-1.m4a recording-2.mov --language en
```

This mode is designed for scripts and agents. It ignores saved preferences and pins its
settings so a personal configuration cannot silently alter an automated run. It chooses
whisper.cpp when the binary and model are ready, otherwise announces a WhisperX fallback.

The language option accepts a language code supported by the underlying recognizer.
Persian and English receive the complete validated UI, settings, and alignment workflow;
other languages are an advanced recognition path and may require compatible alignment
resources.

### Configurable run

```bash
./cli run recording.m4a --profile accurate
./cli run recording.m4a --model large-v3 --output-format all
./cli run recording.m4a --dry-run
```

Use `run` for explicit control over engine, model, language, decoding, voice detection,
chunking, speakers, retries, network behavior, and output.

### Composer

`./cli composer` is a free-form workspace for changing files, actions, settings, and
presets in any order. Use `plan` before `run` to inspect the resolved work.

## Build the Pipeline You Need

The available actions are `prepare`, `boundaries`, `transcribe`, `align`, `diarize`,
`export`, and `analyze`. The planner adds required dependencies and rejects impossible
combinations before work starts.

```bash
./cli run recording.m4a --only export --output-format all
./cli run recording.m4a --only diarize
./cli run recording.m4a --only analyze
./cli run recording.m4a --without diarize
```

This makes it possible to re-render cached recognition, refine an existing transcript, or
run analysis without repeating expensive stages.

## Profiles and Settings

| Profile    | Best for                              | Tradeoff                        |
| ---------- | ------------------------------------- | ------------------------------- |
| `fast`     | First passes and high-throughput work | Narrow decoding search          |
| `balanced` | Everyday transcription                | Default balance                 |
| `accurate` | Important transcripts                 | Wider search and longer runtime |
| `maximum`  | Investigating difficult passages      | Full precision and highest cost |

Profiles set sensible groups of model, precision, search, batching, VAD, and thread
options. Explicit flags still win. Under memory pressure, Transcribe lowers batch size,
thread count, and chunk duration to reduce the chance of a native crash.

Saved defaults and named presets are available through `./cli config` and `./cli preset`.
Resolution order is built-ins, preferences, preset, then explicit overrides and flags.

## Outputs

Canonical `<name>.raw.txt` and `<name>.raw.json` files are written before optional
refinement or formatting. Readable exports include:

- plain text (`.txt`);
- SubRip and WebVTT subtitles (`.srt`, `.vtt`);
- tab-separated data (`.tsv`);
- structured JSON (`.json`);
- Audacity label text (`.aud`).

`--output-format all` writes every readable format. By default, a convenient copy is also
placed beside the source recording unless the source already lives in `input/`.

## Compare Transcripts

```bash
./cli compare output/reference.raw.json output/candidate.raw.json
```

Comparison aligns by time, reports substitutions, insertions and deletions, identifies the
worst windows, and can track domain terms. Persian-aware normalization avoids counting
common codepoint and joiner differences as recognition errors. Without human ground truth,
the result measures disagreement, not absolute accuracy.

## Network Modes

`--network auto` permits required downloads when resources are missing. `offline` refuses
uncached models instead of trying the network. After dependencies, tokenizers, and selected
models are cached, recognition itself does not require a hosted transcription service.
