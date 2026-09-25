<div align="center">

# Transcribe

### Turn recordings into useful text. Keep every word under your control.

Local-first transcription for audio and video, built for quick one-off results,
long-running jobs, and repeatable automation.

[![Python 3.10-3.13](https://img.shields.io/badge/Python-3.10–3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![macOS](https://img.shields.io/badge/macOS-Apple_Silicon-111111?logo=apple&logoColor=white)](#platforms-and-runtimes)
[![Linux](https://img.shields.io/badge/Linux-x86__64_%7C_aarch64-FCC624?logo=linux&logoColor=111111)](#platforms-and-runtimes)
[![Tests](https://img.shields.io/badge/tests-852_passing-2ea44f)](docs/RELIABILITY.md#validation)
[![License: MIT](https://img.shields.io/badge/license-MIT-2ea44f)](LICENSE)

<img src="assets/banner.png" alt="Transcribe local audio and video transcription" width="100%">

[Get started](#start-transcribing) · [Explore features](#everything-you-need-from-first-pass-to-final-copy) · [Read the guide](docs/USER_GUIDE.md) · [Choose an engine](docs/ENGINES_AND_MODELS.md)

</div>

## From recording to ready-to-use text

Transcribe turns meetings, interviews, lectures, voice notes, podcasts, and screen
recordings into timestamped text without sending the recording to a hosted transcription
service. Use the guided experience when you want a result now, or compose an exact pipeline
for repeatable production work.

Persian and English have first-class, fully validated workflows. The underlying Whisper
engines are multilingual, and advanced one-shot automation can pass other language codes
when compatible model and alignment resources are available.

<table>
<tr>
<td width="33%" valign="top">

### Local by design

Recognition runs on your machine. Once dependencies and models are cached, work can run
without a hosted transcription API.

</td>
<td width="33%" valign="top">

### Resume, don't restart

Long recordings are saved chunk by chunk. Interrupted work continues from completed stages
instead of throwing hours away.

</td>
<td width="33%" valign="top">

### Built for real workflows

Use the menu, the free-form composer, explicit CLI flags, saved presets, or the stable
one-path automation command.

</td>
</tr>
</table>

## Everything you need, from first pass to final copy

| | Capability | What it gives you |
| --- | --- | --- |
| **Fast starts** | One path in, transcript out | Drop in audio or video and get readable text with no settings to learn. |
| **Flexible quality** | Four tuned profiles | Move from a fast first pass to wide-beam, full-precision decoding. |
| **Any compatible model** | Named models and repository IDs | Choose the size, speed, language coverage, and accuracy tradeoff your work needs. |
| **Multiple runtimes** | WhisperX and whisper.cpp | Keep a dependable cross-platform path and use native acceleration where available. |
| **Word-level timing** | Optional alignment | Produce precise timing for review, subtitles, and downstream tools. |
| **Speaker labels** | Optional diarization | Turn conversations into readable speaker-attributed transcripts. |
| **Production formats** | TXT, SRT, VTT, TSV, JSON, Audacity labels | Move directly into editing, publishing, analysis, or another pipeline. |
| **Composable stages** | Prepare, transcribe, align, diarize, export, analyze | Run only what changed and reuse everything that is still valid. |
| **Transcript comparison** | Time-aware difference analysis | Compare engines, models, profiles, and difficult passages without guessing. |
| **Automation-ready** | Stable commands, exit codes, plain redirected output | Use Transcribe from scripts, agents, batch jobs, and larger media pipelines. |

## Start transcribing

Python 3.10-3.13 and FFmpeg are required.

```bash
./cli install       # create the local environment
./cli resources     # prepare alignment resources
./cli status        # verify runtimes, models, paths, and caches
./cli               # open the guided experience
```

Or send a recording straight through the predictable one-shot path:

```bash
./cli /path/to/recording.m4a
```

Use `./cli transcribe file-1.m4a file-2.mov --language en` for batches and automation.
See the [User Guide](docs/USER_GUIDE.md) for configurable runs, the composer, presets,
selective stages, network modes, outputs, and transcript comparison.

## One tool, four ways to work

| Workflow | Best for | Start with |
| --- | --- | --- |
| **Guided** | A clear path from recording to result | `./cli` |
| **One-shot** | Scripts, agents, and repeatable batch work | `./cli recording.m4a` |
| **Configurable** | Exact control over model, quality, stages, and output | `./cli run recording.m4a` |
| **Composer** | Building and revising complex jobs interactively | `./cli composer` |

Every workflow resolves to the same staged pipeline and artifact contracts. Moving from a
menu to automation does not mean adopting a second tool or a different output format.

## Speed that fits the job

Choose `fast`, `balanced`, `accurate`, or `maximum`, then override only what matters.
Profiles coordinate the model, numeric precision, beam search, batching, VAD, and thread
count while memory-aware tuning reduces pressure on constrained machines.

Transcribe is most heavily tuned for Apple Silicon. It detects performance cores, adapts to
available memory, and uses whisper.cpp as the native Metal-accelerated path. WhisperX remains
the dependable CPU-oriented runtime and the default Linux setup.

Model choice stays open: use standard model names, a compatible Hugging Face repository ID
with WhisperX, or a local GGML model with whisper.cpp. Transcribe keeps runtime and model
selection explicit so “faster” never silently means “different words.”

[Understand engines and models](docs/ENGINES_AND_MODELS.md) ·
[Review measured tradeoffs](docs/BENCHMARKS.md)

## A pipeline, not a black box

```text
prepare → find boundaries → transcribe → align → label speakers → export → analyze
```

Ask for the whole flow or only the stage you need. Re-render subtitles without recognizing
again, add speakers to cached text, analyze an existing run, or exclude expensive stages.
The planner includes genuine dependencies and rejects impossible requests before execution.

```bash
./cli run recording.m4a --only export --output-format all
./cli run recording.m4a --only diarize
./cli run recording.m4a --without diarize
./cli run recording.m4a --dry-run
```

Saved preferences and named presets make repeatable workflows short without hiding where a
value came from.

## Long recordings deserve honest results

A polished transcript is not useful when half the recording is missing. Transcribe treats
coverage as a contract:

- canonical raw text and JSON are written before optional refinements;
- completed chunks survive interruption and are reused on the next run;
- failed work retries with lower resource pressure;
- recognition-sensitive changes invalidate only the artifacts they make unsafe;
- missing time ranges are reported explicitly;
- partial output is marked `INCOMPLETE` and returns a non-zero exit status.

The repository has **852 automated tests** across the planner, chunking, cache invalidation,
engines, media handling, progress, rendering, comparison, terminal UI, and integration
boundaries.

[Read the reliability contract](docs/RELIABILITY.md) ·
[Explore the architecture](docs/ARCHITECTURE.md)

## Outputs made for the next step

Every successful recognition writes a canonical timestamped raw transcript. From the same
source, Transcribe can produce readable text, subtitles, structured data, and Audacity
labels. Word timings and speaker attribution flow into formats that support them.

```text
meeting.raw.txt     canonical timestamped text
meeting.raw.json    canonical structured recognition
meeting.txt         readable transcript
meeting.srt         SubRip subtitles
meeting.vtt         WebVTT subtitles
meeting.tsv         tabular timeline
meeting.json        enriched structured output
meeting.aud         Audacity label text
```

By default, a convenient copy is also placed beside the source recording while the complete
run remains in the output location.

## Platforms and runtimes

| Platform | Architecture | Default recognition path | Notes |
| --- | --- | --- | --- |
| **macOS** | Apple Silicon | whisper.cpp | Metal acceleration; most heavily tuned target |
| **Linux** | `x86_64` / `amd64` | WhisperX | Standard installer path |
| **Linux** | `aarch64` / `arm64` | WhisperX | Native environment required |

Whisper.cpp is also available on Linux when built for the target machine. Virtual
environments must be created on their destination OS and architecture; compatible model
data can be reused independently.

## Privacy and model access

Recordings are processed locally. Network traffic is limited to actions such as installing
packages, downloading selected models and tokenizers, and authorizing gated resources.
`--network offline` requires cache-only execution.

Speaker labeling uses a gated Hugging Face model and requires a read token plus acceptance
of its access terms. Models and dependencies retain their own licenses; review
[Third-Party Notices](THIRD_PARTY_NOTICES.md) before redistribution.

## Documentation

| Guide | Use it when you need to... |
| --- | --- |
| [User Guide](docs/USER_GUIDE.md) | Install, choose a workflow, configure stages, save presets, or understand outputs |
| [Engines and Models](docs/ENGINES_AND_MODELS.md) | Choose a runtime or model, prepare Apple Silicon/Linux, or work offline |
| [Reliability](docs/RELIABILITY.md) | Understand resume, retries, cache invalidation, coverage, and exit behavior |
| [Benchmarks](docs/BENCHMARKS.md) | Review measured speed/quality tradeoffs and benchmark limitations |
| [Architecture](docs/ARCHITECTURE.md) | Extend the implementation or understand internal boundaries |
| [Contributing](CONTRIBUTING.md) | Set up development and submit a focused change |
| [Support](SUPPORT.md) | Report a reproducible problem without exposing private media |
| [Security](SECURITY.md) | Report a vulnerability privately |

## Open source

Transcribe is available under the [MIT License](LICENSE). Dependency and model terms are
listed separately in [Third-Party Notices](THIRD_PARTY_NOTICES.md).
