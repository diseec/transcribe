# Engines and Models

Transcribe keeps recognition behind a common adapter, so chunking, resume, coverage,
alignment, speakers, and exports behave consistently across runtimes.

## Recognition Engines

| Engine | Runtime | Default use |
| --- | --- | --- |
| `whisperx` | WhisperX with CTranslate2 | Linux default and cross-platform CPU path |
| `whispercpp` | Native whisper.cpp binary | macOS default and optional Linux runtime |

WhisperX transcription is CPU-oriented in this application. whisper.cpp can use the native
acceleration available in its build, including Metal on Apple Silicon. `./cli status`
reports which runtime, models, and supporting resources are ready.

## Model Choice

WhisperX accepts standard model names or compatible Hugging Face repository IDs:

```bash
./cli run recording.m4a --engine whisperx --model large-v3
./cli run recording.m4a --engine whisperx --model owner/compatible-ct2-model
```

Whisper.cpp accepts an explicit GGML model path or discovers named `.bin` files under
`models/whisper.cpp`. It does not download these models automatically.

```bash
./cli run recording.m4a --engine whispercpp --model models/whisper.cpp/ggml-large-v3.bin
```

Model compatibility remains the user's responsibility. A model can change speed, language
coverage, memory use, and recognized words independently of the selected profile.

## Language Coverage

The underlying Whisper engines are multilingual. Transcribe provides first-class,
UI-validated workflows for Persian (`fa`) and English (`en`), including the project's
tested alignment path. The one-shot automation command can pass another language code to
the recognizer, but compatible alignment resources and equivalent end-to-end validation
are not guaranteed for every language.

## Apple Silicon

Apple Silicon is the most heavily tuned local target:

- performance-core detection avoids blindly scheduling every logical core;
- batch and thread settings adapt to available memory;
- whisper.cpp provides the Metal-accelerated recognition path;
- the adapter applies the required Metal residency workaround for affected builds;
- profiles preserve a CPU fallback through WhisperX.

macOS defaults to whisper.cpp. A configurable `run` expects its binary and model to be
ready; the simple one-shot command can announce and use a WhisperX fallback.

## Linux and Architectures

The setup supports Linux on `x86_64`/`amd64` and `aarch64`/`arm64`. Linux defaults to
WhisperX because the standard installer provides that path. whisper.cpp can also be used
when built natively for the target machine and placed on `PATH`.

Never copy `.venv` between macOS and Linux or between CPU architectures. Model data may be
reused when the selected runtime uses the same model format.

## Caches and Offline Use

Hugging Face, Torch, NLTK, normalized media, speaker data, and application state are kept
in explicit local paths. `./cli install` installs code, `./cli resources` acquires the
alignment tokenizer, and recognition models are handled separately.

Use `--network offline` to require cache-only execution. Speaker diarization requires prior
authorization and download of its gated model even when the eventual run is offline.
