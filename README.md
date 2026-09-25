# whisperx-local

[![Ubuntu CI](https://github.com/diseec/transcribe/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/diseec/transcribe/actions/workflows/ubuntu.yml)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%E2%80%933.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-2ea44f.svg)](LICENSE)

**Private, resumable transcription for Persian and English audio and video.**
WhisperX Local turns recordings into timestamped transcripts on your own machine,
with optional word alignment, speaker labels, comparison, and export formats.

No hosted account is required for ordinary transcription. The recording is processed
locally; model downloads are separate setup traffic, and no transcription service is
required.

## Why people use it

- **Start with one command.** Run `./cli recording.m4a` and get a readable transcript.
- **Use a guided workflow or scripts.** The menu is friendly for people; stable commands
  and fixed one-shot settings are predictable for automation.
- **Resume long recordings.** Work is chunked, cached, and retried without recomputing
  completed chunks.
- **Keep the raw result.** Canonical timestamped recognition is written before optional
  alignment, speaker separation, formatting, or analysis.
- **Know when something went wrong.** Missing coverage is reported as gaps and produces a
  non-zero exit status instead of looking like a complete transcript.
- **Run on more than one machine.** Ubuntu `x86_64` and `aarch64`, macOS Apple Silicon,
  and CPU-first fallback execution are supported by the repository's setup path.

## At a glance

| Input                                   | Output                                          | Optional stages                               |
| --------------------------------------- | ----------------------------------------------- | --------------------------------------------- |
| Audio or video files accepted by FFmpeg | `.txt`, `.srt`, `.vtt`, `.tsv`, `.json`, `.aud` | Word timings, speaker turns, quality analysis |

The project is deliberately local-first rather than a hosted transcription service.
It is a good fit for meetings, interviews, screen recordings, research notes, and
automation where the original audio should remain under your control.

**Project links:** [Contributing](CONTRIBUTING.md) · [Support](SUPPORT.md) ·
[Security](SECURITY.md) · [Changelog](CHANGELOG.md) ·
[Third-party notices](THIRD_PARTY_NOTICES.md)

## Getting started

```bash
./cli                 # the menu: choose a recording, then run it
./cli status          # prerequisites, paths, caches
./cli install         # create .venv and install (needed once, if .venv is absent)
./cli resources       # download the alignment tokenizer
```

Speaker labels need a Hugging Face read token. Copy `.env.example` to `.env` and add
it, or leave speakers switched off in the menu. Everything else works without it.

### Licensing and downloaded models

The application code is released under the MIT License. Dependencies, FFmpeg,
and downloaded speech models are separate components with their own licenses.
The speaker-diarization model currently used by the optional speaker stage is
licensed CC-BY-4.0 and requires accepting its Hugging Face access conditions.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistributing a
packaged environment, model files, or generated bundle.

### Ubuntu on x86_64 and aarch64

Ubuntu is supported on both 64-bit Intel/AMD (`x86_64`, also called `amd64`) and
64-bit ARM (`aarch64`, also called `arm64`). Install the system prerequisites and let
the app create a native virtual environment on that machine:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg python3 python3-venv
./cli install
./cli resources
./cli status
./cli selftest
```

Do not copy `.venv` from macOS or from the other CPU architecture; Python and native
machine-learning wheels must be installed on the target host. Model files under
`models/` are data and may be reused across these architectures.

Linux defaults to the WhisperX CPU backend installed by `./cli install`. The optional
`whispercpp` backend also works when `whisper-cli` is built natively for the target
architecture and placed on `PATH`; select it with `--engine whispercpp`. macOS keeps
the Metal-enabled whisper.cpp backend as its default.

## Running one recording

```bash
./cli run input/meeting.m4a                               # your saved settings, the default set
./cli run input/meeting.m4a --profile fast                # faster, slightly less exact
./cli run input/meeting.m4a --dry-run                     # show the plan, change nothing
./cli run input/meeting.m4a --analyze                     # also report quality and talk time
```

Use `run` when you want flags and your own saved settings. A path on its own is the
one-shot command below, which ignores both on purpose.

Video is accepted as readily as audio: screen recordings are a normal input, not a
special case.

A plain run does `prepare → boundaries → transcribe → export`: the speech, its timings,
and the files. It deliberately does not run `align` or `diarize`, which are refinements
that can fail on their own and cannot add a single recognised word. Both are worth
having; neither belongs in the path that decides whether you get a transcript.

## One path in, one transcript out

```bash
./cli /path/to/recording.m4a        # also: ./cli transcribe <files…> for several at once
```

This form exists so that a script or an agent can drop in a path and wait, without
learning a single flag. It **pins every setting that shapes the output**, so the same path
gives the same result for you, for a colleague, and for a script, on any machine — and it
**does not read your saved settings at all**, which is the part that makes it predictable.

What it fixes: the fastest recogniser available, `beam 5`, silero voice detection (so it
needs no Hugging Face token), levelling and pause-cutting on, retries at 3, and a readable
`.txt` under `output/` — no alignment, no speakers, no analysis. On a machine where
whisper.cpp is not installed it falls back to WhisperX and says so, rather than failing.

```bash
./cli /path/to/recording.m4a --language en     # the only flag there is
```

The measured cost of this form: 300 seconds of Persian returns in about **33 s**, against
367 s for a plain `large-v3` run through WhisperX. It reaches that by using the turbo
weights, whose words differ from large-v3's on part of the text -- so it is the right shape
for a script that needs an answer, and `run` is still the shape for a transcript you intend
to rely on word by word.

### Profiles, and which one to pick

| Profile    | Decoding         | Roughly | For                                                          |
| ---------- | ---------------- | ------- | ------------------------------------------------------------ |
| `fast`     | int8, beam 1     | fastest | a first pass, or when you will listen anyway                 |
| `balanced` | int8, beam 5     | fast    | the default; good speed without giving up much               |
| `accurate` | int8, beam 10    | slow    | the widest search at int8 speed. **Start here for accuracy** |
| `maximum`  | float32, beam 10 | slowest | the last fraction of accuracy, at roughly double the time    |

Beam width is what buys accuracy; precision mostly buys time. `accurate` and `maximum`
search equally widely and differ only in numeric precision, which measured about **2×**
on this machine — so reach for `maximum` only when a specific passage needs settling.

### Comparing two transcripts

`./cli compare` measures one transcript against another and says where they differ. It
reads two files and touches nothing else — no recording, no cache, no workspace:

```bash
./cli run input/meeting.m4a --profile accurate --only transcribe --output-dir output/int8
./cli run input/meeting.m4a --profile maximum  --only transcribe --output-dir output/float32
./cli compare output/int8/meeting.raw.json output/float32/meeting.raw.json
```

It exists because "is this engine worse" cannot be answered by reading two transcripts.
Three things make its numbers mean something, and each is easy to get wrong:

- **It aligns by time, never by segment.** Two engines cut the same audio into different
  pieces, so comparing segment one with segment one compares unrelated speech and you end
  up measuring segmentation. Both sides are projected onto a shared 30-second clock first
  (`--window` to change it).
- **It folds Persian first** (`compare/normalize.py`). Persian is written in the Arabic
  script, where the same word can be typed with either codepoint of an identically-shaped
  letter, and ZWNJ is invisible to a reader and a word boundary to a tokeniser. Without
  folding, a keyboard difference reads as a recognition error larger than the difference
  between the engines. Turning folds _off_ is available (`--keep-zwnj`, `--distinct-letters`,
  `--keep-punctuation`) for when the spelling is the question.
- **It separates substitutions from insertions and deletions.** An engine that invents
  words fails differently from one that misses them, and one rate hides which is happening.

It also tracks the vocabulary this domain gets wrong — `subscription`, `سابسکریپشن`,
`دیتابیس`, `API`, `UI`, `پرو` — and reports each term's count on both sides
(`--term` to replace the list), plus mean `avg_logprob` as a confidence proxy.

Two honest limits. There is no ground truth here, so a small disagreement means the two
are interchangeable and **does not** prove either is correct; and a word landing on a
window boundary is scored as a deletion plus an insertion, which is why the window is 30
seconds rather than something smaller. A disagreement right at a boundary deserves a
listen before it is believed. Exit status is `4` when one side held no recognised text and
nothing could be compared.

### Asking for less

`--only` and `--without` name actions, and the planner adds whatever the request
genuinely needs and nothing more.

```bash
./cli input/meeting.m4a --only export              # re-render subtitles from cached work
./cli input/meeting.m4a --only diarize             # label speakers on an existing transcript
./cli input/meeting.m4a --only analyze             # report, do not touch the transcript
./cli input/meeting.m4a --without diarize          # everything except speakers
./cli input/meeting.m4a --only export --output-format all
```

Known actions: `prepare`, `boundaries`, `transcribe`, `align`, `diarize`, `export`,
`analyze`.

A request that cannot be honoured says so instead of failing halfway:

```
$ ./cli input/meeting.m4a --only export --without transcribe
  ! export cannot run: something it needs was excluded
```

## The menu

Run `./cli` with no arguments. Three questions, then a confirmation:

```
Which recording?      the recordings found in input/ and ~/Downloads, newest first
How much work?        transcription only · word timings · speakers · everything
How careful?          accurate · balanced · fast · fine-tune individual settings
Start?                run · preview · change something · cancel
```

Arrow keys move and space marks; numbers select directly; `q` cancels. Free text is
asked for only where a value cannot be offered as a list, such as a path. If the output
is redirected, the same menu prints once and takes numbered answers instead of trying
to redraw itself.

`Fine-tune settings…` opens every setting by name, each with its accepted values offered
and any value still accepted by hand.

## The composer

`./cli composer` is the free-form alternative. There is no first step and no required
order: change files, actions or settings at any point, and the current state is shown
after every change.

```
  Files     meeting.m4a
  Actions   transcribe  align  diarize  export
  Settings  language=fa, output_format=srt
  Preset    none

whisperx › actions -diarize +analyze
whisperx › set output_format            # prompted, with the accepted values offered
whisperx › files add ~/Downloads/talk.mov
whisperx › preset save persian-fast
whisperx › plan                         # exactly what would run
whisperx › run
```

`help` lists every command and `help <action>` explains one, including the options it
accepts.

## Saved defaults

Anything you would otherwise repeat as a flag can be saved once. Layers resolve in
this order, later winning:

```
built-in defaults → preferences → preset → environment → flags
```

```bash
./cli config show                        # every setting, and where its value came from
./cli config set language fa
./cli config set chunk_seconds 300       # checked on the way in
./cli config unset language
./cli config reset
./cli preset save persian-talk           # save the current preferences under a name
./cli preset list
./cli run input/meeting.m4a --preset persian-talk
```

Stored at `~/.config/whisperx-local/settings.json`.

## Long recordings

A one-hour recording is processed in chunks, each saved as soon as it exists.

- **Resume.** A finished chunk is never recomputed. Re-running the same command
  continues where it stopped.
- **Retries.** Each chunk retries with fewer threads, because peak memory is what
  usually causes these aborts.
- **Gaps are reported**, with the exact time range, and the completion marker is
  withheld so a rerun retries only the missing parts. A run with gaps ends on a
  non-zero status and is titled `INCOMPLETE`, because a 52-minute recording once
  produced a tidy transcript covering 42% of it and called that a success. Nothing
  about a shortened transcript looks wrong on inspection, so it is stated instead of
  left to be noticed.
- **Coverage is measured from the chunks that produced text**, not from the last
  timestamp. A hole in the middle of a recording is invisible to a maximum: whatever
  comes after it still moves the end time outward.
- **Boundaries land in pauses** when there are any. When a recording has no silence at
  all, the quietest available moment is used -- but only if the audio has enough
  dynamic range for that to mean anything, because heavily limited audio is flat and
  picking its "calmest" window is picking noise.

## Output

Two kinds of file are written, and the distinction is the point.

**`<name>.raw.txt` and `<name>.raw.json` — the canonical transcript.** Written first,
on every run that recognised anything, before any formatting choice or speaker pass can
go wrong. Timestamped, one segment per line, exactly as recognised: not merged, not
smoothed, no filler removed. If anything downstream fails, this is still there. It is
also what to compare against when a transcript looks thinner than it should.

**`<name>.txt`, `.srt`, `.vtt`, `.tsv`, `.json`, `.aud`** — the readable transcript,
rendered from the same segments. Speaker labels appear when speakers were asked for and
succeeded. Without speaker turns the recognition segmentation is passed through
untouched, because re-cutting is a presentational choice and a tidier file that dropped
a line is indistinguishable from a tidier file.

At the end of every run:

```
✓ Transcript complete
Transcript        /path/to/output/meeting.txt
Recovered         4,812 words
Covered           3,104s of 3,104s (100%)
```

A run that fell short ends differently, and cannot be mistaken for one that did not:

```
✗ Transcript INCOMPLETE
Covered           1,300s of 3,104s (42%)
Missing 1         1,200s–1,800s (not transcribed)
Raw copy          meeting.raw.txt
Next              Re-run the same command...
```

The exit status is `2`, so a script cannot treat it as success either. `--output-format
all` writes every readable format; the raw pair is always written regardless.

### A copy beside the recording

By default the finished transcript is also written next to the recording it came from,
carrying the recording's own name and the exported extension. A transcript is easier to find
beside its recording than inside a directory named after it:

```text
~/Downloads/talk.m4a      the recording
~/Downloads/talk.txt      the transcript, written automatically
output/talk.txt           the same transcript, in the usual place
```

Two deliberate limits. Recordings that already live in `input/` are **excepted** -- they were
collected there to be transcribed, so a copy is clutter rather than convenience. And the copy
is skipped with a warning, never a failure, if the folder cannot be written to: the transcript
is already safe in the output directory by then, and a read-only folder must not turn a
finished run into a failed one.

`--no-copy-beside-input` turns it off for one run, and `copy_beside_input` in the settings
turns it off for good. The completion panel names the path it wrote, so a copy is never left
somewhere you have to guess at.

## Speed, and what actually moves it

Measured on this machine (M1 Pro, 16 GB), on 300 s of real Persian conversation:

| Change                                                                                                     | Effect                                               |
| ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `accurate` (int8, beam 10) vs `maximum` (float32, beam 10)                                                 | float32 costs **1.65×** (583 s vs 353 s)             |
| turbo (`--model deepdml/faster-whisper-large-v3-turbo-ct2`) vs `large-v3`, same profile and the same 300 s | **2.47× faster** (148.6 s vs 366.7 s)                |
| `--chunks-per-call 3` (one model load per three chunks)                                                    | **no measurable difference** (174.7 s vs 174.2 s)    |
| Beam width                                                                                                 | the widest lever: beam 10 costs several times beam 1 |

Two of those are worth spelling out, because each one cost an experiment to learn:

**Precision buys almost nothing.** int8 and float32 differ on **7.8% of words** for the
same audio, and the comparison harness cannot say which is right -- only that they
disagree. So `accurate` is the sane default and `maximum` is for settling one passage.

**Batching chunks into one recogniser call does nothing here.** The reasoning was that
each invocation reloads the model, and a first attempt to price that put it at ~25 seconds.
A direct test over four chunks with a warm cache gave 174.2 s ungrouped and 174.7 s
grouped: half a second, in the wrong direction. CTranslate2 maps the model file, so once
the pages are cached "loading" is nearly free. The mechanism is kept (`--chunks-per-call`)
because it is tested, text-neutral, and should help on slower storage -- but it is off by
default, because nothing here shows a reason to turn it on.

**Transcription cannot use the GPU.** CTranslate2 has no Metal backend, so the part that
dominates a run is CPU-bound on Apple Silicon whatever the settings say. The PyTorch parts
(word alignment, speaker separation) could use `mps`; they do not by default. This is why
"optimise for the M-series" has a low ceiling with this stack, and why the one route to the
GPU is a different recogniser rather than a different setting -- see **Engines** below.

**The turbo model is about 2.5× faster, and a run is reproducible.** On the same 300 s at
the same settings, turbo took **148.6 s against large-v3's 366.7 s**. Two identical turbo
runs produced the _same_ transcript word for word -- 778 words both times, 0.0%
disagreement, the same mean `avg_logprob` (-0.2227) -- so a re-run is a fair test of a
changed setting rather than a source of noise.

A comparison is only worth the audio it is taken from: those two figures come from one file,
because an earlier attempt compared a 300 s excerpt of the recording with a _different_
300 s of it and reported a speed-up that meant nothing.

What turbo is _not_ is a drop-in replacement. On that same 300 s the two models disagreed on
**34.9% of words** -- but `compare` counts a spelling variant as an error like any other, so
the number has to be split before it says anything:

| Kind of difference                                       | Words | Of the text |
| -------------------------------------------------------- | ----- | ----------- |
| The same word, spelled differently (1–2 character edits) | 85    | 10.6%       |
| A word large-v3 wrote that turbo did not                 | 59    | 7.4%        |
| A word turbo wrote that large-v3 did not                 | 38    | 4.8%        |
| The same position filled with genuinely different words  | 97    | 12.1%       |

So a quarter of the text is a real disagreement rather than a spelling one, and the
disagreements are not evenly spread: on the hardest passages turbo invents speech where
large-v3 stays coherent, while large-v3 reads _subscription_ as the loanword it is
(`سابسکریپشن`) and turbo as `سابسکریبشن`. Turbo is also less sure of itself -- mean
`avg_logprob` **-0.22 against large-v3's -0.18**. That is why `large-v3` is still the
default and turbo is an option for when the wait matters more than the words:

```bash
./cli run input/meeting.m4a --model large-v3 --only transcribe --output-dir output/large
./cli run input/meeting.m4a --model large-v3-turbo-q8_0 --only transcribe --output-dir output/turbo
./cli compare output/large/meeting.raw.json output/turbo/meeting.raw.json
```

## Engines

What reads the audio is a seam, not an assumption. One function builds a command line and
one reads the file that command wrote; everything downstream -- chunking, coverage,
alignment, the canonical transcript -- sees one shape and does not care who produced it.

| `--engine`   | What it is                                | State                                             |
| ------------ | ----------------------------------------- | ------------------------------------------------- |
| `whispercpp` | whisper.cpp with native host acceleration | optional; the macOS default when installed        |
| `whisperx`   | WhisperX driving CTranslate2, on the CPU  | the Linux default and the cross-platform fallback |

`whisperx` is a wrapper around the exact command and reader that were already there, and
there is a test asserting that, because a seam is precisely the change that alters an
output filename nobody meant to touch. `whispercpp` is the macOS default when available
because it measured several times faster on the same audio at the same weights. Linux
defaults to WhisperX because `./cli install` can set it up without a separate native
binary build. `./cli status` says which engine is ready on the current machine.

### Trying whisper.cpp

On macOS, it is the route to the Metal GPU because CTranslate2 has no Metal backend. On
Linux, it can use native CPU or accelerator support from the whisper.cpp build. It cannot
run until the binary and model are present, and it says which is missing rather than
failing halfway:

```bash
brew install whisper.cpp                 # the binary, 6.5 MB plus a 19 MB ggml dependency
mkdir -p models/whisper.cpp              # models go beside the others
# weights, from huggingface.co/ggerganov/whisper.cpp:
#   ggml-large-v3.bin              2952 MB  the default: same weights as before, no quantisation
#   ggml-large-v3-turbo-q8_0.bin    834 MB  much faster, slightly different words
# silence trimming, from huggingface.co/ggml-org/whisper-vad:
model=ggml-silero-v5.1.2.bin
.venv/bin/hf download ggml-org/whisper-vad $model --local-dir models/whisper.cpp
```

The VAD model is 865 KB and worth having: without it silence inside a chunk is left in,
the way the other engine does not, and that measurably _widened_ the distance between the
two engines from 35.8% to 39.9% of words. The engine uses it automatically when present,
and says so when it is not.

**Measured on this machine** (M1 Pro, 300 s of the same Persian audio, ASR only):

| Engine       | Model                      | Time                                                     | Against `large-v3` |
| ------------ | -------------------------- | -------------------------------------------------------- | ------------------ |
| `whisperx`   | `large-v3`, int8, beam 10  | 366.7 s                                                  | 1.0×               |
| `whisperx`   | turbo, int8, beam 10       | 148.6 s                                                  | 2.5×               |
| `whispercpp` | `large-v3` q5_0, beam 8    | 128.8 s                                                  | 2.8×               |
| `whispercpp` | **`large-v3` f16, beam 8** | **113.8 s** (135.6 s through this app, prepare included) | **3.2×**           |
| `whispercpp` | turbo q5_0, beam 8         | 38.3–42.3 s                                              | ~9×                |
| `whispercpp` | turbo q8_0, beam 8         | **34.4 s** (39.0 s through this app, prepare included)   | **~10×**           |

**The default is `large-v3`, at f16**, and the comparison that justifies it is the row
above against the one it replaces: same weights, same 300 s, same app, 135.6 s against
348.8 s — **2.57× faster**, with 788 words against 793. Note that f16 beat both
quantisations on _time_ as well as fidelity: 113.8 s against q5_0's 128.8 s and turbo
q8_0's 34.4 s. The smaller file is repeatedly the slower one, because f16 has a native
Metal kernel path while the k-quants pay to dequantise on every pass.

That is roughly **4× faster than CTranslate2 running the same turbo weights**, and the
reason is the one the "no Metal backend" note above predicts: the work leaves the CPU. It
is also an understatement of the gap, because CTranslate2 is given silence-trimmed audio
(pyannote VAD runs inside WhisperX) while whisper.cpp was given the raw 300 s.

Driven end to end through this app on the same file it took **39.0 s including preparing
the audio**, and its transcript was **0.0% different from the standalone run** — 794 words
and 100% agreement on every 30-second window. That is the check that the app feeds the
recogniser the same audio and reads back the same words.

Three things that cost an afternoon and are worth knowing before trying it:

- **Homebrew's build aborts without `GGML_METAL_NO_RESIDENCY=1`.** ggml 0.24.0 asserts in
  its Metal residency-set teardown (`ggml-metal-device.m:1025`), so the run dies in half a
  second with `SIGABRT` after loading the model. Setting that variable makes it work; the
  adapter sets it itself.
- **Beam size is capped at 8, and the cap does not move with `--threads`.**
  `whisper_full_with_state: too many decoders requested (10), max = 8`. This matters here
  because `accurate` and `maximum` use beam 10, so a like-for-like A/B against them is not
  possible without lowering the shared setting and saying so. The adapter brings the beam
  down to 8, and the preflight prints a line before the run saying that it did.
- **It does not create the directory it writes into.** Given an output prefix inside a
  missing directory it exits **zero** and writes nothing at all, so the pipeline reports a
  chunk it could not transcribe -- after four retries that each reload the whole model.
  The adapter creates the directory, which is why the other engine's command builder does
  the same thing.

The output shape the adapter was written against is now confirmed rather than assumed: the
`--output-json` payload has a `transcription` list whose entries carry both `offsets` in
milliseconds and a `timestamps` clock, exactly the two forms `read_whispercpp_json` handles.
And the format it _cannot_ read is the one that matters for A/B: `./cli compare` finds no
segments in that payload, because it expects `{"segments": [{start, end, text}]}`, so
teaching it whisper.cpp's native output is the one piece a `./cli ab` would still need.

**What the disagreement is, and is not.** whisper.cpp and CTranslate2 turbo disagree on
about a quarter of the words, and two explanations have now been tested and rejected:

- **Not quantisation alone.** q8*0 and q5_0 of the \_same* model, same flags, differ by
  **18.1%** -- so 5-bit does cost something real, but it is not the whole gap, and at 8-bit
  whisper.cpp has fewer differences from CTranslate2 (40.3%) than q5_0 had (43.1%).
- **Not the VAD.** Giving whisper.cpp its own VAD, so both sides trimmed silence, made the
  disagreement _wider_ (45.0% against 40.3%), not narrower.

What is measurable is that the engines genuinely decode differently, and that on this
recording one of them produced an artefact the other three never did: CTranslate2 turbo
repeats `بایمون` **13 times inside one 5-second window** (`[00:01:56 --> 00:02:01] خوب بایمون
این همون بایمون بایمون بایمون ...`), while whisper.cpp at both quantisations and `large-v3`
produce it **zero** times. Twelve repeats of one nonsense word in five seconds is a loop,
not speech -- but the honest reading is “CT2 turbo is the outlier here”, not “whisper.cpp is
more accurate”, because there is still no ground truth for the other quarter.

An adapter that cannot read its input has to fail loudly rather than quietly: a payload whose
shape is not understood **raises** instead of being read as an empty transcript, because an
adapter that silently returned nothing would be indistinguishable from a recording with no
speech in it -- the failure the rest of this program spends its effort preventing.

### Adding another runtime

An adapter answers four questions, in `engine/backends.py`: is it usable; what command runs
it; where did it write; what does its output mean. Register it in `BACKENDS` and it appears
in `--engine` and in the settings screen with no other change.

## Known limits

- **A first run cannot show accurate progress.** Estimates are learned from measured
  stage durations, so the first run of a new profile interpolates with built-in
  weights and each chunk's model loading shows as a pause. Later runs are predictable.
- **Simultaneous speech cannot be separated** from one mixed channel. No setting fixes
  this; it needs separate microphones.
- **Speaker separation is slow** on CPU, and on some recordings costs more than
  transcription itself.
- **A turbo model is faster, not thereby better.** Measured at 2.47× large-v3's speed on the
  same audio, it also disagreed with large-v3 on a quarter of the words, inventing speech on
  the hardest passages where large-v3 stayed coherent. Run both and compare before trusting
  the shorter run.
- **The remaining speed lever is a different runtime.** whisper.cpp is the one route to
  Metal, since CTranslate2 has no GPU backend. The adapter exists (`--engine whispercpp`)
  and is documented, but unrun until a build and a GGML model are both present.
- `--only` names a _set_ of actions; a dependency excluded with `--without` blocks the
  action that needed it, and the plan says so rather than failing later.

## Tests

```bash
./cli selftest
```

Tests live beside the code they cover (`src/whisperx_local/<module>/tests/`), with
suites that span modules in `tests/`. The runner discovers both.
