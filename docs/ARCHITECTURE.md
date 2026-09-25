# Architecture

How the code is arranged, and why. Two goals shape it, both of which the flat
six-stage pipeline could not meet:

1. **Any subset of work, in any combination.** Not everyone wants speakers, not
   everyone wants text, and re-rendering subtitles must not re-transcribe an hour.
2. **Growth without a rewrite.** A new capability should be a new file with its own
   tests, not another branch in a growing module.

## Layers

A module may import from the layers below it, never above.

```
        cli / ui                 parsing, prompts, rendering
            │
        actions                  what can be done, declared as data
            │
        services                 planning, job state, orchestration
            │
        media / chunking /       ffmpeg, chunk maths, transcript assembly,
        transcript / engine      recogniser invocation, progress model
            │
        config / paths           settings, layout, environment
```

Everything below `services` is pure or nearly so: chunk planning, progress maths,
transcript assembly, format rendering and settings resolution take values and return
values, with no subprocess, no terminal and no clock of their own. That is why the
awkward cases can be tested without a recording -- and why the suite runs in a tenth
of a second.

## The action model

An action declares what it needs and what it leaves behind.

```python
Action(key="align", provides=("words",), requires=("segments",), options=(...))
```

Artifacts are typed facts, each with a cached location:

| Artifact   | Produced by  | Meaning                            |
| ---------- | ------------ | ---------------------------------- |
| `source`   | —            | the input file                     |
| `audio`    | `prepare`    | levelled mono audio                |
| `plan`     | `boundaries` | the chunk boundaries               |
| `segments` | `transcribe` | recognised text with segment times |
| `words`    | `align`      | per-word timings                   |
| `turns`    | `diarize`    | speaker turns                      |
| `files`    | `export`     | written transcripts                |
| `report`   | `analyze`    | measured statistics                |

`actions/plan.py` turns a request into the **smallest dependency-closed plan** that
satisfies it, given what already exists on disk. That single function is what makes
`--only export` mean "re-render", `--only diarize` mean "label what is already there",
and `--only export --without transcribe` mean "impossible, and here is why".

The resolution is a fixed-point loop rather than a backwards walk, because inserting a
producer can reveal that the producer has unmet needs of its own.

## Options as data

An `Option` is declared once and drives the flag, the prompt, the saved preference and
the validation:

```python
Option("language", str, "Spoken language", choices=("fa", "en"))
```

The flags themselves are written out explicitly rather than generated from the catalog,
because explicit declarations give better help and better errors, and a generator bug is
harder to see than a missing line. What keeps the two in step is a test asserting that
every option the catalog offers is reachable as a flag -- so an option cannot be added
to the composer and forgotten on the command line.

## Package map

| Package       | Responsibility                                                                                     |
| ------------- | -------------------------------------------------------------------------------------------------- |
| `paths/`      | layout, child-process environment, external tool discovery                                         |
| `config/`     | settings declarations, preferences, presets, layer resolution                                      |
| `progress/`   | weighted progress model and learned timings; no Rich, no subprocess                                |
| `media/`      | ffmpeg only: probing, filter chains, safe running, scanning, derived tracks                        |
| `chunking/`   | where boundaries go, how results rejoin, how the cache is keyed                                    |
| `transcript/` | speaker cleaning, word-to-line assembly, format rendering, `canonical.py` (the untouched raw copy) |
| `engine/`     | recogniser commands, stage functions, child-process worker, profiles                               |
| `actions/`    | what can be done, and the resolution of a request into a plan                                      |
| `services/`   | artifact tracking, job state, plan execution, `report.py` (coverage and gaps)                      |
| `ui/`         | console panels, live dashboard, prompts, `menu.py` (the navigable menu), `keys.py`, the composer   |
| `cli/`        | recording discovery, the argument parser, command handlers, `guided.py` (the menu screens)         |

## Integrity: why a short transcript cannot look finished

A 52-minute recording once produced 143 tidy speaker-labelled lines covering 42% of the
audio and was reported as a success. Nothing about that transcript looks wrong; the
shortfall was only visible by comparing word counts against an earlier crashed run. The
rules below exist so that cannot happen again, and each one is tested.

- **The canonical transcript is written first, unconditionally.**
  `<stem>.raw.txt` and `<stem>.raw.json` hold the recognised text with its timestamps,
  exactly as recognised: not merged, not smoothed, no filler removed, never overwritten
  by a formatted export. Everything else in `output/` can be recomputed from the cached
  chunks; this file is the one thing that cannot, so it reaches disk before any formatting
  choice, speaker pass, or later stage can go wrong.
- **Coverage is measured from the chunks that produced text**, never from the last
  timestamp. A hole in the middle of a recording is invisible to a maximum, because
  whatever comes after it still moves the end time outward. Failed and empty chunks are
  excluded from the covered figure and named individually as gaps.
- **Silence and failure are distinguished.** Time in an empty chunk is _accounted for_;
  time that produced nothing and was not explained is _unaccounted for_. Only the second
  kind, above a floor, means something went wrong that nothing reported.
- **A partial run is never presented as success.** It is titled `INCOMPLETE`, names the
  stretches it could not cover with time ranges, points at the raw copy, and exits `2`. The
  completion marker is withheld, so the recording stays visibly unfinished and a rerun
  retries only the missing parts.
- **No text is cleaned by default.** Without speaker turns the recognition segmentation
  is passed through untouched; re-cutting is a presentational choice and a tidier file that
  dropped a line is indistinguishable from a tidier file. Where smoothing is justified, a
  fragment labelled as a different speaker is only absorbed below `REPLY_MAX_DURATION`
  (0.5 s), because a real one-word reply -- `بله`, `آره`, `خب` -- looks exactly like a
  flicker from the label alone.

## Progress

Progress is a model, not a widget (`progress/model.py`), and every rule exists because
the first attempt broke it:

- each stage owns a slice of **one** bar, sized by measured cost, so the bar cannot
  fill and reset once per stage;
- the value is **monotonic**; a source reporting 100% does not finish a stage, because
  only the pipeline does;
- silent phases advance on **elapsed time**, capped so a stage never looks finished
  early -- this is what keeps model loading and native alignment from looking hung;
- the estimate uses a rate measured over a **30-second window**, falling back to learned
  predictions, because an estimate from one instant swings between minutes and seconds;
- measured durations persist as **rates per second of audio**, so a 60-second
  measurement predicts a two-hour file;
- the dashboard repaints on a **ticker thread**; a bar that only moves when a child
  process prints something is a bar that freezes.

## Resume and durability

| Path                                              | Holds                                                                     |
| ------------------------------------------------- | ------------------------------------------------------------------------- |
| `output/.work/<id>/chunks/`                       | per-chunk audio, recognised text, aligned text                            |
| `output/.work/<id>/plan.json`                     | the slicing fingerprint _and_ its boundaries                              |
| `output/.work/<id>/turns-*.json`                  | speaker turns, keyed by requested count                                   |
| `output/.state/<stem>-<id>.json`                  | completion marker and job fingerprint                                     |
| `output/.state/timings.json`                      | learned stage durations                                                   |
| `output/.logs/<stem>-<id>.log`                    | full child-process log                                                    |
| `output/<stem>.raw.txt`, `output/<stem>.raw.json` | the canonical transcript: written first, never cleaned, never overwritten |
| `output/<stem>.<ext>`                             | the readable transcript, rendered from the same segments                  |

The boundaries are stored, not only a hash of them: a run that starts mid-plan has to
re-derive the same chunks, and a hash cannot be inverted. Getting this wrong would mean
a later stage silently reading files cut differently from what the plan describes.

## Deliberate trade-offs

- **The runner owns execution, not the actions.** Actions declare dependencies and
  options; a handler per action in `services/runner.py` performs the work. Giving each
  action its own `run(context)` would be more extensible, at the cost of threading a
  shared context through every one of them. The declaration is what the rest of the
  system consumes, so the visible behaviour is the same; if the action set grows past
  about a dozen, moving execution into the actions is the natural next step.
- **Some ported modules are still single files.** `media/`, `chunking/` and
  `transcript/` are split by concern. Nothing named `library`, `menu` or `profiles`
  survives from the previous design: their responsibilities are now in `cli/library`,
  `ui/menu` with `cli/guided`, and `engine/profiles`. The old `menu` was a typed console
  with a menu-sounding name; the new one is an actual menu.
- **`--force` clears the completion marker too.** The marker lives outside the work
  directory, so clearing the workspace alone did not actually force anything.

## Migration status

| Area                                                                              | State                                                                                 |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `paths/`, `config/`, `progress/`, `media/`, `chunking/`, `transcript/`, `engine/` | split, ported, tested                                                                 |
| `actions/`, `services/`, `ui/`, `cli/`                                            | rewritten for this layout, tested                                                     |
| `transcript/canonical.py`, `services/report.py`                                   | new; written for the integrity rules above                                            |
| `ui/menu.py`, `ui/keys.py`, `cli/guided.py`                                       | new; the menu is the default way in                                                   |
| Cross-cutting integration tests in `tests/`                                       | written (`tests/test_integration.py`)                                                 |
| `ui/dashboard.py`                                                                 | ported from the previous design, still one file, no direct tests                      |
| `ui/composer.py`                                                                  | no direct tests; exercised only through the command line                              |
| A per-feature document                                                            | does not exist. There is no `docs/FEATURES.md`; the feature list lives in `README.md` |

The counts, so this table can be checked rather than believed: `./cli selftest` runs two
roots in separate processes and reports 14 cross-cutting tests plus the colocated ones.

## Testing shape

Tests live beside the code they cover, one file per module, because a module's rules are
easier to find next to it than in a separate tree. Two things live outside that pattern:

- `tests/test_integration.py` covers the seams -- whether the planner and the artifact
  scanner describe a recording the same way, whether text survives chunking, merging and
  every renderer, and whether job state means what it says.
- `services/tests/test_runner.py` drives the runner's handlers with a recording stub, so
  the order the handlers do things in can be asserted: that a failed alignment still
  leaves the text on disk, and that a run with a hole in it does not record itself as done.

The stdlib `unittest` runner only. No pytest: it is not installed, and the two roots are
run as separate processes because their module names would otherwise collide.
