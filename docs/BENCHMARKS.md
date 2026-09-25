# Benchmarks and Tradeoffs

These measurements are case studies, not universal performance promises. They were collected
on an M1 Pro with 16 GB of memory using 300 seconds of real Persian conversation. Hardware,
model, build flags, language, audio quality, and decoding settings all affect results.

## Profile and Model Findings

| Change | Observed result |
| --- | --- |
| `maximum` float32 vs `accurate` int8 at beam 10 | About 1.65x the runtime |
| CTranslate2 turbo vs `large-v3` at matched settings | About 2.47x faster |
| Grouping three chunks per recognizer call | No measurable gain on warm storage cache |
| Wider beam search | The largest quality/time control in these tests |

Repeated turbo runs produced identical text in the measured case, but turbo and `large-v3`
disagreed materially. Faster is not automatically more accurate, and model comparisons
without human ground truth measure disagreement rather than correctness.

## Runtime Findings

| Engine and model | Recognition time on the case study |
| --- | ---: |
| WhisperX `large-v3`, int8, beam 10 | 366.7 s |
| WhisperX turbo, int8, beam 10 | 148.6 s |
| whisper.cpp `large-v3` f16, beam 8 | 113.8 s |
| whisper.cpp turbo q8_0, beam 8 | 34.4 s |

The Apple Silicon advantage comes from whisper.cpp's native Metal path. CTranslate2 does
not provide a Metal backend, so changing a WhisperX setting cannot produce the same class of
acceleration.

## Reading Comparisons Correctly

`./cli compare` projects both transcripts onto shared time windows before calculating edit
distance. It separates substitutions, insertions, and deletions and applies optional
Persian-aware normalization. A boundary word can still appear as a deletion plus an
insertion, so inspect high-disagreement windows against the audio before drawing a conclusion.

Do not generalize these figures to another machine or present them as accuracy scores. Run
the same recording through both configurations on the target machine and compare the output
that matters to that workflow.
