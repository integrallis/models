# Private release — squad2 Noul v0.1

A decision artifact you can run. Not a benchmark claim; see the limits below before quoting it.

## Run it

```
./gradlew :models-decisions:decide \
  -Partifact=../models-decisions-release/squad2-noul-v0.1.idsn \
  -Pbase=$HOME/.jvllm/models/granite-4.1-3b-Q4_K_M.gguf \
  -Pin=questions.jsonl \
  -Pout=verdicts.jsonl
```

Input, one JSON object per line:

```json
{"id":"q1","context":"...","question":"..."}
```

Output, one per line:

```json
{"id":"q1","probability":0.978495,"decision":true,"truncated":false,"latency_s":25.80,"prompt_tokens":74}
```

## Cut a new one

```
./gradlew :models-decisions:release \
  -Pharvest=../models-decisions-harvest/squad2.jsonl -Pcorpus=squad2 \
  -Pbase=granite-4.1-3b -Pout=../models-decisions-release/squad2-noul-v0.2.idsn
```

Fits on `train`, calibrates on `calibration`, reads `sealed` **once**, and writes the artifact that
produced those numbers. The sealed split refuses a second reading, so a configuration cannot be
chosen after seeing the result.

## What is in the file

`squad2-noul-v0.1.idsn`, 82,020 bytes,
sha256 `efe04e35843c92ee2629129e73125e106ec47d9d6ac31f5c7d0de350a6ed6c8f`.

The standardiser, the head and the temperature that were measured together — 2,560-wide, fitted on
Granite 4.1 3B hidden states. Not the base model: this is the small learned part, and it names the
base so a mismatch is caught rather than silently producing a plausible wrong answer. Writing the
same artifact twice produces identical bytes, so a release is pinned by its digest.

## Measured, sealed split read once

| | |
| --- | ---: |
| sealed items | 300 |
| majority-class floor | 0.5033 |
| **head accuracy** | **0.8000** |
| base decode accuracy | 0.7867 |
| agreement with decode | 0.9333 |
| **ECE** | **0.0477** |
| Brier | 0.1459 |
| temperature | 4.1060 |
| truncated in sealed | 0.0000 |

The base clears its floor by 0.28, so the comparison is informative rather than a coin flip.

## Limits — read before quoting

- **This answers one third of JevBench.** It is a Noul head: binary only. The public 72-item cohort
  is 24 noul, 36 choice, 12 score. Choice and score still run through the untrained base, which is
  why routing sits at 0.333 (chance) and adequacy at 0.417. There is no trained head for them
  because the harvest produced binary labels only. That is a missing training corpus, not a
  packaging problem.
- **It learned span-presence, not entailment.** On a six-item probe it went 5/6; the miss was
  "can an opened item be refunded?" against a policy stating opened items may be exchanged but not
  refunded. The text settles the question, and the head scored it 0.077 — confidently wrong. squad2
  teaches "is there a matching span", so a negative inference reads as absence. This is the same
  weakness as the adequacy family.
- **Local latency is not the latency.** There is no `macos-x86_64` Rust kernel, so on a Mac this
  falls back to pure Java at roughly 22 s/item against 2.26 s/item measured on Linux with rust-ffm.
  The CLI prints which kernel ran; a number taken from the fallback is not comparable.
- **0.8000 is squad2, not JevBench.** Different task, different corpus. The two numbers are not
  interchangeable and neither predicts the other.

## Jev comparison

Running both arms on the same input file is evaluation, which MCA 2.3(b) permits. Publishing the
comparison waits on counsel for 2.3(b) and 14.1. Numbers stay internal until then.
