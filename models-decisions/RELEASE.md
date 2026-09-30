# Private release — squad2 Noul v0.1

A decision artifact you can run. Not a benchmark claim; see the limits below before quoting it.

## Run it

```
./gradlew :models-decisions:decide \
  -Partifact=../models-decisions-release/squad2-noul-v0.2.idsn \
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

`squad2-noul-v0.2.idsn`, 82,088 bytes,
sha256 `d1b7f3d54d1c04218e3986db66d217490220239b3cc97623a8978ecfb0bce274`,
fitted against base sha256 `662b0626cd58f443baea23559b469df6576a81d349649c59413b36a9fb32eb29`.

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
- **Check the kernel line on every run.** The tasks below build and attach the rust-ffm kernel for
  the host platform, and print which one loaded. It is worth 7-8x: on this 2019 Intel Mac the same
  ten-question run is 28.3 s with rust-ffm and 211.9 s on the pure-Java fallback. A number taken
  from the fallback is not comparable with one taken from the kernel.
- **0.8000 is squad2, not JevBench.** Different task, different corpus. The two numbers are not
  interchangeable and neither predicts the other.

## Jev comparison

Running both arms on the same input file is evaluation, which MCA 2.3(b) permits. Publishing the
comparison waits on counsel for 2.3(b) and 14.1. Numbers stay internal until then.

## Many questions against one document

```
./gradlew :models-decisions:briefing \
  -Partifact=../models-decisions-release/squad2-noul-v0.2.idsn \
  -Pbase=$HOME/.jvllm/models/granite-4.1-3b-Q4_K_M.gguf \
  -Pdoc=document.txt -Pquestions=questions.txt -Ptimings=timings.json
```

The document is prefilled once and frozen; every question forks that physical prefix. Measured on
this Mac with rust-ffm: 13.85 s prefill, then 1.449 s per question. Prefix sharing is asserted with
a second witness fork on every question rather than assumed.

## Base identity is checked before any inference

The artifact stores the base file's SHA-256 and refuses to run against a file that disagrees. This
is not theoretical: the HuggingFace copy of `granite-4.1-3b-Q4_K_M.gguf` is the **same byte size**
as the one this head was fitted on, 2,099,501,664, with a different digest. A head fed hidden
states from different weights returns confident nonsense and nothing errors.

## Side by side against Jev

`benchmark-results/2026-09-21-jev-sidebyside/` holds the video, the raw timings and `run_demo.sh`.
Ten questions on a master services agreement: Jev 10/10 in 2.75 s, ours 7/10 in 18.86 s on an
8-core EPYC. We lose on both. The three misses are clauses the contract answers by exception or
negation, which is what squad2 does not teach.
