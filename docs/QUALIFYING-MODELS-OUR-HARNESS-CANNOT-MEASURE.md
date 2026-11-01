# Qualifying models our harness cannot measure

Some candidates are blocked not by their own throughput but by
`RagProductionQualificationPolicy` being unable to express a verdict about them. This document names
a harness for each, states what it gates, and derives those gates from the standard one rather than
inventing them. Each recipe is anchored to a number the model's creator published, because a gate
with no external anchor only scores our own assumptions back to us.

**Status: design. None of these harnesses exists yet, and no model below has been qualified through
one.** What follows is the specification to build against, written down before any data is collected
so the gates cannot be chosen after seeing the results.

**Section 3 is a correction, not a design.** It originally specified a third harness for TTS. A
TTS gate already ships — `speech-oracle-streaming-latency-v1`, two models qualified — so that
design is withdrawn and what remains is the one narrow case the shipped gate cannot express. Writing
a gate for a modality that already had one is the failure this document was supposed to avoid; the
cure was reading `catalog/speech-qualifications.json` and running the existing gate, which took
under an hour and no rented compute.

## What the standard harness actually guarantees

`RagProductionQualificationPolicy` (`production-rag-model-contribution-v6`) is worth copying because
of five properties, not because of its particular numbers:

1. **It measures the model, not the pipeline.** `modelAnswerRate >= 1/3` and
   `modelAnswerCorrectRate >= 0.90` count only runs where `grounding.decision.modelContributed()`,
   because `GroundedAnswerPolicy` substitutes extracted text when a generation is screened out and a
   model that never answers can otherwise read as 1.000 correct.
2. **It compares against a reference nobody here wrote,** on the same host, in the same session, on
   the identical artifact: `decodeThroughputRatio >= 0.80`, `endToEndLatencyRatio <= 1.50`.
3. **It refuses a comparison it cannot trust.** `sameWorkload()` excludes a comparator over corpus,
   cases, template, top-k, output cap, context, **threads**, grounding policy and generation
   controls; a differing artifact SHA-256 excludes it outright.
4. **A single run is not a verdict.** Learned expensively on 2026-11-01: one mellum pair returned
   `QUALIFIED` at 0.8112 and three replications returned 0.7386, 0.7862 and 0.7643. The comparator's
   own spread was about 8%.
5. **Every switch is observable.** A toggle that silently does nothing produces a flat result that
   reads as "this does not matter" when the truth is "this never ran".

Any substitute harness has to keep all five. The metrics change with the modality; these do not.

---

## 1. `CrossArtifactPublisherQualification` — generative models no authoritative engine will load

**Policy id:** `production-rag-cross-artifact-v1`
**Blocks today:** `maple` (deepgrove/maple-preview, TQ1_0 ternary)

### Why the standard harness cannot answer

`AUTHORITATIVE_ENGINES` is `{ollama, transformers}`. ollama refuses the artifact outright —
`Error: unsupported tensor "blk.0.attn_k.weight" size overflows` — and the publisher's transformers
path loads the **safetensors**, not our GGUF. So the two arms can never present the same
`artifactSha256`, and `exclusion()` drops the comparison before any number is computed. The blocker
is the identical-artifact rule, which exists for a good reason and is simply too strong here.

### What it gates

Everything the standard policy gates, with one rule replaced:

- **Contribution gate: unchanged.** Same nine cases, same corpus SHA-256, same rendered prompts, and
  the same `0.3333` / `0.90` floors. This is modality-identical work, so there is no reason to move
  it.
- **Identical artifact, replaced by a declared equivalent pair.** A reviewed entry pins *both*
  digests — the publisher's safetensors and our GGUF — plus the conversion provenance: converter
  name, version, and the exact command. Modelled on `RagPublisherReference`, which already pins a
  reviewed per-artifact reference; this extends the idea from "the publisher's own llama.cpp" to
  "the publisher's own runtime and weights".
- **Relative gate: `0.80` / `1.50`, with a compute-class rule the standard policy never needed.**
  Both arms must run on the same compute class. A GPU comparator against our CPU decoder is
  **inadmissible** — not merely unflattering — because it measures the hardware, not the
  implementation. If the publisher's runtime only runs on an accelerator our decoder does not use,
  the honest verdict is `NO_COMPARABLE_BASELINE`, and the harness must return that rather than a
  ratio.

### Anchor

deepgrove publish **218 tok/s on an M4 Mac mini** via their own on-device runtime, and a 5.31 GB
checkpoint at 20B total / ~1B active. That figure anchors the order of magnitude but is not the
comparator: it is a different host and a runtime we do not have. The admissible comparator is their
transformers or vLLM path on the same box as our arm.

### Honest limit

If maple's advertised transformers path needs FlashAttention 3 (its `fa3.py` prefers
`flash_attn_interface`, which is Hopper-only) and vLLM also declines the architecture, then there is
no same-compute-class comparator and maple stays `NO_COMPARABLE_BASELINE`. That is a real outcome,
not a harness failure, and the harness should say which of the two it is.

---

## 2. `AsrProductionQualificationPolicy` — speech recognition

**Policy id:** `production-asr-transcription-v1`
**CLI:** `AsrBenchmarkCli`
**Blocks today:** `parakeet` (#284)

### Why the standard harness cannot answer

`RagBenchmarkCli` measures grounded text generation. An ASR model consumes audio and emits a
transcript; there is no prompt, no retrieval, no grounding decision, and no decode-tokens-per-second
in the sense the policy means. Nothing in the existing harness applies. This is the case where
"build a custom harness" is the only answer.

### What it gates, mapped from the standard gate

| standard gate | ASR equivalent | why |
|---|---|---|
| `modelAnswerCorrectRate >= 0.90` | **WER <= the comparator arm's WER on the same audio under the same judge, within that arm's replication spread** | the direct analogue of "when the model answered, it was right", measured against an arm we run rather than a number we read |
| `modelAnswerRate >= 1/3` | **no substitution path exists** — every utterance yields a transcript, so the rate is 1.0 by construction and the gate is dropped rather than faked | inventing a second quality gate to mirror the shape would be ceremony |
| `decodeThroughputRatio >= 0.80` | **RTFx ratio >= 0.80** against the publisher's runtime, same host, same audio, same batch size | RTFx is already the publisher's own throughput unit |
| `endToEndLatencyRatio <= 1.50` | **same**, over per-utterance wall clock | unchanged in meaning |
| `sameWorkload()` | identical audio set, sample rate, batch size, beam settings **and text normalizer** | see below |

### The trap this harness must close

WER is trivially flattered by normalization. Lowercasing, stripping punctuation, expanding numerals
and dropping filler words each move it, and the published leaderboard numbers assume one specific
normalizer. **The recipe pins the Open ASR Leaderboard's Whisper text normalizer by version and
records it in the report**, exactly as `sameWorkload()` pins the prompt template. An unpinned
normalizer is this modality's version of `correctAnswerRate` — a number that looks like model
quality and is not.

### Anchors

Published by NVIDIA on the Hugging Face Open ASR Leaderboard:

| model | published WER | published RTFx |
|---|---:|---:|
| `parakeet-tdt-0.6b-v2` | 6.05% avg | 3380 at batch 128 |
| `parakeet-tdt-0.6b-v3` | 6.34% avg | 3332.74 |
| `parakeet-rnnt-1.1b` | 7.04% avg | RTF 14.4 (1e-3) |

**The `x 1.10` margin this section originally carried is withdrawn.** It was picked with no
derivation, and the review flagged it as having the appearance of rigour without the substance.
Replacing it with a derivation from the leaderboard's own rerun spread was the other option offered,
and it is not available: the leaderboard publishes one number per model, not a spread.

What replaces it is the form measured on 2026-11-01 in
`benchmark-results/2026-11-01-tts-roundtrip-judge-validation/`. Scoring a transcript against a
reference is dominated by the corpus and the normalizer, not by the model: on ten sentences a
compounding convention cost **0.250** on a sentence whose audio was correct, and an initialism cost
**0.167**. A floor expressed as a multiple of somebody else's number silently inherits their corpus
and their normalizer, and a 10% margin is smaller than the artifacts just measured. So the floor is
**relative to a comparator arm we run on the same audio with the same pinned judge**, and the
published figures below are context for whether a port is in the right regime at all -- never the
threshold.

The two corpus rules from that validation carry over: no initialisms unless measuring them, and no
words whose compounding is unsettled unless the charge is accepted and recorded.

### Reference runtime

NVIDIA NeMo, pinned by version, on the same host. Not an authoritative engine in the current policy
because the current policy has no ASR notion at all; `AUTHORITATIVE_ENGINES` for this policy is
`{nemo}` and the reason is that NVIDIA run parakeet with it themselves.

---

## 3. TTS — withdrawn as written; a second gate is needed for one stated reason

**Corrected 2026-11-01.** This section originally proposed
`TtsProductionQualificationPolicy` / `production-tts-synthesis-v1` as a new harness. That was
wrong: **a TTS gate already ships.** `speech-oracle-streaming-latency-v1` is recorded in ModelJars'
`catalog/speech-qualifications.json` and has already qualified both Soprano entries. It gates
waveform parity against the publisher's own implementation at a pinned oracle revision
(`minimumPcmCosine` 0.998389 and 0.999331, `minimumSignalToDifferenceDb` 24.9 and 28.7) plus
streaming behaviour and p95 real-time factor and time-to-first-audio over five trials.
`docs/MODALITY-EXPANSION.md:67` already said an ASR gate should be built in that gate's shape. The
design below was written without reading either.

Reproducing it on Soprano is what surfaced this:
`benchmark-results/2026-11-01-soprano-tts-gate-reproduction/`. That run also found the gate's own
two weaknesses — nothing in the repository computes its performance block, and two five-trial
passes of the same artifact on the same machine differ by **2.68x** in p95 real-time factor.

**What still needs a new gate, and only this:** a vocoder that is nondeterministic by construction.
Paradee-8M v1.1's ONNX graph contains `RandomNormalLike` and its card says identical inputs give
varying output in the noisy components, so a waveform-cosine comparison against an oracle has no
reference waveform. For that model class — and no other — round-trip WER is the right quality
signal, and the existing p95 RTF / TTFA block is reused unchanged. The alternative is to inject the
noise tensor identically into both sides, which keeps one gate at the cost of an oracle that is no
longer the thing the publisher ships. That is a decision to take, not a measurement to run.

### The round-trip-WER design, kept for the nondeterministic case only

| standard gate | TTS equivalent | why |
|---|---|---|
| `modelAnswerCorrectRate` | **round-trip WER <= threshold**: synthesize a pinned text set, transcribe with a pinned ASR model, score against the input text | the only reproducible objective quality signal available without human raters |
| `decodeThroughputRatio >= 0.80` | **RTF ratio**: our RTF <= publisher RTF / 0.80 | RTF is the publisher's own unit |
| `sameWorkload()` | identical text set, voice, sample rate, and **the pinned ASR judge** | the judge is part of the measurement |

### What this harness must refuse to claim

Kokoro's published standing is a **TTS Arena Elo** — #1 at v0.19 release — and MOS. Both are human
preference judgements. **Round-trip WER measures intelligibility, not naturalness, and the harness
must report it under that name.** A port that scores well on round-trip WER has not reproduced an
Elo ranking and must not be recorded as having done so. This is the same discipline as refusing to
report `correctAnswerRate` as model quality: the available number is not the number people care
about, and saying so is the point.

### Anchors

Published for Kokoro-82M on CPU: **RTF 0.469 (PyTorch), 0.509 (ONNX)** — roughly 2x realtime — at
82M parameters. Our arm must reach RTF <= 0.586 to clear a 0.80 ratio against the PyTorch figure.

Paradee-8M v1.1, the 7.9M distillation of that same Kokoro-82M, publishes **62.1x real time, UTMOS
4.39 and WER 5.9% under Whisper base** on 200 held-out sentences it does not publish. Its own paper
says 25.0x and its own repository description says 18x, each measured in a different setup: a 3.4x
spread across three self-reported speeds for one model, which is the reason none of them can be an
anchor without being re-derived. Triage and digests:
`modeljars/docs/model-evidence/paradee-8m/`.

### Grapheme-to-phoneme is the blocker in front of all of this

`inflect-v2`, the second half of #286, is not a model and does not take this gate: it emits no
audio and has no throughput ratio. Its gate is **exact-match accuracy against a pinned lexicon test
set**, in the unit test suite rather than a qualification campaign. Recording it as a qualified
model would be a category error.

It is also, measured on 2026-11-01, the hard prerequisite for Paradee rather than a parallel
nicety. Paradee's graph input is `input_ids` int64 `[1, tokens]` — **phoneme ids, not text** — and
`grep -riE 'phoneme|grapheme|g2p|espeak|misaki'` over `models-audio/src/main/java` matches nothing;
Soprano takes text and needs none. The publisher reports that feeding eSpeak NG's phoneme spelling
instead of misaki's moves Whisper's word error rate from about 2% to about 31%. A port without a
misaki-equivalent G2P would not be a worse Paradee; every number taken from it would be measuring
the phonemizer.

---

## What is common to all of them, and non-negotiable

- **Check whether the gate already exists.** Section 3 is here because this step was skipped once.
- **Replication.** Three paired runs minimum, and the **minimum across replications** must clear the
  floor; mean and spread is a procedure, not a decision. A verdict taken from the favourable end of
  a comparator's own noise is not a verdict. Measured on the shipped TTS gate on 2026-11-01: two
  five-trial passes of the same artifact, backend, machine and protocol differ by 2.68x in p95
  real-time factor.
- **The comparator is the creator's runtime,** pinned by version, on the same host and compute class.
- **Both arms see identical inputs,** byte for byte, and the report records them.
- **Nulls are published.** A harness built and then found to show no difference is a result.
- **A gate is written down before the data.** Every threshold above is stated in this document, at
  design time, for exactly that reason.
