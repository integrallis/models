# Does the 256-token output cap explain the domain-model qualification failures?

Measured here, on the qualification fleet (16 vCPU AMD EPYC 7R13, Zen3, AVX2, no AVX-512,
JDK 25.0.4.1 Temurin). Every number below is produced by `derive.py` from the raw artifacts
in `raw/`. Nothing is hand-typed.

```
python3 benchmark-results/2026-10-09-max-tokens-sweep/derive.py
```

## Why this was run

Several domain-specialist models failed `production-rag-model-contribution-v6` with
`modelAnswerRate = 0.000` while showing a non-trivial `truncatedAnswerRate`. Reasoning-tuned
models (HuatuoGPT-o1, Qwen2.5-Math, Fin-R1) emit long chains of thought, and the harness capped
output at a hard-coded 256 tokens. The suspicion was that the cap, not the model, was failing
the gate: a model whose answer is cut off mid-sentence cannot produce a citable statement, so
the grounding policy falls back to extraction and the model is scored as contributing nothing.

That is a real mechanism, and it had to be separated from genuine underperformance.

## Decision rule — registered before the data arrived

| `truncatedAnswerRate` | verdict | conclusion |
| --- | --- | --- |
| falls to ~0 | changes | the cap was the cause |
| falls to ~0 | unchanged | the model genuinely does not contribute |
| does not fall | either | the cap was not the binding constraint |

## Design

Paired, one variable: the same artifact sha against the same corpus sha, run at
`maxOutputTokens = 256` and again at `768`. `derive.py` refuses to pair two runs whose
`artifactSha256` or `corpusSha256` differ.

The `mt=256` cells were produced by backend build `48caca0dafed` and the `mt=768` cells by
`7ac41536f5c2`, so the pairs do cross a backend-build change. `derive.py` admits such a pair
only after resolving the git delta between the two builds and confirming it touches nothing on
the measured path — here 9 files, all in `models-audio/` (a module a text-generation run never
loads) and `scripts/fleet/` (the harness that sets the swept knob). If the delta had touched
`backend-java/`, the Rust shim or the RAG harness, the pair would have been rejected as
inadmissible rather than reported.

`derive.py` is fail-closed about this. git is the authority on the delta; the answer is
written to `backend-delta/<a>..<b>.txt` and any existing copy is checked against it, so a
rewritten history is caught rather than absorbed (verified: corrupt the record and the script
exits 1 with "History was rewritten under this evidence"). If git cannot answer at all — a
shallow clone, or the branch pruned — the committed record is used and the script says so in
its output. With neither, the pair is **rejected**, not admitted.

Getting the cap to be per-job at all was itself a fix: `scripts/fleet/qual-worker-two-arm.sh`
had 256 hard-coded in both arms, so before `7ac41536f5c2` this sweep could not be expressed.

## Result

Truncation was real and large, and raising the cap eliminated it completely:

| model | workload | `truncatedAnswerRate` 256 → 768 | `modelAnswerRate` 256 → 768 | verdict |
| --- | --- | --- | --- | --- |
| huatuogpt-o1-7b Q4_K_S | healthcare | 0.333 → 0.000 | 0.000 → 0.000 | unchanged |
| qwen2.5-math-1.5b Q4_0 | math | 0.778 → 0.000 | 0.000 → 0.000 | unchanged |
| qwen2.5-math-1.5b Q4_K_S | math | 0.556 → 0.000 | 0.000 → 0.000 | unchanged |

`extractiveFallbackRate` is **0.889 in all six cells** — identical before and after. The models
were given three times the room, used it (`totalOutputTokens` rose 4938 → 6135, 6225 → 8046,
6294 → 7866), stopped being truncated, and still produced nothing the grounding policy accepts
as a model answer.

**This is a null, and the null is the result.** By the rule registered above: the cap was a
genuine defect in the harness and is now fixed, but it was not what failed these models. They
genuinely do not contribute on these corpora.

## What this does and does not say

- It does **not** say reasoning models cannot qualify. It says these three artifacts do not, on
  these corpora, under this grounding policy.
- `huatuogpt_o1_7b_q4_k_m` is already QUALIFIED in the catalog on `general`. So for this model
  family the healthcare corpus is *harder* than the general one — the inverse of the EuroLLM
  case, where a `general` failure became a `multilingual` QUALIFIED with workload the only
  variable changed. Workload match cuts both ways and neither direction is predictable from
  the model's name.
- Seven further cells are measured at `mt=256` only and `derive.py` lists them as UNPAIRED with
  no conclusion drawn. Two more (`fin_r1_7b_q4_k_m` finance, `bartowski_qwen2_5_math_7b` math)
  were still running when this was written; re-run `derive.py` after syncing their shards and
  the table extends itself.

## Consequence for the harness

`maxOutputTokens` is now a per-job field, defaulted to 256, settable per model. A reasoning
model should be qualified at a cap that lets it finish; a cap that truncates a third to
three-quarters of answers makes the resulting `modelAnswerRate` uninterpretable, which is
exactly the trap this sweep was run to close. The numbers above are the first measurements on
this harness that are *not* subject to it.

## Provenance

- `raw/mt{256,768}-{workload}-shard{N}/` — the candidate report, comparator report and verdict
  for every cell, synced verbatim from the fleet result bucket.
- `max-tokens-sweep.csv` — generated by `derive.py`.
- `backend-delta/48caca0dafed..7ac41536f5c2.txt` — the 9-file delta between the two backend
  builds, recorded so this directory can re-derive its own admissibility check.
- `derive.py` also checks this file: every rate and token count stated in the table above must
  match the derived pairs, or it exits 1. The prose cannot drift from the data.
- Backend: `models@0.3.56-dev+q41-48caca0dafed` (mt=256) and `-7ac41536f5c2` (mt=768). Both are
  **unreleased dev builds**; `derive.py` prints that warning itself. No verdict here is eligible
  to land in the catalog until it is re-measured against a released build.
