# Laya vs Harriet, one host and one scorer, 2026-10-23

Both arms re-run here. Nothing in this file is lifted from an earlier record or a leaderboard.

## Why this exists

The two models already had numbers, and they were **not comparable**. Laya's published evidence was
measured on released 0.5.4 on macOS; Harriet's accuracy came from `../2026-09-24-decision-latency/`
on **0.3.46** on a Hetzner CCX33, and its latency from `../../models-decisions/benchmark-results/
2026-09-21-jev-sidebyside/` on RunPod Zen5 boxes. Three libraries, three hosts. Comparing across
them would have produced an ordering of press releases, not a measurement.

## Protocol

| | |
| --- | --- |
| cohort | `../2026-09-24-decision-latency/clm-arm/cohort-jevbench-120.jsonl`, 120 items, sha256 `4bcce87d40fd8edefd8984297d76390794c9fe2613c77fb9bb334006f5969b02` |
| cohort provenance | byte-identical to the file Laya's published evidence names in its own `cohortSha256`, verified by digest, not by filename |
| host | one macOS x86_64 box, 12 processors, 32 GB, both arms back to back |
| library | one local build of `main`; both arms loaded the same `backend-java` classes |
| scoring | `../2026-09-24-decision-latency/options-first/score_arm.py`, which imports JevBench's own `composite_v12`, `metrics` and `scoring` rather than reimplementing them |
| Laya arm | `LayaLatency.java` (retained here), `ModernBertDecisionModel.open` → `decide`, options built exactly as `LayaJevBenchQualificationIT` builds them |
| Harriet arm | `com.integrallis.models.decisions.JevBenchRunner`, `-Ddecisions.letterLogits=true`, context 4096, prefill batch 512, released `backend-native-0.5.4.jar` from Central on the classpath |

Both raw arm outputs and their digests are in this directory.

## Result

| axis | Laya GGUF Q8_0 | Harriet (Qwen3.5-4B Q4_K_M) |
| --- | ---: | ---: |
| overall accuracy | 94/120 — **0.7833** | 106/120 — **0.8833** |
| easy tier | 46/48 — 0.9583 | 48/48 — **1.0000** |
| judge tier | 48/72 — 0.6667 | 58/72 — **0.8056** |
| Intelligence | 76.4 | **87.0** |
| Calibration (ECE) | 67.2 (0.1638) | **75.9 (0.1203)** |
| Speed | **61.8** | 55.9 |
| p50 per decision | **3.318 s** | 6.203 s |
| p95 per decision | **4.800 s** | 10.130 s |
| one-off cost | **0.22 s** model load | 0 here — see below |
| artifact | **428 MiB** | 2.55 GB |

**Harriet is 10 accuracy points better and better calibrated. Laya is 1.9x faster per decision and
6x smaller.** They disagree on 26 of 120 items, so neither dominates item by item.

### Per family

| family | Laya | Harriet | n |
| --- | ---: | ---: | ---: |
| policy | **0.8333** | 0.6667 | 12 |
| fact | 0.9167 | **1.0000** | 12 |
| intent | 0.7917 | **0.8750** | 24 |
| ordinal | 0.7500 | **0.8333** | 12 |
| extraction | 0.8750 | **1.0000** | 24 |
| adequacy | 0.5833 | **0.7500** | 12 |
| routing | 0.4167 | **0.8333** | 12 |

`policy` is the one family Laya wins, and `routing` is where it is weakest by a wide margin.

## Cross-validation

The scorer independently reproduced Laya's **published** figures from this run's raw TSV:
accuracy `0.7833` against the catalog's `0.783333`, ECE `0.1638` against `0.163770`. Two unrelated
paths — the qualification IT and this probe plus JevBench's scorer — agree, so neither the probe
nor the published evidence is a measurement artefact.

Laya's per-decision latency also reproduced across two runs: p50 3264.85 ms and 3317.58 ms.

## What this does not establish

- **Not a dedicated timing host.** This is a shared laptop, not the CCX33 that
  `../2026-09-24-decision-latency/REPRODUCE.md` requires. The *ratio* is same-host and back to
  back, so it is meaningful; the absolute seconds are not comparable to any published figure and
  must not be quoted as a product latency.
- **Harriet ran with no prefix reuse — 0 of 120, by its own report line.** That is not a setup
  failure and not a missing capability: **every JevBench item carries its own state, so there is no
  common prefix to reuse.** `../2026-09-24-decision-latency/REPRODUCE.md` says so directly — the
  per-question latency the release is about is not measurable on this cohort. **This table is
  therefore Harriet's unshared mode, which is its worst case**, and its amortised profile
  (`prefill 5.89 s` once, then `~0.42 s` per question) is a different and legitimate measurement
  recorded in `../../models-decisions/benchmark-results/2026-09-21-jev-sidebyside/`.

  **Correction, 2026-10-23.** An earlier version of this file claimed that amortised profile was
  unreproducible because "the mechanism does not exist for Harriet's own base architecture." That
  was wrong, and it was wrong in an instructive way: it read one capability flag and inferred the
  absence of a different mechanism. There are two, and they are not interchangeable:

  | capability | qwen35 | what it does |
  | --- | --- | --- |
  | `supportsSharedPrefixes` | false | freeze one prefix and `fork()` it into parallel branches — batch candidate scoring |
  | `supportsResumption` | **true** (`Qwen35DecoderAdapter:153`) | return to a position already read without re-reading it |

  Harriet's briefing asks its questions **sequentially** against one document, which needs
  resumption, not forking — and `Qwen35ForwardPass:358` captures the Gated DeltaNet state required
  to resume a retained prefix. The reuse in those records is real, and the numbers in the same file
  prove it independently: a 5.89 s prefill paid once against ten marginals near 0.42 s is not what
  re-reading the document per question would cost.
- **Harriet's accuracy here is 0.8833 against 0.8917 and 0.9000 in the two older records** — one
  to two items, at a cohort whose stated noise floor is one item. Consistent, not identical.
- **No TTFT or TPOT for Laya.** It answers in one non-autoregressive pass; there is no first token
  and no inter-token interval. Reporting a zero or a win on those axes would be fabrication. The
  comparable quantity is time to decision, which is what the table gives.
