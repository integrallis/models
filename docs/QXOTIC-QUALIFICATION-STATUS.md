# qxotic ports: qualified or not — 2026-10-24

**None of the nine is qualified.** That is certain and it is not a provisional answer: in this
repository "qualified" means a workload run that emitted `qualified: true` into a qualifications
manifest, under a workload matched to the model's declared capabilities, with
[two arms measured on the same host in the same session](CAMPAIGN-RUNBOOK.md). No such run exists
for any of the nine. Eight of them have never been implemented, so there was nothing to run.

What follows is the determination per port, the evidence behind it, and the specific thing each
one is waiting on. Numerical parity with llama.cpp — which is what the maple work below measured —
is a **precondition** for attempting qualification, not qualification itself. Conflating the two
is how a port gets called done.

## Summary

| port | implemented | numerically validated | qualified | waiting on |
| --- | --- | --- | --- | --- |
| `maple` | **yes** | **partially — see below** | no | throughput, then a workload run |
| `mellum` | no | — | no | the port; #281 |
| sigmoid + `exp_probs_b` routing | no | — | n/a | the shared unit; #282 |
| `lfm2moe` | no | — | no | routing above, then the port; #283 |
| `parakeet` | no | — | no | primitives that do not exist in the tree; #284 |
| `kokoro` | no | — | no | a G2P licensing decision; #285, #286 |
| `inflect-v2` | no | — | no | same licensing decision; #285, #286 |
| `nemotron_h_moe` | no | — | **not qualifiable** | nothing — see below |
| `laguna` | no | — | **not qualifiable** | nothing — see below |
| `bailingmoe3` | no | — | unassessable | an artifact to exist; #288 |

## `maple` — implemented, numerically validated at small scale, not qualified

Four defects were found and fixed (#298, PR #300), each against upstream source and the oracle's
own graph dump rather than against output plausibility:

1. the declared sliding window was applied to **no** layer, because the gate was
   `isGemmaFamily()`;
2. `swiglu_clamp_exp` was not implemented at all, in any file;
3. RoPE was applied to the six global layers maple deliberately gives **no** positions;
4. TQ1_0 products used an exact-F32 path where ggml quantizes activations to Q8_K (#299).

A fifth was not a defect but a configuration: llama.cpp holds `cache_k`/`cache_v` as **F16**, and
`KvCache.HALF_PRECISION` already existed behind `-Dmodels.purejava.halfPrecisionKvCache`, off by
default.

### What is proven

With the four fixes and F16 KV enabled, against ggml-org/llama.cpp `3f03b1772`:

| scale | result |
| --- | --- |
| 1 token | `result_norm` **bit-identical** — our sum 43.202413, llama.cpp's 43.202412, every printed element equal |
| 1 token | expert selection identical on **24 of 24** layers |
| 5 tokens | expert selection identical on every layer |
| 5 tokens | layer-0 residual within **0.0012** on a 10240-element sum |

Internal consistency, which needs no oracle: batched prefill and sequential decode are
**bit-identical** at 5 tokens and again at 355 tokens across roughly eleven batches.

### What is not proven, stated plainly

Greedy continuations at 355 and 1222 tokens **do not** match llama.cpp. The residual difference
grows with depth and position — 0.0012 at layer 0 of a 5-token prompt, 0.23 by layer 5 — and the
remaining known difference is floating-point accumulation order: llama.cpp uses `FLASH_ATTN_EXT`
and we use a separate grouped-query path.

**That prompt is a poor instrument and should not be used as a gate.** Two configurations of our
own code produce wildly different continuations on it:

| configuration | continuation at 355 tokens |
| --- | --- |
| F32 KV | `Item 19 (Item 19 (Item ` |
| F16 KV | `</think>\n\n**Answer: 8**  \n\n**Explanation:**` |
| llama.cpp | `Item 19: the gravel beside the harbor was counted` |

A 1.6-bit 20B MoE sits near behavioural forks, and greedy-text equality after hundreds of tokens
is not a stable property to gate on. The sound criteria are the ones already measured — per-layer
numerical agreement and expert-selection agreement — plus, for qualification proper, a workload
metric.

### The throughput precondition

**Prefill of 355 tokens takes 147.88 s.** From the artifact's own dimensions that is 868.2 MMAC per
token, so 308.2 GMAC in 147.88 s = **2.08 GMAC/s** on an i9-14900HX with AVX2 and 12 worker
threads, which should exceed that by more than an order of magnitude. The cause is known and
recorded: the TQ1_0 x Q8_K kernel added in #299 is **scalar**, with no AVX2 variant, and 168 of
maple's tensors are TQ1_0 including all three expert matrices (25M of the 36M MACs per layer).

A RAG qualification shard runs many cases over two arms. At this rate it is hours per model of
fleet time to answer a question that a vectorized kernel would answer in minutes. **Vectorize
first, then qualify** — otherwise the campaign pays for the missing kernel.

### So: `maple` is not qualified

It is implemented, it runs, it reproduces llama.cpp exactly at one token and selects the same
experts at five, and it has never been put through a workload. Nothing here licenses a catalog
entry, and maple is correctly **absent** from `model-jars/catalog/`.

## `nemotron_h_moe` and `laguna` — not qualifiable, and that is final

Closed as #287. The smallest published artifacts are **18.09 GB** and **20.27 GB**
(`benchmark-results/2026-10-22-qxotic-header-intake/repo-file-sizes.txt`). This is a property of
what the publishers shipped, not a backlog item that clears with effort, and it should not be
re-opened as one. `laguna` additionally needs the sigmoid routing unit (#282).

## `bailingmoe3` — unassessable

Absent from qxotic's own test manifest; no artifact has been located (#288). "Unassessed" is not
"assessed as hard", and the issue title says so.

## The four that need code

- **`mellum`** (#281) — 8.07 GB Q4_K_M, 28 blocks, 2304 dim, **64 experts / 8 used**, YaRN x16,
  SWA 1024. Its issue title claimed SWA and YaRN were "both already present"; maple disproved the
  SWA half, and the title is corrected. It needs the **uniform-window** case that neither maple's
  declared-pattern path nor Gemma's stride covers: mellum declares a window and no pattern and
  makes every layer local (`src/models/mellum.cpp:13-14`). Also `MELLUM` in
  `DecoderArchitecture.seedsSlidingWindowRopeFromBase()`, and a regression prompt longer than its
  1024 window.
- **sigmoid + `exp_probs_b` routing** (#282) — `ExpertRouting` implements softmax gating only. Pure
  Java, and the shared prerequisite for `lfm2moe` and `laguna`.
- **`lfm2moe`** (#283) — 4.84 GB, 32 experts / 4 used, short-conv, and **two leading dense blocks**.
  That last part is a genuine new requirement: maple has no dense lead layers, so nothing in the
  tree exercises the split today.
- **`parakeet`** (#284) — needs a transcription model class, FastConformer, a 2-layer LSTM
  predictor and a TDT joint. `Mamba-2`/selective scan, `LSTM` and `conformer` are each **0 files**
  in this tree, verified by grep.

## The two that need a decision, not code

`kokoro` (0.16 GB) and `inflect-v2` (0.005 GB) are the cheapest artifacts on the list and both are
blocked on the same thing: a **G2P/phonemizer licensing decision** (#285). No phonemizer exists in
the tree, and CLAUDE.md §5 requires permissive licensing for new runtime components. Porting either
before that decision risks packaging something that cannot ship.

## Calibration, for the next port

`docs/QXOTIC-PARITY-FEASIBILITY.md` lists maple's new primitives as "none (+ read and assert
`expert_gating_func`)". Measured, maple needed four code changes, a new Rust kernel, a
configuration change, and a day. It was also the **easiest** port on the list: the smallest
artifact with a working oracle and no missing primitives.

Treat "new primitive: none" in that table as "no primitive we could name from the header", not as
an effort estimate. The two entries that inherit the same claim — `mellum` and `lfm2moe` — should
be read the same way until measured.

## Open decisions that are not mine

1. **Is F16 KV the default?** It is what llama.cpp does, it halves KV memory, and maple needs it
   for parity. It is off because it flips a greedy token on the pinned SQLCoder Q5_K_M fixture,
   whose expectation was captured from **llama.cpp b9960** — far older than the `3f03b1772` oracle
   used here. Since llama.cpp's own KV default is F16, one of those two references is stale, and
   deciding which requires the SQLCoder artifact (~4.8 GB, not on this disk) run against current
   llama.cpp. Until then maple's parity result must state the flag, which it does.
2. **Does `maple` get fleet time?** Only after the kernel is vectorized, on the arithmetic above.
