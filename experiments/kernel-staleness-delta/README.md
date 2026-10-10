# How wrong was the September kernel?

**Registered before any data was collected.** Protocol frozen here; results go in `RESULTS.md`.

## What happened

`qual-worker-two-arm.sh` fetched the Rust kernel under the fixed S3 name
`models-kernels-linux-x86_64.jar`. The object under that name was built **2026-09-28** from
`ee8dab27`. Every fleet measurement in the 2026-10-09 campaign therefore ran that kernel against a
0.3.55/0.3.56 Java library. Both declare `abi=6`, so it loaded cleanly and no report showed a
symptom.

| | sha256 | bytes |
| --- | --- | --- |
| released 0.3.56 (`v0.3.56`) | `064cfae7…` | 485,960 |
| the one actually used (`ee8dab27`) | `ce25a986…` | 480,536 |

## What differs in the code, read from the diff

`git diff ee8dab27 v0.3.56 -- backend-native/src/main/rust` is +1658/-9. Of that, `neon.rs` (874
lines) is ARM and cannot affect a linux-x86_64 run. In `lib.rs` the x86-relevant changes are:

1. **Q5_1 support, entirely new.** `CAPABILITY_Q5_1_F32_BATCHED_MATMUL` (bit 23) plus scalar and
   AVX2 kernels. The September kernel does not advertise bit 23, so Java never routed Q5_1 to it
   and those projections fell back to the Java path. Affects only artifacts carrying Q5_1.
2. **A worker-pool activation-gate fix** (`fd0de371`). A published job wrote its own partition
   count into `shared.partitions`, which is the pool's activation gate rather than the job's
   partition count. Its own comment: *"harmless only while the two were always equal; once a job
   can ask for fewer…"*. **This is a shared path and applies to every model**, which is why this
   experiment does not assume the delta is Q5_1-only.
3. **Two F32-activation scratch-sizing fixes**, which exist because Q5_1 introduced the F32
   activation format.

## Hypotheses

- **H1 (numerics).** Generated token ids are identical between the two kernels for every model.
  Rationale: the shim is specified to reproduce the Java arithmetic bit for bit, and the gate fix
  changes scheduling rather than arithmetic. A divergence here would be far more serious than a
  throughput difference and would mean landed evidence is wrong, not merely mislabelled.
- **H2 (throughput, non-Q5_1).** For a model with no Q5_1 tensors, the activation-gate fix alone
  moves decode throughput by some amount `d`. Direction unknown, magnitude unknown. Measuring `d`
  is the point.
- **H3 (throughput, Q5_1).** For a Q5_K model the effect is larger than `d`, because Q5_1
  projections move off the Java fallback onto an AVX2 kernel.

## Decision rule, fixed in advance

| H1 | consequence |
| --- | --- |
| token ids identical | the campaign's *correctness* findings stand; only speed-derived fields are suspect |
| token ids differ | every verdict from the campaign is void, including the published nulls, and must be re-run before anything lands |

**Direction decides whether anything must be re-run, and it matters more than magnitude.**

Every gate the campaign applies is a *speed comparison against an ollama comparator measured on the
same host*, and ollama does not use our kernel. So our kernel's throughput moves the ratio
directly and in one direction only.

| direction with the correct `.so` | what the September numbers were | consequence |
| --- | --- | --- |
| **faster (above)** | pessimistic — every candidate ran with a handicap | **No requalification needed.** A QUALIFIED verdict earned on a slower kernel is earned again on a faster one; the margin only widens. `FAILED_RELATIVE_GATE` verdicts become possible *false negatives* — models that might now pass — which is an opportunity to chase later, not a correction to make now. |
| **slower (below)** | optimistic — every candidate ran with an advantage | QUALIFIED verdicts near a gate boundary are suspect and must be re-run before landing. Magnitude then decides how many: <5% boundary cases only, 5–20% any verdict within that margin of a threshold, >20% all speed-derived verdicts. |
| **indistinguishable** | neither | nothing to do but record it. |

That asymmetry is the whole reason to measure direction first on two small models rather than
re-running thirteen. A faster-kernel result costs one box and settles the question; only a
slower-kernel result justifies spending the campaign again.

The relative gate compares candidate against an **ollama comparator measured on the same host**,
and ollama is unaffected by our kernel. So a throughput change on our side moves the ratio directly
and cannot be cancelled out.

## Protocol

One `m6a.4xlarge` (16 vCPU, AMD EPYC 7R13, Zen 3, AVX2, no AVX-512) — the campaign's own instance
type. One payload, `models-rag-bench-0.3.56-v25.tar`, byte-identical in its class files to the
published 0.3.56 jars (462 classes across `models-rag`, `models-runtime`, `backend-java`,
`backend-native`, all verified identical). Two arms differing **only** in which kernels JAR is on
the classpath.

Two models, chosen to separate the two causes:

| model | quant | isolates |
| --- | --- | --- |
| `qwen2_5_coder_0_5b_instruct_q4_k_m` | Q4_K_M | the activation-gate fix alone (no Q5_1 tensors) |
| `qwen2_5_coder_1_5b_instruct_q5_k_m` | Q5_K_M | gate fix **plus** Q5_1 routing |

Both are in the twelve-model vetted set, so the result speaks directly to the pending verdicts.
Same workload, prompt template, context, max tokens, threads, decode threads, warmups and
iterations in both arms, and the seed is already pinned at 42 with temperature 0 and top-k 1, so
token ids are comparable.

Arms run **in both orders** (A,B then B,A) on the same box, because a first-run arm pays page-cache
and JIT costs the second does not, and that asymmetry is exactly the size of effect being hunted.

## Kill criterion

If the two arms cannot be shown to have loaded different kernels — the worker logs each JAR's
`.so` sha256 from its own `native.properties` — the run is void and reports nothing. A measurement
that cannot prove its one variable changed is not evidence.
