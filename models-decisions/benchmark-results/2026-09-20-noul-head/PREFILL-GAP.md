# The hidden-state prefill is not batched, and it costs 14x

**Sizing measurement, not a pre-registered result.** Taken on `decisions-bench-noul-20260920`
(Hetzner ccx33, 8 vCPU AMD EPYC-Milan, AVX2, 30 GiB, Ubuntu 24.04.4, Temurin 25.0.4.1),
Granite 4.1 3B Q4_K_M (SHA-256 `662b0626...`, catalog-verified), `PureJavaBackend` with the
host-built `RustGgufBatchedMatrixKernel`. Panama active: `panama=true maxBits=256 fastVectorFMA=true
ggufExecutor=persistent ggufThreads=8`.

## Measured

| Call | Tokens | Time | Rate |
| --- | ---: | ---: | ---: |
| `prefill` — logits, **batched** | 128 | 4,813 ms | **26.6 tok/s** |
| `prefillHiddenState` — **sequential** | 128 | 66,538 ms | **1.9 tok/s** |
| `prefillHiddenState` | 64 | 34,385 ms | 1.86 tok/s |
| marginal question on a frozen prefix | 12 | 6,181 ms | — |

**14.0x**, on identical input, same process, same kernels.

## It is not the poll budget

`vectors.gguf.pollMillis` was the leading hypothesis, because vectors 0.1.22 warns that a caller
running its own pool beside the persistent executor should set 0. Ablated directly:

| `pollMillis` | 64-token hidden prefill | 12-token marginal question |
| --- | ---: | ---: |
| 0 | 34,385 ms (1.86 tok/s) | 6,181 ms |
| 5 (default) | 34,198 ms (1.87 tok/s) | 6,285 ms |

Identical within noise. **Hypothesis refuted.** The poll budget explains none of the gap.

## It is a missing branch, located exactly

`LlamaForwardPass.prefill(Session, int[], int)` line 817:

```java
if (batchedPrefill && tokens.length > 1) {
  return prefillSessionBatched(session, tokens);
}
```

`LlamaForwardPass.prefillHiddenState(Session, int[], int)` line 851 has **no such branch**. After its
validation it goes straight to a token-by-token loop of `forwardSessionInternal(..., Head.NONE)`,
finishing with one `Head.HIDDEN` step. The batched machinery exists and is qualified; the
hidden-state route simply never reaches it.

The reason batching matters so much here is memory bandwidth, not arithmetic. A single-token forward
streams the whole ~2 GB of Q4_K_M weights to produce one row; a batched prefill streams them once
for the whole block. That is why one sequential step costs about 0.54 s while the batched path
averages 0.038 s per token.

## Why this blocks the harvest rather than merely slowing it

The decision tier's economic claim is that N decisions over one state cost one prefill plus N head
reads. Today both halves of that are paid at sequential rates:

| | state prefill (64 tok) | + per question (12 tok) |
| --- | ---: | ---: |
| today | 34.4 s | 6.2 s |
| batched all but the final row | ~2.4 s | ~0.95 s |
| batched path emitting hidden directly | ~2.4 s | ~0.45 s |

A harvest of 1,000 items at roughly 160 tokens each costs about **23 hours** at 1.9 tok/s and about
**1.7 hours** at 26.6 tok/s, on a host billed by the hour.

More important than the cost: any competitive number measured now would be measuring **our own
missing optimization**, not the architecture. Publishing a per-decision latency from this state of
the code would be reporting a pipeline number and attributing it to the wrong stage.

## Two candidate fixes, both in `models`, neither taken here

1. **Surgical.** Give `prefillHiddenState` the same `batchedPrefill` branch: batch tokens `[0, n-2]`
   through the existing qualified path, then one sequential `Head.HIDDEN` step for the final token.
   Semantically identical to today's loop, reuses machinery that is already qualified, and recovers
   most of the gap. The final full-weight sweep remains as a floor.
2. **Deeper.** Let the batched path emit the final row's normalized hidden state instead of the
   vocabulary projection, removing the last sequential sweep too. Larger change, touches head
   handling inside `prefillBatchTransient`.

**Neither is being made on this branch.** `LlamaForwardPass` is live territory for the GPU campaign
(PR #195 edits `computeBatchedAttention` and `attendHeads` in this same file), and a decisions branch
quietly rewriting its prefill would collide. Recorded here for a deliberate decision instead.

---

## Fixed, and measured on the same host

`models` branch `perf/batched-hidden-state-prefill`, commit `d23f440d`. Both `prefillHiddenState`
variants now advance every position but the last through the existing qualified batched path, then
take one ordinary step for the final token. Same host, same artifact, same kernels, same process
shape as the measurement above.

| `prefillHiddenState` | 128 tokens | rate | 512 tokens | rate |
| --- | ---: | ---: | ---: | ---: |
| before | 66,538 ms | 1.9 tok/s | — | — |
| **after** | **4,283 ms** | **29.9 tok/s** | 13,973 ms | 36.6 tok/s |
| `prefill` (logits, batched), for reference | 4,819 ms | 26.6 tok/s | 13,444 ms | 38.1 tok/s |

**15.5x at 128 tokens.** The hidden-state route now runs slightly *faster* than the logits route at
128 tokens and at parity by 512, which is the expected shape: it skips the vocabulary projection on
every row but the last, and that saving shrinks in relative terms as the prompt grows.

Correctness is unchanged and asserted rather than assumed: hidden state, key/value cache contents,
and the next generated token all match running the same prompt one token at a time, within the
repository's existing `SIMD_REDUCTION_TOLERANCE` of 2.0e-7. 681 `backend-java` tests pass.

### Consequence for the harvest

A 1,000-item harvest at roughly 160 tokens an item moves from about **23 hours** to about
**1.5 hours** on this box. The pre-registered protocol is now affordable as written, at all three
corpora, without trimming N.

### A vacuous pass, caught by the counter

Worth recording because it nearly shipped. The first version of the fix's tests used an all-F32 nano
model, and `TensorOps.supportsBatchedMatmul` does not include F32 — so the fixture could not batch at
all, and the three equivalence tests passed by comparing the sequential path against itself. Only
the counter assertion failed, which is the entire reason it was written. The fixture now uses Q8_0
projections at the dimensions Q8_0 alignment demands.
