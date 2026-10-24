# maple against llama.cpp, layer by layer — 2026-10-24

**This record was corrected after the fact. Read "Corrections" at the bottom first** — three
claims in the original version were wrong, and one of them was wrong in both directions before it
was measured properly. The corrected findings are in the body; the superseded ones are named so
nobody rebuilds an argument on them.

Why maple is **not qualified**, and where exactly it leaves the reference.

Three structural defects were fixed on the way here (PR #300, issue #298): the declared sliding
window was never applied, `swiglu_clamp_exp` was not implemented, and RoPE was applied to the six
global layers that maple gives no positions. All three are verified to engage on the artifact. The
model still does not match llama.cpp, and the remainder is issue #299.

## Provenance

| | |
|---|---|
| artifact | `maple-preview-TQ1_0-head-Q4_K.gguf`, deepgrove/maple-preview-GGUF rev `f5466f918e0c50cdb9d4d47a6f35813509a42a30` |
| sha256 | `54016e4d543bd688829e67103fc85b8396db94b7f8eb3f81fa95884e44393872` |
| oracle | ggml-org/llama.cpp `3f03b17724ad8b69040825f45c43c59b092a0244`, binaries rebuilt at that revision |
| our branch | `fix/maple-swa-and-clamp`, stacked on #297 |
| route | Rust batched-matmul shim (`RustGgufBatchedMatrixKernel.openBundled()`), Panama SIMD 256-bit |
| host | i9-14900HX, AVX2, **no AVX-512**; macOS |

Our side reports both switches live, so this is not inferred from the fact that code was written:

```
sliding-window-attention = ENABLED {pattern-source=declared-per-layer, local-layers=18/24, declared-window=512}
expert-swiglu-clamp      = ENABLED {clamped-layers=24/24}
```

## Greedy continuations, identical prompts

| prompt | tokens | ours | llama.cpp |
|---|---|---|---|
| `The capital of France is` | 5 | ` Paris` (14.1071) | ` Paris.` — **match** |
| `prompt-below.txt` | 355 | `Item 19 (Item 19 (Item ` | `Item 19: the gravel beside the harbor was counted` |
| `long-prompt.txt` | 1222 | `</think>\n\n\n<|im_start|>assistant\n<|im_start|>assistant` | `Item 61: the mortar beside the basalt was` |

The 355-token prompt is **under** the 512 window, so `attentionStartPosition` returns 0 at every
position and the window clips nothing. The divergence is therefore not a masking problem. We also
get `Item 19` — the right number — and then collapse at the next token.

## Where it starts: one token, `The` (785)

Attention is self-only and RoPE at position 0 is near-identity, so this isolates the arithmetic
from the window, the mask and the positions entirely.

| layer | llama.cpp | ours | delta |
|---|---|---|---|
| 0 | -1.676972 | -1.683112 | 0.37% |
| 1 | -6.667455 | -6.681352 | 0.21% |
| 2 | -6.248443 | -6.270636 | 0.36% |
| 3 | -6.232207 | -6.221243 | 0.18% |
| 4 | -6.068861 | -6.057305 | 0.19% |
| 5 | -11.170424 | -11.050081 | 1.08% |
| 6 | -12.525764 | -12.389461 | 1.09% |
| 7 | -11.976501 | -11.951504 | 0.21% |
| 8 | -11.807573 | -11.726268 | 0.69% |
| 9 | -12.542089 | -12.537124 | 0.04% |

Final hidden state (`result_norm`), one token: our sum **50.863013** against **43.202412**.
Logits `[0]`: **3.7826** against **3.4109**; logit sum **-572392.5** against **-527969.0**.

`Paris` is still the argmax at five tokens while every layer is already off, which is why output
plausibility cannot be the gate for this port.

## What is ruled out, by measurement

- **The token embedding.** `token_embd.weight` is F16. An independent Python read of row 785 sums
  to `-0.3830704092979431`; llama.cpp's `embd` sum is `-0.383070`. Exact. This also confirms the
  tensor offsets and makes the rest of the oracle's ladder trustworthy.
- **Tokenization and BOS.** llama.cpp's `inp_tokens{5}` for the five-token prompt matches our
  `[785, 6722, 315, 9625, 374]`. Neither side prepends a BOS.
- **Context clamping.** `contextCapacity = 131072`, equal to the declared `context_length`. 1222
  tokens does not overflow or wrap.
- **The window bound.** `is_masked_swa` for `LLAMA_SWA_TYPE_STANDARD` masks when
  `p1 - p0 >= n_swa` (`src/llama-hparams.h:515`), so visible keys are `p0 >= p1 - n_swa + 1` —
  identical to our `max(0, position - slidingWindow + 1)`. No off-by-one.
- **Expert routing.** llama.cpp softmaxes over all 256, takes the top 8, and divides by their
  clamped sum. Our `selectExperts` softmaxes over the chosen 8, which is the same function — the
  normalizer cancels. maple passes `norm_w` as a hardcoded `true`, matching the renormalizing
  path, which is why it never reads `expert_weights_norm`.
- **Graph shape**, from the oracle's own dump: per-head RMS norm over the 128-wide head dimension
  on Q and K, `ROPE` on layer 0 and absent on layer 3, `SWIGLU_CLAMP` on the expert FFN, no bias
  tensors anywhere in the artifact, no dense lead layers, no shared experts.
- **The dequantizer.** `TernaryDequantizer.trits` is a faithful transcription of
  `dequantize_row_tq1_0` (`ggml/src/ggml-quants.c:2429`): stage widths 32 then 16 for the 48 `qs`
  bytes, base-3 position in the outer loop, the `& 0xFF` wrap on `q * pow3[n]`, `(q * 3) >> 8`,
  `(xi - 1) * d`, scale last, then four planes of `qh`.

## What closed the parity gap: the KV cache dtype

ggml holds `cache_k`/`cache_v` as **F16** — visible as `(f16) SET_ROWS` in the
`llama-eval-callback` dump. `KvCache.HALF_PRECISION` already implemented this behind
`-Dmodels.purejava.halfPrecisionKvCache`, off by default, with a javadoc that already said storing
F32 "made this runtime more precise than the reference rather than equivalent to it."

With it enabled, plus the TQ1_0 x Q8_K kernel (#299, #302):

| | F32 KV | **F16 KV** | llama.cpp |
| --- | --- | --- | --- |
| expert flips, 1 token | 3 of 24 (L7, L11, L17) | **0 of 24** | — |
| `result_norm` sum | 50.863 | **43.202413** | 43.202412 |
| `result_norm[0]` | 0.8398 | **0.9629** | 0.9629 |
| `result_norm[1]` | -1.2193 | **-1.1616** | -1.1616 |

Bit-identical at one token on every printed element. At five tokens expert selection matches on
every layer and the layer-0 residual is within 0.0012 on a 10240-element sum. Raw run in
`ours-355-tokens-f16kv.txt`.

The three expert flips were therefore a **symptom** of the KV dtype, not an independent defect.

## Greedy continuations still diverge at length, and that is the wrong gate

`ours-355-tokens-f16kv.txt` continues `</think>\n\n**Answer: 8**` where llama.cpp continues the
prompt's pattern. Internal consistency is clean — batched prefill is bit-identical to sequential
decode at 5 tokens and again at 355 across eleven batches — so this is not our batching.

Two configurations of our own code diverge wildly from each other on this prompt:

| configuration | 355-token continuation |
| --- | --- |
| F32 KV | `Item 19 (Item 19 (Item ` |
| F16 KV | `</think>\n\n**Answer: 8**  \n\n**Explanation:**` |
| llama.cpp | `Item 19: the gravel beside the harbor was counted` |

A 1.6-bit 20B MoE sits near behavioural forks, so greedy-text equality after hundreds of tokens is
not a stable property to gate on. Per-layer agreement and expert-selection agreement are, and for
qualification proper a workload metric is.

## The qualification run — EC2, not this Mac

The throughput numbers in the original version of this record were taken on a macOS i9-14900HX.
`docs/CAMPAIGN-RUNBOOK.md` says plainly: "Do not burden the local Mac with model downloads, oracle
runs, or benchmarks." They were also taken with `--threads 12`, where every prior certified record
used **8**, and `sameWorkload` requires threads to match. They were not valid for any gate.

Shard 900, `m6a.4xlarge`, AMD EPYC 7R13, 16 vCPU, Linux, `models@0.5.4+maple1-416ccc3d`.
`STATUS=SHARD_COMPLETE`. Report in `ec2-shard-900-report.json`, worker log in
`ec2-shard-900-worker.log`.

| gate term | threshold | maple | |
| --- | --- | --- | --- |
| `modelAnswerRate` | >= 0.333 | **0.7778** | **pass** |
| `modelAnswerCorrectRate` | >= 0.90 | **1.0** | **pass** |
| `truncatedAnswerRate` | — | 0.0 | clean |
| relative gate | ratio | **no comparator ran** | blocked |

p50 decode **12.647 tok/s**, p50 prefill 14.903 tok/s — **3.9x** the 3.25 tok/s measured on the
Mac, which is why the Mac figures were worthless for a ratio.

**The model clears the quality gate. The comparator is what blocks the verdict, for two separate
reasons:**

1. **Ollama cannot read maple at all.** `ollama create` fails during GGUF parsing with
   `Error: unsupported tensor "blk.0.attn_k.weight" size overflows` on Ollama 0.40.11
   (`ec2-shard-900-ollama.log`). TQ1_0 is unsupported there, so Ollama cannot be maple's
   comparator until that changes. Not ours to fix.
2. **The llama.cpp publisher arm never ran, and that was an omission in the bootstrap.**
   `PUBLISHER_READY=0` because `PUBLISHER_REFERENCE`, `PUBLISHER_REPO` and `PUBLISHER_REVISION`
   were left empty. Setting a reviewed `RagPublisherReference` also tightens the threshold from
   0.45 to 0.80, so admitting a specific llama.cpp commit as a publisher reference is a policy
   decision rather than a mechanical step.

On the arithmetic, 12.647 tok/s against llama.cpp at roughly 30-35 tok/s is ~0.35-0.40, under both
0.45 and 0.80. maple is expected to fail the relative gate by ~2.5-3x, but **that is an
expectation and not a measurement** — no comparator arm has been run on the same host.

## Where the time actually goes

A JFR profile (`settings=profile`) puts it in the Java attention path, not the matmul:

| frame | samples |
| --- | --- |
| `LlamaForwardPass.attendGroupHeadByHead` | 142 |
| `LlamaForwardPass.attendGroup` | 137 |
| `KvCache.writeAttentionScores` | 85 |
| `PanamaVectorUtilSupport.matVecDotExact` | 98 |
| `TensorOps.swiGluClamped` | 54 |

So the AVX2 TQ1_0 kernel (#302) is worth its 1.76x — 147.88 s to 83.88 s on a 355-token prefill,
`ours-355-tokens-avx2.txt` — and was never going to close a 10x gap, because the matmul was about
43% of the time.

`swiGluClamped` showing here is **not** a regression introduced with the clamp: `TensorOps.swiGlu`
is also a scalar `Math.exp` loop, so maple pays that cost either way. Checked before claiming it.

Enabling the existing native grouped-attention shim for maple was measured and rejected: +21%
decode (3.217 to 3.909 tok/s at 355-token context) but -12% prefill, and it costs llama.cpp
bit-parity. It does not reach the gate, so it was not shipped.

## Files

| file | what |
|---|---|
| `oracle-ladder-token785.txt` | full `llama-eval-callback` tensor ladder, one token |
| `ours-ladder-token785.txt` | our per-layer residual sums via `LlamaForwardPass.LayerObserver` |
| `oracle-continuations.txt` | llama.cpp greedy output for all three prompts |
| `ours-1222-tokens.txt`, `ours-355-tokens.txt` | our runs, with the diagnostics header and per-step tokens |
| `long-prompt.txt`, `prompt-below.txt`, `prompt-above.txt` | the prompts, byte for byte |
| `MapleLadder.java`, `MapleLongPrompt.java`, `MapleShort.java` | the probes |

## Reproducing

```sh
scripts/oracle-revision.sh          # refuses a stale clone AND stale binaries
# reference ladder
~/Code/llama.cpp/build/bin/llama-eval-callback -m <maple> -n 1 -p "The"
# reference continuations — note llama-simple takes ONLY -m, -n, -ngl, [prompt]
~/Code/llama.cpp/build/bin/llama-simple -m <maple> -n 12 "$(cat long-prompt.txt)"
```

## Corrections

Every item here was asserted in an earlier version of this record, a commit message, or an issue
comment, and every one was wrong. They are named rather than deleted because the wrong versions
were published and somebody may have read them.

### 1. "Throughput makes a qualification shard impractical" — asserted, withdrawn, then confirmed

Three states, only the last of them measured against the right thing.

| | claim | basis | verdict |
| --- | --- | --- | --- |
| first | throughput is the blocker; vectorize before qualifying | 2.08 GMAC/s against what the hardware *should* reach | **unmeasured inference** |
| second | throughput is **not** a blocker, ratio 0.869 | maple 3.248 tok/s against **Qwen3.5-4B** 3.738 tok/s, both on our runtime, on the Mac | **wrong comparator and wrong host** |
| now | relative gate expected to fail by ~2.5-3x | 12.647 tok/s on EC2 against llama.cpp ~30-35; still no same-host comparator arm | **expectation, labelled as one** |

The second was the worst of the three, and the adjudicator caught it, not me:
`exclusions.rust-ffm = unsupported comparator backend`. The policy requires the **same model on a
reference runtime**, not a different model on ours. The relative gate is a ratio against a
comparator — never against hardware capability, which is what the first claim used.

### 2. "The scalar TQ1_0 kernel is the bottleneck"

Concluded from arithmetic and acted on: a document was written saying "vectorize then qualify", and
the AVX2 kernel was built on it. Measured afterwards, the matmul was ~43% of the time and
vectorizing gave 1.76x where 8-16x was predicted. The profile should have come first. The kernel
is still correct and still worth having; the reasoning that produced it was not evidence.

### 3. "None of the nine qxotic ports is qualified, and that is certain"

A tautology presented as a determination. Eight of the nine were never implemented, so they cannot
have failed a gate nobody ran — that describes work not done and says nothing about the models. A
document built on that framing (PR #301) was closed rather than patched.

### 4. `-c 2048` passed to `llama-simple`, which does not accept it

Its usage is `-m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]`. The flag was concatenated
into the prompt text, so the first oracle comparisons were not on identical inputs. Caught when a
short prompt came back `is is is is`. Re-run clean, the continuations are unchanged, so the
conclusions survive — but they were not rigorous when first drawn.

### 5. The first prefill figure came through a route the product does not use

`PureJavaBackend.load(Path)` passes `GgufBatchedMatrixKernel.none()` and
`BatchedCausalAttentionKernel.none()` **by construction**, and the probe's classpath had no
`backend-native` on it at all. A timing from that route is not a measurement of the product. Same
error as the earlier false "Panama falls back to scalar" finding, which came from omitting
`build/resources/main` so the ServiceLoader found no provider.

### 6. The F16 KV cause was reached by hypothesis when it was already written down

Several multi-minute runs went into narrowing the residual divergence. `KvCache.HALF_PRECISION`
and its javadoc — "llama.cpp holds `cache_k` and `cache_v` as F16 … storing F32 made this runtime
more precise than the reference rather than equivalent to it" — were in the file the whole time.
One grep for a dtype flag, the moment the oracle dump showed `(f16) SET_ROWS`, would have found it.

### 7. Benchmarking on this Mac at all

`docs/CAMPAIGN-RUNBOOK.md`: "Do not burden the local Mac with model downloads, oracle runs, or
benchmarks." Every throughput number in the original record violated that, and EC2 then measured
3.9x on the same work. The runbook was read reactively, after the numbers were published, rather
than before the first benchmark.

## Files

| file | what |
| --- | --- |
| `oracle-ladder-token785.txt` | full `llama-eval-callback` tensor ladder, one token |
| `ours-ladder-token785.txt` | our per-layer residual sums via `LlamaForwardPass.LayerObserver` |
| `oracle-continuations.txt` | llama.cpp greedy output for all three prompts |
| `ours-1222-tokens.txt`, `ours-355-tokens.txt` | our runs before the F16 KV change |
| `ours-355-tokens-f16kv.txt` | with F16 KV, the parity configuration |
| `ours-355-tokens-avx2.txt` | with the AVX2 TQ1_0 kernel, 83.88 s prefill |
| `ec2-shard-900-report.json` | the qualification report from the fleet |
| `ec2-shard-900-worker.log` | the worker log, including `PUBLISHER_READY=0` |
| `ec2-shard-900-ollama.log` | Ollama's GGUF parse failure on TQ1_0 |
| `long-prompt.txt`, `prompt-below.txt`, `prompt-above.txt` | the prompts, byte for byte |
| `MapleLadder.java`, `MapleLongPrompt.java`, `MapleShort.java` | the probes |

## Reproducing

```sh
scripts/oracle-revision.sh          # refuses a stale clone AND stale binaries
# reference ladder
~/Code/llama.cpp/build/bin/llama-eval-callback -m <maple> -n 1 -p "The"
# reference continuations -- llama-simple takes ONLY -m, -n, -ngl, [prompt]
~/Code/llama.cpp/build/bin/llama-simple -m <maple> -n 12 "$(cat long-prompt.txt)"
# parity configuration
java -Dmodels.purejava.halfPrecisionKvCache=true ... MapleShort <maple> "The" hidden
```

Throughput and gate numbers reproduce on the fleet, not here:
`BOOTSTRAPS=<dir> bash scripts/fleet/run-shards.sh <shard>`.
