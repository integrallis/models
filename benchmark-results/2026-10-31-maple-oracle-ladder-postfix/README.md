# maple against llama.cpp, after the TQ1_0 computation fix — 2026-10-31

Re-runs the ladder in `../2026-10-24-maple-oracle-ladder/` against the same pinned oracle, after
`53ad2b17` ("Compute TQ1_0 the way ggml computes it: against Q8_K activations"). #299's body
records an accumulating per-layer divergence measured **before** that commit; this says whether it
survived it.

**It did not. The worst relative delta across all 24 layers is 0.00064%, against 31.95% before.**

## Provenance

| | |
|---|---|
| artifact | `maple-preview-TQ1_0-head-Q4_K.gguf`, deepgrove/maple-preview-GGUF rev `f5466f918e0c50cdb9d4d47a6f35813509a42a30` |
| sha256 | `54016e4d543bd688829e67103fc85b8396db94b7f8eb3f81fa95884e44393872`, verified on this disk before the run |
| oracle | ggml-org/llama.cpp `3f03b17724ad8b69040825f45c43c59b092a0244` — the same revision the 2026-10-24 record pins, and the local clone's HEAD |
| oracle arm | **re-run here**, not copied: `llama-eval-callback -m <artifact> -p "The" -n 1 -t 12` |
| our branch | `fix/q8-f32-activation-kernel`, on `main` at `b201d7ee` |
| our route | Rust batched-matmul shim, `-Dmodels.purejava.halfPrecisionKvCache=true` |
| host | i7-9750H, AVX2, **no AVX-512**; macOS. **A different box from the 2026-10-24 record**, which used an i9-14900HX |

The host differs from the earlier record, which does not affect this comparison: both arms ran
here, and per-layer residual sums are arithmetic rather than timing. It does mean **no number in
this directory is comparable to a time in the earlier one.**

## The oracle file was re-derived, not trusted

`../2026-10-24-maple-oracle-ladder/SHA256SUMS` verifies clean, and the fresh `llama-eval-callback`
run reproduces that committed oracle ladder on every one of the 24 layers. So the oracle is stable
across the two runs and the change measured below is entirely on our side. `embd` is `-0.383070`
in both, which is also the value an independent Python read of row 785 gave in the earlier record.

## Result

`python3 compare_ladder.py oracle-rerun-token785.txt ours-routed-token785.txt \
  ../2026-10-24-maple-oracle-ladder/ours-ladder-token785.txt`

| | worst relative delta, 24 layers |
|---|---|
| before the computation fix | **31.95%** (layer 22) |
| after it | **0.00064%** (layer 22) |

Layer 0, where #299's table starts: `-1.676972` oracle against `-1.676975` ours, having been
`-1.683112`. The divergence was present at one token and accumulated with depth; it now does not.

## The harness had to be replaced, and that is the part worth keeping

`MapleLadder.java` in the earlier directory **cannot measure the route this question is about.**
Every observer constructor in `LlamaForwardPass` hard-codes `GgufBatchedMatrixKernel.none()`
(the two observer-taking constructors are `LlamaForwardPass.java:386` and `:399`, hard-coding it at
`:394` and `:411`), so passing a kernel is not possible through the constructor that harness calls —
it measured the exact F32 path whatever was on the classpath. 168 of maple's
tensors are TQ1_0, and the whole point of the fix is that ggml quantizes their activations to Q8_K
while the exact path does not. The old ladder was therefore measuring the arm that the fix does not
touch.

`MapleLadderRouted.java` here calls the kernel-accepting constructor (`:416`), which already existed
and is package-private, so **no runtime change was needed to make this measurable** — only a harness that
reaches it.

It also **prints the route instead of assuming it**, because passing a kernel and the kernel being
eligible are different things and `isEligible` can decline silently:

```
kernel=rust-ffm-quantized-v14  tq1_0-supported=true  tq1_0-eligible-1x2048x2048=true  halfPrecisionKvCache=true
```

A ladder without that line is a number whose provenance cannot be checked.

## The 355-token continuation still differs, and still must not be gated on

| configuration | continuation |
|---|---|
| ours, post-fix (this run) | `</think>\n\n**Answer: 8**  \n\n**Explanation:**  \n\nThe question is` |
| ours, F16 KV, 2026-10-24 | `</think>\n\n**Answer: 8**  \n\n**Explanation:**` |
| ours, F32 KV, 2026-10-24 | `Item 19 (Item 19 (Item ` |
| llama.cpp | `Item 19: the gravel beside the harbor was counted` |

So the fix did not change this prompt's continuation, and it still is not llama.cpp's. #299 already
settled how to read that: **do not gate on it.** Two configurations of our own code produce wildly
different continuations here, a 1.6-bit 20B MoE sits near behavioural forks, and greedy-text
equality after hundreds of tokens is not a stable property. The stable criteria are per-layer
agreement and expert-selection agreement, and the first of those is now effectively exact.

## What this does not establish

- **No timing claim, and the prefill number in `ours-355-tokens-postfix.txt` must not be quoted.**
  It reads `prefill = 1110.58 s`, measured while two other jobs and an 8 GB download were running —
  the load average was over 300 on a 12-thread box. It is in the file because the file is the raw
  arm and editing it would be worse; it is not a measurement of anything.
- **No expert-selection re-check.** The earlier record's flip count went to 0 of 24 with F16 KV,
  and that metric was not re-derived here. Per-layer agreement at this precision implies it, but
  implication is not measurement.
- **Nothing about qualification.** maple has no catalog entry, no comparator arm has run, and
  `../2026-10-24-maple-oracle-ladder/README.md` records that Ollama cannot parse TQ1_0 at all. This
  directory answers the parity question in #299 and no other.
- **One token.** The ladder drives position 0, where attention is self-only and RoPE is
  near-identity. That isolates the arithmetic, which is what it is for, and it is not a claim about
  long contexts.
