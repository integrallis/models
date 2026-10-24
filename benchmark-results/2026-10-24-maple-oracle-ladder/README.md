# maple against llama.cpp, layer by layer — 2026-10-24

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

## What remains — issue #299

ggml computes TQ1_0 products against **Q8_K-quantized activations**
(`ggml/src/ggml-cpu/ggml-cpu.c:402-404`: `vec_dot_type = GGML_TYPE_Q8_K`,
`ggml_vec_dot_tq1_0_q8_K`). We have no such kernel: #297 made TQ1_0 multipliable by routing it to
`TensorOps.ggufExactBatchedMatmul`, which by its own javadoc keeps activations in F32. 168 of
maple's tensors are TQ1_0, including every projection and all three expert matrices.

Stated carefully: the two paths **differ**, from the first layer. Exact F32 is plausibly closer to
the unquantized product than Q8_K activations are, so this is not a claim that our arithmetic is
worse. It has **not** been established that this difference alone accounts for the collapse in the
table above — only that it exists, that it is present at one token, and that we cannot reproduce
the reference bit-for-bit without that kernel.

Prefill on the exact path: **1177.78 s** for 1222 tokens, single-threaded. Not comparable to
llama.cpp's figure and not quoted against it — a different kernel on a different thread count is
not a budget-matched comparison.

`PTQ1_0` (PrismML ternary, group 128, 28-byte blocks) already has a fast Rust Q8_0 kernel. ggml's
`TQ1_0` is a different format at group 256 and has none; the existing kernel cannot be pointed at
these weights.

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

## Two measurement errors made here, recorded so they are not repeated

**`-c 2048` was being passed to `llama-simple`, which does not accept it.** Its usage is
`-m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]`, so the flag was concatenated into the
prompt text and the first comparisons were not on identical inputs. Caught when a short prompt
came back `is is is is`. Re-run clean, the continuations above are unchanged, so the conclusions
stand — but they were not rigorous when first drawn.

**The first long-prompt run used `PureJavaBackend.load(Path)`,** which passes
`GgufBatchedMatrixKernel.none()` and `BatchedCausalAttentionKernel.none()` by construction, with
no `backend-native` on the classpath at all. Its prefill time was quoted before that was noticed.
A timing from a route the product does not use is not a measurement of the product.
