# LFM2.5-Embedding-350M: an `lfm2` file that is not a decoder

Date: 2026-10-08. Box: i9-14900HX, AVX2, no AVX-512.
Oracle: llama.cpp `llama-embedding`, version recorded per reference file.
Gate: `EmbeddingEquivalenceCli`, 8 pinned probes, floor = worst-probe cosine >= 0.999 and unit
length within 1e-3.

Every number below is re-derived by `./compare.py` from the raw files in `raw/`.

## What was wrong

All seven published artifacts of LFM2.5-Embedding-350M were unservable, in two different ways:

- five quantized artifacts were refused at load, because the `lfm2` decoder exposed no hidden
  state and an embedding is the hidden state;
- the F16 and BF16 artifacts were refused earlier still, at `Lfm2Weights.MATRIX_TYPES`.

Exposing the hidden state and accepting the two half formats got all seven to load and produced
**worst-probe cosine ~0.03** — on BF16 too, so not a quantization effect. The cause is in the
artifact's own metadata:

    lfm2.attention.causal = False
    lfm2.pooling_type     = 2        (cls)

llama.cpp reads `LLM_KV_ATTENTION_CAUSAL` generically for every architecture
(`src/llama-model.cpp:1069`, default `true` at `src/llama-hparams.h:182`), so the oracle ran this
file bidirectionally without any lfm2-specific handling. Our side read that key only in
`BertConfig`. Two operators change when it is false, both read from
`src/models/lfm2.cpp:build_shortconv_block`:

1. attention spans the whole sequence rather than the past;
2. the short convolution becomes a **centred** window — `pad = (l_cache - 1) / 2` columns of
   symmetric zero padding — instead of a shift register over the past.

`src/models/lfm2.cpp:276` also confirms the head contract this work relied on: `res->t_embd` is the
activation after the final norm and before the vocabulary projection.

## Result

`Lfm2SequenceEncoder` runs the file layer-major over the whole sequence; a non-causal `lfm2` now
loads through `EncoderDecoderAdapter` and pools from its own metadata.

| artifact | verdict | worst-probe cosine |
| --- | --- | --- |
| F16 | REPRODUCED | 0.9999994 |
| BF16 | REPRODUCED | 0.9999915 |
| Q8_0 | REPRODUCED | 0.9997380 |
| Q4_0 | REPRODUCED | 0.9997358 |
| Q4_K_M | NOT_REPRODUCED | 0.9986829 |
| Q6_K | NOT_REPRODUCED | 0.9985676 |
| Q5_K_M | NOT_REPRODUCED | 0.9983368 |

F16 at 0.9999994 is what says the architecture is now right: it exercises the whole pass with no
quantization anywhere.

## The three that still fail, and two hypotheses that did not survive

The three K-quant artifacts land at 0.9983-0.9987, below the floor. They stay NOT_REPRODUCED. The
floor is not being moved to fit them.

**Hypothesis 1 — our Q8_K activation quantization is the loss. Refuted.** Switching the encoder's
projections to `ggufExactBatchedMatmul` (F32 activations, no activation quantization at all) moved
nothing: Q4_K_M 0.9986829 -> 0.9987225, Q6_K 0.9985676 -> 0.9985100, one up and one down. The fast
batched path is therefore the shipped path, since the slower one buys no accuracy.

**Hypothesis 2 — the oracle is not stable at this precision. Refuted.** llama.cpp run twice on the
same artifact, once with its default batching and once with `-b 1 -ub 1` (which takes the per-token
vec_dot path instead of the repacked GEMM), agrees with itself at **1.0000000** on both Q6_K and
F16. The oracle is deterministic across its own kernel choice, so the disagreement is ours.

**What is measured, and left open.** llama.cpp's own Q6_K differs from llama.cpp's own F16 by
**0.9966762** on these probes. Our Q6_K agrees with llama.cpp's Q6_K to 0.9985676 — that is, our
residual disagreement with the oracle is smaller than what quantizing the model does to the
embedding in the first place, which is the scale one expects from two different arithmetic
orderings over the same quantized weights (we dequantize to F32 and dot; llama.cpp dots in Q8_K
integers). This says the remaining gap is small and plausible. It does **not** identify it, and it
is not a reason to pass the artifacts: unexplained is unexplained. The same signature appears on
six other K-quant artifacts from unrelated models (nomic Q4_0/Q4_K_S/Q4_K_M, all-MiniLM
Q5_K_S/Q5_K_M, qwen3-embedding-4b Q4_K_M), so a single K-quant cause is worth finding.

## Reproduce

    ./compare.py                      # every number above, from raw/
