# Q4_1: found by the fleet, fixed, and checked against the oracle

Date: 2026-10-09. Box: i7-9750H (MacBook Pro), AVX2.

## How it was found

Shard 600 was launched to close enterprise capability gaps with five models that needed only a
qualification run. Four of the five died at the first token:

    java.lang.UnsupportedOperationException: GGUF matmul not supported for: Q4_1

`qwen2.5-math-1.5b Q4_0`, `phi-4-mini Q4_0`, `fin-r1-7b Q4_0` and `huatuogpt-o1-7b Q4_0`. The
triage tool shows why it was invisible: Q4_1 is **three or four tensors per model, about 3% of
weights**, inside an otherwise Q4_0 or K-quant build. Four models across three capability gaps
(math, finance, medical) blocked by a handful of tensors.

Cost of finding it: one 16-vCPU box, six minutes. `raw/shard-600-progress.tsv` is the run.

## The fix

`Q4_1Dequantizer`, transcribed from `block_q4_1` in llama.cpp `ggml-common.h:205-212` and
`dequantize_row_q4_1`. Twenty bytes per thirty-two weights: `d` as f16, `m` as f16, sixteen nibble
bytes. The value is `q * d + m` with `q` unsigned in 0..15.

**The one thing that fails silently** is the centring bias: `Q4_0` subtracts eight from its quants,
and applying that here would shift every weight in the block by `-8 * d` while still producing
fluent text. A test pins it, along with the nibble split (element `j` low, element `j + 16` high)
and the unsigned range reaching 15.

Both dispatches now share one routine, `exactSingleTokenMatmul`, with Q5_1. A `q * d + m` format
can be folded as `d * sum(q*x) + m * sum(x)`, which is a different fold order from
`PinnedReduction.dot` -- so a decoder and an embedder on the same weights would otherwise disagree
in the last bits depending on which dispatch they took. The parity test asserts equality, not
closeness.

## Checked against llama.cpp, not just against itself

Same artifact (`qwen2.5-math-1.5b-instruct Q4_0`, sha
`b8978c7f9eb19b3c83c1a3d1c90044c7c3ce1efb920c60771e12ba56c710e93f`), same prompt, greedy:

| | continuation |
| --- | --- |
| ours (pure-java) | ` To find the product of 17 and 24, we can use the standard multiplication method. Here are the` |
| llama.cpp `llama-simple` | ` To find the product of 17 and 24, we can use the standard multiplication method. Here are the` |

**Token-for-token identical.** Before the fix the same invocation threw; `trial 1/1: ok
ttft=5617.7ms decode=3.84 tok/s` after, on this 2019 laptop.

1026 tests pass across backend-java and backend-native, clean `--rerun-tasks`.

## What this does NOT yet license

0.3.55 is released and **does not contain Q4_1**. Qualifications produced with a Q4_1 build
therefore cannot be landed under `backendVersion: models-0.3.55` -- that is the same false claim
that held the lfm2 and Q5_1 artifacts back until 0.3.55 shipped. The re-run of shard 600 is a
**validation** run; its verdicts land only after a release that carries this commit.

## The other verdict from shard 600

`eurollm-1.7b-instruct Q4_K_S` ran both arms and returned **FAILED_MODEL_CONTRIBUTION_GATE** -- a
real verdict, not an error. It is a published null: the pipeline answered, the model did not
contribute enough of it. Translation coverage stays thin until a model passes.
