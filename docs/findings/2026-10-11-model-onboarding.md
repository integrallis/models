# EmbeddingGemma 2 and Bonsai 2 onboarding protocol

Frozen before Java/oracle comparisons on 2026-10-11. Candidate intake:
ModelJars #214 (EmbeddingGemma 2) and #215 (Bonsai 2).

## Artifact and scope

- EmbeddingGemma 2 text GGUF Q8_0: ggml-org revision
  `bfcd298762cc34d0357ece5ebdd31791a3a374d8`, SHA-256
  `2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135`.
  This file has no vision/audio tower. Text qualification cannot qualify another modality.
- Bonsai 2 27B PTQ1_0: prism-ml revision
  `b072e1d3b35a0a630cece372c2127528e0994386`, SHA-256
  `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`.
  5,946,648,928 bytes describes the file, not total runtime memory.

## Embedding gate

Use the existing `oracle-equivalence-v1` probe set unchanged. Independent oracle:
llama.cpp `5c8854fd88319125125dc4b836c98b1b215b0f23`, CPU, mean pooling,
non-causal attention, L2 normalization, context 512. Required worst-probe cosine
is at least 0.999 and absolute norm error at most 1e-3, matching the existing gate.
Record token IDs, raw vectors, artifact/probe hashes and environment. No average
can hide a failing probe. Confirm state reset by repeating a probe after a different
input; dimension must be 768 and every output finite. Test the symmetric attention
window boundary and future-token access separately. A failed comparison means fix
the implementation or retain a failed candidate; never change the reference to fit Java.

The model card, Safetensors config and GGUF differ in how the window/context is
expressed. The execution contract for this candidate is its pinned GGUF and the
independent implementation over those exact bytes. Do not borrow qualification
for a different Safetensors checkpoint.

## Generation gate

Bonsai's header declares Prism Hadamard preprocessing in addition to PTQ1_0
packing. Pin the publisher's reference revision and verify that transform before
running the existing controlled generation workload. Compare both arms on one
host, same prompt/template, context, thread count and generation controls. Use
existing contribution and relative-performance gates unchanged. Retain the
separate default-configuration smoke. Do not interpret pipeline fallback success
as generated-answer correctness. Do not claim total memory or vendor throughput.

## Publication

First fail the public Java path with real weights, then implement and rerun.
No external runtime is linked or shipped. New runtime code requires the normal
full build/compliance and integration gates, then a Models release. Qualification
must name that released version. ModelJars marker publication follows its documented
verify → Central staging → finalization → artifact download verification → Pages
sequence. Unsupported modalities and unsuccessful gates remain explicit.

## Bonsai reference pin (before any Java comparison)

PrismML-Eng/llama.cpp `e8fd209c975c80aa7fb36a8423571ff3b1b7e955`,
CPU, context 512, 12 threads, temperature 0, seed 1. First continuation smoke:
raw completion prompt `The quick brown fox`, eight generated tokens, no chat wrapper.
The signed transform applies before listed projections and in inverse order after
token lookup. The reference reorders tiled recurrent value heads before `ssm_out`;
Java's recurrence also emits tiled heads (`head % keyHeadCount`), so the same
tiled-to-grouped permutation is required before signs and Hadamard at `ssm_out`.

## Bonsai default workload (declared before execution)

Run all nine `general` cases once, no warmups, `chatml-no-think`, context 2048,
64 output tokens, retrieval top-k 1, 12 threads, default greedy generation controls.
Use the public pure-Java backend with no Models tuning properties. This is the existing
versioned controlled workload, not a new authored corpus. Record model contribution separately
from pipeline correctness. Until the authoritative comparator supports PTQ1_0, this run is
compatibility/quality evidence only and does not qualify publication or throughput.

## Default workload outcome

The [complete Bonsai smoke report](../../benchmark-results/2026-10-11-model-onboarding/bonsai2-default.json)
completed all nine cases: four model answers, four extractive fallbacks and one
retrieval abstention. All passed the existing checks; there were no failures or
truncations. Model contribution was 44.44%, with 100% correctness among accepted
model answers. This source-build smoke meets the absolute contribution floors,
but does not replace the missing authoritative paired evaluation or released-runtime
workload evidence. See the [provenance and limits](../../benchmark-results/2026-10-11-model-onboarding/README.md#bonsai-default-rag-smoke-result).
