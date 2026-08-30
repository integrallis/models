# FreeToken compatibility and optimization audit: 2026-08-29

This audit identifies model families, checkpoint formats, and implementation techniques worth
bringing into Models. It is a source-level comparison, not a claim that Models can execute every
model listed here.

## Revisions and rules

- FreeToken: `bd372b630a028e3faa51f4ab0ef6a98c2f2de501`
- Models audit range: `292e461e4fe90502db1072138e3f82238dab0e12` through
  `5b283c75935c8579639db49b97492a44c0992689`
- llama.cpp oracle: `a58222229`
- FreeToken license: Apache-2.0
- Production inference remains in-process and Java-first. External engines are reference oracles
  and benchmark peers only. A native shim needs a measured Java limitation and must remain
  replaceable.

## Architecture comparison

| FreeToken family | Models status | Practical next action |
| --- | --- | --- |
| Llama, Qwen 2, Qwen 3, Mistral | GGUF execution already implemented in Java | Retain as regression families; borrow measured kernel ideas only |
| Gemma 4 | GGUF hybrid attention and streamed experts already implemented in Java | Compare expert-cache scheduling after JVM accelerator work lands |
| Qwen 3.5 dense | Added in Java during this audit | Qualify 0.8B and 4B first; screen 2B and 9B with the same oracle gates |
| Qwen 3 MoE | Not implemented | Reuse the existing Gemma 4 expert loader/cache, but add Qwen routing and graph semantics test-first |
| Qwen 3.5 MoE | Dense hybrid graph exists; routed MLP does not | Best MoE follow-on after dense Qwen 3.5 qualification |
| GPT-OSS | Not implemented | First validate MXFP4 decoding and attention-sink primitives independently; 20B is the smallest end-to-end candidate |
| Muse Glimmer | Not implemented | Defer until native NVFP4 linear execution is measured; smallest cited model is 30B |
| MiniMax M2/M3, GLM MoE/DSA, DeepSeek V4 | Not implemented | Defer: very large MoE and sparse-attention systems are poor first catalog artifacts |

The registry source is `python/freetoken/models/register.py`. Model-specific graph and format
semantics come from the adjacent package for each family; the table does not infer support from a
class name alone.

## Qwen 3.5 result

The dense Qwen 3.5 graph was the highest-value small-model result of this audit:

- A pinned 0.8B Q4_K_M GGUF verifies hybrid full-attention/Gated DeltaNet layout, public backend
  loading, sessions, rewind, and llama.cpp greedy-token parity.
- Official 4B and 9B configurations use 16 key heads and 32 value heads in Gated DeltaNet. The
  original equal-head implementation would silently compute the wrong recurrence.
- GGUF conversion tiles the value-head order. Java now maps value head `h` to key head
  `h % keyHeadCount`; a focused synthetic test distinguishes this from the Hugging Face
  safetensors grouping.
- A pinned 4B Q4_K_M file verifies token `2614` after `The` and token `33075` after
  `The quick brown fox`, matching llama.cpp `a58222229`. The latter also exercises non-zero text
  positions through Qwen 3.5's text-equivalent MRoPE path.

Pinned fixtures:

| Model | Revision | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `unsloth/Qwen3.5-0.8B-GGUF`, `Qwen3.5-0.8B-Q4_K_M.gguf` | `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0` | 532,517,120 | `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517` |
| `unsloth/Qwen3.5-4B-GGUF`, `Qwen3.5-4B-Q4_K_M.gguf` | `e87f176479d0855a907a41277aca2f8ee7a09523` | 2,740,937,888 | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |

## Techniques worth carrying forward

### Implement next

1. **Batched Qwen 3.5 prefill.** FreeToken projects a complete prefill batch and runs chunked GDN;
   Models currently advances the complete graph one token at a time. Models already has batched
   GGUF matrix kernels, so this can remain Java-native. Acceptance requires identical logits,
   convolution state, recurrent state, and continuation tokens before measuring throughput.
2. **State snapshots.** FreeToken snapshots recurrent and convolution state at reusable prefix
   boundaries. Models currently reconstructs Qwen 3.5 rewind state by replaying retained tokens.
   A bounded Java snapshot should first demonstrate lower rewind latency without changing session
   ownership or results.
3. **Qwen MoE on the existing expert cache.** Qwen 3/3.5 routing should be a new graph path over the
   existing Java mapped-expert machinery, not an imported serving runtime.

### Experiment before adoption

- **Fused GDN projection/norm operations.** FreeToken fuses QKV/Z/B/A projection and gated RMSNorm
  for GPU launch efficiency. Java already shares activation quantization across grouped
  projections; JFR must show the remaining elementwise work matters before adding special paths.
- **MXFP4 and NVFP4 safetensors execution.** Models parses safetensors bundles and recognizes NVFP4
  layout, but that is not end-to-end model execution. Start with independently checked unpack,
  scaling, and dot-product primitives before adding GPT-OSS or Muse Glimmer graphs.
- **Expert residency scheduling.** FreeToken's pinned host banks and asynchronous GPU transfers are
  useful design references once Models has a qualified JVM GPU backend. They do not justify adding
  FreeToken, PyTorch, CUDA Python, or another external inference dependency.

### Rejected experiment

A Vectors Q4_K paired-nibble scratch-write change was tested with correctness checks and stable
JMH measurement. It regressed from roughly `0.403 ms/op` to `0.452 ± 0.051 ms/op`, so the change was
fully reverted and Vectors remains unchanged.

### Retained workspace experiment

The Qwen 3.5 GDN path allocated query, key, value, gating, normalization, and recurrence arrays for
every linear-attention layer and token. A test-first caller-owned recurrence workspace and one
session-owned GDN workspace removed those temporary model arrays without sharing mutable state
between sessions.

The repository prefill profiler ran the same 10-token prompt with one warmup on the same host:

| Variant | Prompt throughput | Logit checksum | GC | JFR allocation samples |
| --- | ---: | ---: | ---: | ---: |
| Baseline at `9197f88` | 9.39 tok/s | -438243.971 | 0 | 8 |
| Session-owned GDN workspace | 9.38 tok/s | -438243.971 | 0 | 3 |

The sampled `float[]` allocations attributed to `Qwen35ForwardPass.gatedDeltaNet` disappeared from
the second JFR. Throughput is essentially unchanged, so this is retained as allocation hygiene, not
claimed as a speedup. Both 0.8B and grouped-head 4B reference gates pass after the change. Raw JFR
files are deliberately not retained because the standard recording includes initial process
environment metadata; the table contains the non-sensitive aggregate evidence.

## Format policy

- GGUF remains the first executable format for dense Qwen 3.5 because it is compact and already
  supported by the Java quantized kernels.
- Standard safetensors is the next execution format to extend. MXFP4, NVFP4, FP8, and BF16 are
  tensor encodings inside that standard container, not reasons to invent another container.
- FreeToken's FTW is a serving-oriented derived layout. It may inform memory mapping and expert
  bank design, but it should not become a public Models format unless independent ecosystem value
  is demonstrated.

## Release gate

No architecture reaches ModelJars from this audit merely because its parser accepts the file.
Promotion requires a public Models backend, pinned artifact identity, deterministic reference
equivalence, generation through the public runtime, documented limitations, and hardware evidence
appropriate to the claimed backend.
