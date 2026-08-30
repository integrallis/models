# FreeToken compatibility and optimization audit: 2026-08-29

This audit identifies model families, checkpoint formats, and implementation techniques worth
bringing into Models. It is a source-level comparison, not a claim that Models can execute every
model listed here.

## Revisions and rules

- FreeToken: `bd372b630a028e3faa51f4ab0ef6a98c2f2de501`
- Models audit range: `292e461e4fe90502db1072138e3f82238dab0e12` through
  `b06bdf91f864fe90546c7d12d56becaf983055fa`
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
| GPT-OSS | Official 20B Safetensors/MXFP4 graph executes through the public pure-Java backend | Complete public generation, framework tool-call, quality, memory, and throughput gates before catalog promotion |
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
- Prefill now projects bounded prompt chunks through the existing Java GGUF batch kernels. Gated
  DeltaNet recurrence remains token ordered, uses Vector API row operations, and retains its
  memory, delta, normalization, output, and state workspaces in the owning session.

Pinned fixtures:

| Model | Revision | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `unsloth/Qwen3.5-0.8B-GGUF`, `Qwen3.5-0.8B-Q4_K_M.gguf` | `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0` | 532,517,120 | `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517` |
| `unsloth/Qwen3.5-4B-GGUF`, `Qwen3.5-4B-Q4_K_M.gguf` | `e87f176479d0855a907a41277aca2f8ee7a09523` | 2,740,937,888 | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |

## Techniques worth carrying forward

### Implemented and measured

The 0.8B Q4_K_M artifact ran a fixed nine-case general workload on the controlled eight-vCPU AMD
EPYC Milan host. Every retained variant answered 9/9 cases correctly and used the in-process
pure-Java backend. The measurements below are one-iteration qualification screens, not a claim of
production readiness:

| Variant | p95 TTFT | p50 decode | p95 end to end | Result |
| --- | ---: | ---: | ---: | --- |
| Sequential 0.3.16 baseline | 12,731.5 ms | 10.59 tok/s | 14,997.5 ms | Correct; `OFFLINE` |
| Batched projections | 11,718.8 ms | 9.45 tok/s | 14,790.7 ms | Correct; retained |
| Batched + SIMD recurrence | 9,779.3 ms | 10.50 tok/s | 12,533.1 ms | Correct; retained |
| Batched + SIMD recurrence + Q5_K two-query tail | 9,255.7 ms | 11.05 tok/s | 11,854.6 ms | Correct; retained |

The final variant lowers p95 TTFT by 27.3% from the sequential baseline, but it still exceeds the
production policy. Qwen3.5 therefore remains outside ModelJars qualification. The Q5_K tail is a
Vectors 0.1.15 kernel improvement: a separate three-fork JMH gate improved batch-2 execution from
`0.556 ± 0.008 ms/op` to `0.330 ± 0.003 ms/op` with exact outputs.

Two experiments were rejected rather than added to the runtime. Parallelizing recurrence heads
improved the controlled TTFT screen by only 3.3% and introduced coordination machinery; JFR showed
that recurrence was no longer a material hotspot. Selecting the two-query Q6_K kernel regressed
p95 TTFT to 10,737.6 ms, so the established one-query default remains.

### State snapshot experiment

The first snapshot gate is now implemented as an internal, same-session Java primitive. It copies
only the Gated DeltaNet convolution and recurrent state; the full-attention KV prefix stays in its
owning session. A restore is rejected if it targets another session or if the retained token prefix
has changed. Focused tests prove exact continuation, copy isolation, byte accounting, and those
ownership checks.

The pinned Qwen 3.5 0.8B Q4_K_M file then ran a 16-token diagnostic screen on an Intel
`MacBookPro16,1`, Temurin 25.0.3, with one warmup and five measurements:

| Operation | Median | State bytes | Continuation |
| --- | ---: | ---: | --- |
| Reconstruct the prefix by executing it again | 4,832.643 ms | n/a | exact |
| Restore the immutable Java state snapshot | 2.079 ms | 20,201,472 | exact |

That is a 2,324x difference for the isolated restore operation. It is strong evidence to retain the
primitive, not an end-to-end serving claim: this screen does not include radix lookup, eviction,
concurrent requests, or copying a separate KV prefix. Raw samples and exact environment details are
in `qwen35-linear-state-snapshot-intel-mac.json`.

### GPT-OSS MXFP4 routed-layer experiment

The first GPT-OSS slice remains Java-native and in-process. Models now maps the official
Safetensors expert tensor layout directly into Vectors 0.1.16 `Mxfp4Matrix` views, implements
FreeToken-compatible stable top-k routing and SwiGLU-OAI activation, and executes a complete
single-token expert layer without expanding MXFP4 weights. A pinned byte-range fixture from the
official `openai/gpt-oss-20b` checkpoint verifies nibble order and E8M0 scale decoding against real
artifact bytes.

The routed-layer implementation has two activation paths. The exact path keeps activations in
F32. The W4A8 path quantizes the shared hidden vector once, reuses it for all selected gate/up
projections, and quantizes each expert activation before its down projection. A focused synthetic
test checks the packed exact path against an independent scalar implementation and requires the
W4A8 output to exceed 0.9999 cosine similarity with less than 0.06 maximum absolute error.

A three-fork JMH diagnostic then used GPT-OSS 20B's official dimensions—2,880 hidden,
2,880 intermediate, and four selected experts—on the same Intel Mac environment:

| Activation path | Mean routed-layer time | 99.9% error | Relative result |
| --- | ---: | ---: | ---: |
| F32 exact | 159.305 ms/op | ±8.170 ms | baseline |
| Java W4A8 | 12.516 ms/op | ±1.919 ms | 12.73x faster |

The weights are deterministic synthetic MXFP4 data at production geometry, so this isolates the
expert compute path; it is not a full-model tokens-per-second result. The measurement is enough to
retain the Java implementation and continue to attention sinks and the decoder graph without a
Rust shim. Raw samples, the pinned revisions, and the exact command are in
`gpt-oss-20b-mxfp4-moe-intel-mac.json`.

### GPT-OSS official checkpoint compatibility

The complete pure-Java decoder now maps and executes the official, three-shard
`openai/gpt-oss-20b` checkpoint at immutable revision
`6cee5e81ee83917806bbde320786a8fb61efebee`. An independent Transformers 4.57.6 run
produced all 201,088 first-token logits for the exact 61-token Harmony prompt. Java selected the
same `<|channel|>` token and the complete oracle top-10 set on both EPYC Milan and Cascade Lake.
Full-vector cosine was 0.998585 and 0.997862 respectively; the hardware envelope is enforced by
the repeatable integration gate rather than assumed from one CPU.

The EPYC prefill took 80.918 seconds. The six-core Cascade Lake virtual CPU took 344.874 seconds,
despite being sold as part of a GPU instance; the Java path correctly did not use an unrelated
external GPU runtime. This passes checkpoint compatibility, not product performance. The next
experiment is a profile-led bounded/batched prefill change, followed by actual public generation
and Spring AI/LangChain4j tool execution. Complete hashes, oracle provenance, host measurements,
and thresholds are in `gpt-oss-20b-official-checkpoint-java.json`.

### Implement next

1. **Bounded cross-request prefix cache.** FreeToken stores convolution and recurrent state at
   aligned radix-tree prefix boundaries. A cache hit resumes only from the deepest node that still
   owns a live snapshot, then copies that immutable snapshot into a request-owned slot before
   execution; KV entries and state snapshots have separate eviction accounting. The same-session
   Java gate succeeded. The next experiment must add explicit byte limits, pair each linear-state
   snapshot with the matching full-attention KV prefix, and prove eviction, concurrency, and exact
   continuation before this becomes a runtime feature.
2. **Qwen MoE on the existing expert cache.** Qwen 3/3.5 routing should be a new graph path over the
   existing Java mapped-expert machinery, not an imported serving runtime.

### Experiment before adoption

- **Fused GDN projection/norm operations.** FreeToken fuses QKV/Z/B/A projection and gated RMSNorm
  for GPU launch efficiency. Java already shares activation quantization across grouped
  projections; JFR must show the remaining elementwise work matters before adding special paths.
- **NVFP4 Safetensors execution.** Models parses Safetensors bundles and recognizes NVFP4 layout,
  but that is not end-to-end model execution. Start with independently checked unpack, scaling,
  and dot-product primitives before adding Muse Glimmer graphs. MXFP4 and the GPT-OSS graph have
  passed official-checkpoint equivalence; GPT-OSS still needs generation, framework, quality,
  memory, and throughput qualification.
- **Expert residency scheduling.** FreeToken's pinned host banks and asynchronous GPU transfers are
  useful design references once Models has a qualified JVM GPU backend. They do not justify adding
  FreeToken, PyTorch, CUDA Python, or another external inference dependency.

### Earlier rejected experiment

A Vectors Q4_K paired-nibble scratch-write change was tested with correctness checks and stable
JMH measurement. It regressed from roughly `0.403 ms/op` to `0.452 ± 0.051 ms/op`, so the change was
fully reverted.

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
