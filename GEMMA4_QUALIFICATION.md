# Gemma 4 26B-A4B: TurboFieldfare transfer and Models qualification

## Outcome

Models now supports the pinned Gemma 4 26B-A4B Instruct Q4_K_M graph through
its Java transformer and Rust/FFM projection backend. Production qualification
passes: all 27 guarded-RAG requests are correct and 1,861.7 ms p95 TTFT clears the 2,000 ms
usable ceiling. The pinned `.2` artifact is qualified at the `USABLE` tier.

The graph implementation is `37fc4a8b9d421505487b678c7ce841d1baa20eb4`; prompt-boundary
correction `ac80da9f89d908c1ac4b69cb95fa63c6b21cc890` is included in the
superseding machine-readable evidence under
[`benchmark-results/certified-20260825/rag/gemma4-26b-a4b-q4_k_m/`](benchmark-results/certified-20260825/rag/gemma4-26b-a4b-q4_k_m/).

## Source design

The transfer started from TurboFieldfare commit
[`76f6611467207a80c4a4d72bd11c2818f0703739`](https://github.com/drumih/turbo-fieldfare/commit/76f6611467207a80c4a4d72bd11c2818f0703739),
especially its immutable
[`docs/SYSTEM_DESIGN.md`](https://github.com/drumih/turbo-fieldfare/blob/76f6611467207a80c4a4d72bd11c2818f0703739/docs/SYSTEM_DESIGN.md)
and the numbered summaries covering model installation and expert I/O, MoE
decode and routing, expert caching, prefill, and validation. Those source
documents describe a Swift/Metal system optimized for Apple Silicon and a much
smaller resident-memory envelope. Models reused the engineering principles,
not its code or platform assumptions.

## What transferred

| TurboFieldfare principle | Models implementation | Qualification evidence |
| --- | --- | --- |
| Preserve exact Gemma 4 semantics before optimizing | Dedicated graph for hybrid local/full attention, shared plus routed MoE, asymmetric K/V details, learned scaling, and logit softcap | Scalar-reference graph tests and exact pinned-model generation |
| Bound expert residency | Zero-copy mapped GGUF expert slices; no copied expert-slot cache or custom repack | Stable mapped-slice tests and 15.72 GB peak RSS on a 32 GB host |
| Batch tokens that route to the same expert | 128-token prefill, grouped route construction, and ragged independent native projection dispatch | `gemma4-batched-prefill=ENABLED`, batch size 128 |
| Expose independent projection work to persistent workers | ABI 4 dispatches different projection shapes in one worker generation across eight workers | Native boundary tests and `rust-ffm-v12` diagnostics |
| Pay compilation cost during load | A resettable warmup prefill runs while loading and clears sequence state before the first request | Load-warmup unit test and `load-warmup=ENABLED` |
| Validate each optimization against its numerical claim | Lossless graph changes retain exact outputs; the bounded GELU table has an explicit error limit; SIMD reduction reordering uses narrow, explicit tolerances | Full Gradle check, 18 Rust tests, Clippy, native boundary tests, exact integration output |

The storage decision deliberately differs from TurboFieldfare. Its bounded
`pread` slot cache solves an 8 GB-class Apple memory problem around a repacked
model. The qualification host has 32 GB and the GGUF is already map-friendly,
so Models maps routed expert ranges directly. Copying those ranges into another
cache would add memory traffic without evidence that it improves this CPU
workload.

## What did not transfer

Trace-trained expert prediction, speculative prefetch, a larger resident
expert cache, and a custom model repack were not carried over. They add state,
artifact coupling, or I/O speculation without controlled evidence on the
target JVM/FFM/CPU platform. The implementation also does not claim Metal or
Apple GPU behavior: this qualification used Java 25, Panama FFM, CPU kernels,
eight AMD EPYC-Milan vCPUs, and 32 GB RAM.

## Results

The exact artifact is revision
`ae4d537a6345467d1c86bb5cc0d4505ff3ebe0f3`, 16,796,015,136 bytes, SHA-256
`88f4a13b0bb95f031a7fad973e10854122fb67ebc34d214d39a2f65053046abc`.
The standard run used one warmup and three iterations over nine general RAG
cases.

| Result | Measurement | Gate |
| --- | ---: | --- |
| Successful and correct | 27/27 | pass |
| Correct retained model answers | 18/18 | pass |
| Retrieval recall / MRR | 1.0 / 1.0 | pass |
| p95 retrieval | 1.89 ms | pass |
| p95 TTFT | 1,861.7 ms | pass; usable ceiling 2,000 ms |
| p95 TPOT | 89.02 ms | pass |
| p95 end to end | 4,318.2 ms | pass |
| Median prefill / decode | 45.10 / 12.91 tok/s | diagnostic |

The corrected candidate and both native controls tokenize the first `gemma4`
prompt as 191 tokens. This verifies that template markers are controls while
retrieved evidence and the question remain ordinary text.

Matched 27-attempt llama.cpp and Ollama controls were also 27/27 correct. Models
achieved 81.03% and 94.54% of their respective median decode throughput, while
its p95 end-to-end latency was 73.85% and 65.86% of theirs. Both comparisons
pass `production-rag-model-contribution-v5`, producing a `QUALIFIED` verdict.

## Performance work and publication decision

The qualifying implementation parallelizes safe Gemma prefill regions, uses
retained per-row scratch storage for F32 projections, vectorizes accumulation,
precomputes RoPE frequencies, and replaces hot GELU evaluation with a bounded
lookup table. Its Rust projection workers use stack-backed scratch storage for
the 128-row qualification batch, eliminating per-dispatch heap allocation.
Attention remains sequential after a sliding-window cache wraps, preserving
the cache dependency boundary.

The standard report is bound to the implementation commit, exact artifact,
generation controls, and qualification host. ModelJars can therefore publish
`org.modeljars.huggingface:ggml-org.gemma-4-26b-a4b-it-gguf.q4_k_m:4.0.0-q4_k_m.2`
with a `QUALIFIED`/`USABLE` record. The `.1` attempt remains historical evidence
and must not be relabeled.
