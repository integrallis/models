# Gemma 4 26B-A4B: TurboFieldfare transfer and Models qualification

## Outcome

Models now supports the pinned Gemma 4 26B-A4B Instruct Q4_K_M graph through
its Java transformer and Rust/FFM projection backend. Exact generation and all
three application adapters pass. The production qualification does not: all
27 guarded-RAG requests are correct, but 3,793.7 ms p95 TTFT exceeds the
2,000 ms usable ceiling. This is a supported, integration-tested artifact, not
a qualified performance profile.

The implementation is `5ec2c5d446e60baa955ef58bd7e4e3dd52281747`. The retained
machine-readable evidence is in
[`benchmark-results/certified-20260802/rag/gemma4-26b-a4b-q4_k_m/`](benchmark-results/certified-20260802/rag/gemma4-26b-a4b-q4_k_m/).

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
| Bound expert residency | Zero-copy mapped GGUF expert slices; no copied expert-slot cache or custom repack | Stable mapped-slice tests and 15.58 GB peak RSS on a 32 GB host |
| Batch tokens that route to the same expert | 32-token prefill, grouped route construction, and ragged independent native projection dispatch | `gemma4-batched-prefill=ENABLED`, batch size 32 |
| Expose independent projection work to persistent workers | ABI 4 dispatches different projection shapes in one worker generation across eight workers | Native boundary tests and `rust-ffm-v12` diagnostics |
| Pay compilation cost during load | A resettable warmup prefill runs while loading and clears sequence state before the first request | Load-warmup unit test and `load-warmup=ENABLED` |
| Validate each optimization against its numerical claim | Lossless graph changes retain exact outputs; SIMD reduction reordering uses narrow, explicit tolerances | Full Java check, 17 Rust tests, Clippy, native boundary tests, exact integration output |

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
| Correct retained model answers | 21/21 | pass |
| Retrieval recall / MRR | 1.0 / 1.0 | pass |
| p95 retrieval | 1.91 ms | pass |
| p95 TTFT | 3,793.7 ms | **fail**; usable ceiling 2,000 ms |
| p95 TPOT | 104.62 ms | pass |
| p95 end to end | 6,497.5 ms | pass |
| Median prefill / decode | 25.00 / 10.86 tok/s | diagnostic |

Plain Java runtime, LangChain4j, and Spring AI each generated the exact expected
adapter answer with one shared backend. Separate nine-case runs were 9/9
correct through every framework. These checks establish API integration; their
roughly 5.9-6.2 second p95 TTFT values do not establish production readiness.

## Publication decision and next gap

The decoder, prompt template, artifact fixture, and adapters can be released as
supported functionality. ModelJars should record Rust/FFM and the native local
engines as supported backends, and should retain a
`FAILED_ABSOLUTE_GATE`/`OFFLINE` qualification record whose measured failure is
latency. It should not publish an automatic performance profile for this
artifact.

Clearing the present gate requires about a 1.9x reduction in p95 TTFT without
changing the prompt, model, generation controls, or quality policy. The next
credible work is shape-specific projection and attention/FFN fusion, including
native attention where measurement justifies it. Relaxing the qualification
threshold would conceal the gap rather than close it.
