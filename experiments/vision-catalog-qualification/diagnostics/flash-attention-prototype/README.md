# External attention arithmetic prototype

This is an isolated Linux x86 AVX2/FMA/F16C experiment, outside the shipping Cargo
crate and Java classpath. It is not a qualified runtime or a performance result.
The process owns its scratch arrays and thread pool for its entire lifetime. The
controlled diagnostic caller supplies complete, nonaliasing arrays, width 128,
8 threads, batch-major queries and head-major F32 retained K/V. Do not expose this
experimental ABI to arbitrary callers or unload it with workers running.

The arithmetic is translated from ggml at
`2cbca389f515df455412dafa07e2b1359cc72a6d`, specifically CPU `ops.cpp`,
`vec.cpp`, `vec.h`, `simd-mappings.h` and `simd-gemm.h`. The upstream MIT terms
and copyright holders are retained in `LICENSE-GGML`.

The reference uses F16 Q for fewer than 64 queries, F16 K/V, an online half
accumulator, and split-KV decode with padded positions. Larger batches use F32
queries, tiled online softmax, and ordered F32 GEMM. Reduction order, fused
operations, and the wider tile-sum addition are observable: seven seeded operator
cases on the 2026-11-08 owned worker match the independent ggml operator bit for
bit after those details are preserved. Earlier prototypes did not, and their
results remain in the raw archive. This is operator evidence, not full-model
accuracy or performance qualification.

Before promotion: replace the experimental ABI with length-checked owned runtime
buffers; preserve platform support and pool teardown; avoid repeated conversion
of the entire prefix; add retained independent fixtures; verify all frozen model
cases and paired timing gates. No reference ggml library is linked into this Rust
prototype. The reference library is loaded only by the separate differential test.
