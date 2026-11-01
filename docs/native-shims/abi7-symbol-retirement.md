# Native ABI 7 symbol retirement

The unpublished native ABI 7 removes eight redundant format-specific C entry points. Q4_0 and Q8_0 computation, capability bits, generic dispatch and worker support remain available:

- `jmodels_q4_0_f32_batched_matmul`
- `jmodels_q4_0_f32_batched_matmul_with_context`
- `jmodels_q8_0_f32_batched_matmul`
- `jmodels_q8_0_f32_batched_matmul_with_context`
- `jmodels_q4_0_f32_grouped_batched_matmul`
- `jmodels_q4_0_f32_grouped_batched_matmul_with_context`
- `jmodels_q8_0_f32_grouped_batched_matmul`
- `jmodels_q8_0_f32_grouped_batched_matmul_with_context`

These symbols appeared in published native libraries, but the supported Java bindings have always used the generic format-dispatch entries. No standalone C API compatibility contract is documented. Any direct native consumer must migrate to `jmodels_quantized_f32_batched_matmul_with_context`, `jmodels_quantized_f32_grouped_batched_matmul_with_context`, or `jmodels_quantized_f32_independent_batched_matmul_with_context` using the appropriate format identifier and live worker context.

This removal belongs to the already-breaking ABI 7 transition. Published ABI 6 Java loaders require exactly ABI 6 and reject ABI 7 independently of this change. No ABI 6 artifact is changed, and consumers must update matching Java/native artifacts together.

Removing the redundant callers changes native compiler inlining even though the generic arithmetic remains unchanged. Correctness and resource qualification must therefore use the rebuilt final native artifact. The symbol inventory records 14 CPU exports after this removal; operation proof status is unchanged.
