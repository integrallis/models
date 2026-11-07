# Multimodal Vision Support — Implementation Notes

## Status

This package provides a **pure-Java CLIP vision encoder** capable of loading from
mmproj GGUF files and running a complete ViT forward pass. It is designed to
support OCR/RAG pipelines where images need to be converted into embeddings that
can be consumed by text decoders.

## Components

### `ClipVisionConfig`

Reads vision encoder configuration from mmproj GGUF metadata. Supports keys:
- `clip.vision.embedding_length` — hidden dimension
- `clip.vision.block_count` — number of transformer layers
- `clip.vision.attention.head_count` — attention heads
- `clip.vision.patch_size` — conv patch size
- `clip.vision.image_size` — expected input resolution
- `clip.vision.num_position_embeddings` — learned position table size

### `ClipVisionWeights`

Loads and validates `v.*` tensors from the mmproj file:
- `v.patch_embd.weight` + `v.patch_embd.bias` — patch embedding
- `v.position_embd.weight` — learned position embeddings
- `v.post_ln.weight` + `v.post_ln.bias` — post-transformer norm
- `v.blk.N.*` — per-layer weights (ln, qkv, attn_out, ffn)

Follows the same `projection()` / `vector()` / `matrix()` pattern as `BertWeights`
for consistency with the rest of the codebase.

### `ClipVisionEncoder`

The forward pass reuses existing production kernels:
- `TensorOps.ggufExactBatchedMatmul` / `TensorOps.ggufBatchedMatmul` — linear projections
- `TensorOps.layerNorm` — layer normalization
- `TensorOps.softmax` — attention score normalization
- `TensorOps.geluErf` — FFN activation
- `VectorUtil.batchDotProductExact` — Q·K attention dot products
- `VectorUtil.addWeightedRowsInPlace` — weighted value summation

Supports all GGUF quantization types that `TensorOps` supports (F32, F16, BF16,
Q4_0, Q5_0, Q8_0, K-quants via dequantization).

## Integration with Text Decoder

The `Qwen35ForwardPass.forwardEmbeddings(Session, float[], int)` method accepts
pre-computed embeddings and runs the text decoder without calling `embedToken()`.
This is the mechanism by which vision features become the initial hidden state
for autoregressive generation.

## Known Limitations

1. **No M-RoPE support**: Position encoding is 1D. Models requiring multimodal RoPE
   (e.g. Qwen3-VL with `[yyyyxxxx]` section layout) need additional position tables.
2. **No spatial merger**: The 2×2 patch pooling used by Qwen3-VL is not implemented.
3. **No companion file loading**: `PureJavaBackend` loads a single GGUF. Multimodal
   models needing an mmproj alongside the text model require separate load logic.
4. **Pure-Java path only**: No Rust kernel acceleration for vision ops yet. The
   architecture supports `GgufBatchedMatrixKernel` injection but no vision-specific
   kernels are implemented.

## Next Steps

1. Add `MediaProjector`-style interface (inspired by qxotic) for position-aware
   chunking and M-RoPE support.
2. Implement Qwen3-VL dual patch embedding + 2×2 spatial merger (see qxotic
   `Qwen35Vision` for reference).
3. Add mmproj loading to `PureJavaBackend` or a standalone multimodal backend.
4. Add multimodal prompt tokenizer with media marker replacement.
5. Run end-to-end parity against llama.cpp `mtmd-cli` with a real model artifact.

## References

- llama.cpp mtmd: `tools/mtmd/` (llama.cpp multimodal support)
- qxotic Qwen35Vision: `jinfer/jinfer-qwen35/src/main/java/.../Qwen35Vision.java`
- HuggingFace Surya OCR GGUF: `datalab-to/surya-ocr-2.5-gguf`
