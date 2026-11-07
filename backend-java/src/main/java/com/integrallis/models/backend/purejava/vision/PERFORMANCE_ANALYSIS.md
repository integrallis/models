# ClipVisionEncoder Performance Analysis

## Test Hardware
- CPU: Intel i9-14900HX (24 cores, 8 performance + 16 efficiency)
- Threads: 8 (performance physical cores)
- SIMD: AVX2, 256-bit vectors
- OS: Linux x86_64
- RAM: 94 GB

## Model Configuration
- Image: 224×224 RGB
- Patch size: 14×14 → 256 patches + 1 CLS = 257 tokens
- Architecture: CLIP ViT-S (student)
- Hidden size: 384
- Layers: 6
- Heads: 6
- Intermediate: 1536
- Parameters: ~22M

## Measured Results

### Our Implementation

| Path | Avg Latency | Min | Max | Std | Throughput |
|------|-------------|-----|-----|-----|-----------|
| Pure-Java F32 | **257 ms** | 252 ms | 274 ms | 6.1 ms | 3.9 img/s |
| Rust Q8_0 kernel | **150 ms** | 136 ms | 164 ms | 8.0 ms | 6.7 img/s |
| **Speedup** | **1.7×** | | | | |

The Rust shim accelerates the 24 projection matmuls (4 per layer × 6 layers)
using quantized Q8_0 weights and AVX2 integer SIMD. Java handles patch embed,
attention, GELU, layernorm, and residuals.

### llama.cpp / Ollama (llava-phi3)

| Metric | Value |
|--------|-------|
| Model | llava-phi3 (CLIP ViT-L/14 + Phi-3 text, ~2.9B total) |
| Vision encoder | CLIP ViT-L: 24 layers, 1024 hidden, ~307M params |
| Cold vision encode | **~1.8–2.2 s** (estimated from total − prompt_eval) |
| Text prompt eval | **~9.6–12 s** (588 tokens through Phi-3 2.7B) |
| Total cold multimodal | **~11–14 s** |
| Cached re-encode | **~80 ms** (embeddings cached by Ollama) |

### Per-FLOP Normalization

ViT-L has ~13.6× more FLOPs than ViT-S. Normalizing llama.cpp's cold vision
encode to ViT-S scale:

- llama.cpp equivalent ViT-S: ~2000 ms / 13.6 ≈ **~147 ms**
- Our Rust Q8_0: **150 ms**

**Normalized comparison: approximately on par with llama.cpp per-FLOP.**

The 1.7× gap from pure-Java to Rust-Q8_0 is real and meaningful. The remaining
~2× gap to 80 ms (Ollama cached) comes from:
1. Attention scores: still scalar per-position/head Java loops
2. GELU: scalar `erf()` approximation
3. LayerNorm: scalar mean/variance computation
4. No caching (each call is a fresh encode)

## Bottleneck Breakdown (Rust Q8_0, ~150 ms)

Estimated component times based on FLOP counts and measured scaling:

| Component | Est. Time | % of Total |
|-----------|-----------|-----------|
| Patch embed | ~30 ms | 20% |
| Position embed | ~5 ms | 3% |
| QKV projections (Rust Q8_0) | ~12 ms | 8% |
| Attention scores (Java) | ~30 ms | 20% |
| Attention output proj (Rust) | ~6 ms | 4% |
| FFN up (Rust Q8_0) | ~18 ms | 12% |
| GELU (Java scalar) | ~15 ms | 10% |
| FFN down (Rust Q8_0) | ~9 ms | 6% |
| LayerNorms (Java) | ~20 ms | 13% |
| Residuals/overhead | ~5 ms | 3% |
| **Total** | **~150 ms** | **100%** |

## Optimization Targets (to reach <100 ms)

1. **Batched attention Q·Kᵀ matmul** (~30 ms → ~10 ms, −20 ms)
   - Current: per-position per-head scalar loops
   - Target: single [tokens, tokens] = Q × Kᵀ batched matmul
   - Use Rust `groupedAttention` kernel or vectorize in Java

2. **Vectorized GELU** (~15 ms → ~3 ms, −12 ms)
   - Current: scalar `erf()` with polynomial approximation
   - Target: Panama Vector API `FloatVector` fast approximate GELU
   - Or add Rust `swiGluHandle` / activation kernel shim

3. **Vectorized LayerNorm** (~20 ms → ~8 ms, −12 ms)
   - Current: scalar mean/variance per row
   - Target: vectorized row-wise reduction with Panama

Combined projected: 150 ms → **~106 ms** (1.4× improvement).

## Pluggable Rust Architecture

```
VisionEncoder
├── ImagePreprocessor (Java, stays Java — simple pixel ops)
├── VisionKernel (SPI — pluggable)
│   ├── PureJavaVisionKernel (default)
│   │   └── Patch embed + all blocks in Java
│   └── RustVisionKernel (optional, injected)
│       ├── Patch embed (Java → future: Rust)
│       ├── Projections (✓ Rust Q8_0, done)
│       ├── Attention (future: Rust groupedAttention)
│       ├── GELU (future: Rust activation kernel)
│       └── LayerNorm (future: Rust or vectorized Java)
└── Output embeddings (float[])
```

As JVM Panama Vector API matures:
- Vectorized GELU and LayerNorm → move from Rust to Java
- Batched attention matmul → move from Rust to Java
- Keep Rust only for quantized projections (Q8_0,Q4_K,etc.)
- Eventually Rust can be removed entirely when Java GEMM catches up

## Comparison Summary

| System | Model Size | Cold Encode | Cached | Notes |
|--------|-----------|-------------|--------|-------|
| Our Java | 22M (ViT-S) | 257 ms | 257 ms | No cache; full encode every call |
| Our Rust Q8_0 | 22M (ViT-S) | 150 ms | 150 ms | No cache; projections accelerated |
| Ollama llava-phi3 | 307M vision (ViT-L) | ~11–14 s total | ~80 ms | Embeddings cached after first encode |
| llama.cpp granite | ~100M vision | ~14 s total | N/A | Document tiling; not comparable |

**Per-FLOP normalized**: Our Rust Q8_0 (~150 ms) ≈ llama.cpp equivalent (~147 ms).
