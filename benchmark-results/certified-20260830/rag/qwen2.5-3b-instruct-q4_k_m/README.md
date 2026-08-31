# Qwen2.5 3B Instruct Q4_K_M qualification

The exact 2,104,932,768-byte GGUF artifact with SHA-256
`626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d`
clears `production-rag-model-contribution-v5` at the `USABLE` tier.

The final three-iteration run used Java 25.0.4 on a 16-core AMD EPYC 9R14
host. The selected Rust/FFM profile enabled native quantized decode with eight
workers plus batched Java attention scores and values. Models answered all 27
grounded requests correctly; 24 answers retained model-written text.

| Metric | Models | Ollama | Gate result |
| --- | ---: | ---: | ---: |
| p95 TTFT | 1,085.4 ms | 487.2 ms | `USABLE` |
| Median decode | 29.45 tok/s | 33.53 tok/s | 0.878x (minimum 0.8x) |
| p95 end to end | 2,046.7 ms | 1,514.5 ms | 1.351x (maximum 1.5x) |
| Correct grounded answers | 27/27 | 27/27 | pass |

`models-rust-ffm.json` and `ollama.json` retain every measured request. Their
SHA-256 values are bound by `qualification.json`.

The separate library-default smoke used the released Models `4934a6c1` code,
left native quantized decode disabled, and passed all 9 workload cases with no
generation failures. Its immutable report is
`default-correctness/models-rust-ffm.json` (SHA-256
`5b7227303ff0cf341f0906ada78bbe78e24d9248ec89c18394fca01ccc2cbf03`).
