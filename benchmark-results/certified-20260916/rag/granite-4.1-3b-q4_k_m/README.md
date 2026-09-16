# Granite 4.1 3B Q4_K_M qualification

The exact 2,099,501,664-byte GGUF artifact with SHA-256
`662b0626cd58f443baea23559b469df6576a81d349649c59413b36a9fb32eb29`
clears `production-rag-model-contribution-v6` at the `USABLE` tier.

The final three-iteration run used Java 25.0.4.1 on a 16-vCPU AMD EPYC 9R14 host (AWS c7a.4xlarge, us-east-1).
Recorded tuning for the performance phase: 16 native workers with 8 for single-token projections (`models.native.kernels.threads=16`, `models.native.kernels.decodeThreads=8`), native quantized decode, native grouped attention, and batched Java attention scores and values (`models.purejava.batchedAttentionScores/Values=true`). Prompt template `granite-documents`.
Models answered 27/27 grounded requests correctly;
model answer rate 0.444, model answer correct rate 1.000.

| Metric | Models | Ollama | llama.cpp | Gate result |
| --- | ---: | ---: | ---: | ---: |
| p95 TTFT | 1,039.3 ms | 626.4 ms | 999.7 ms | `USABLE` |
| Median decode | 25.72 tok/s | 30.25 tok/s | 30.62 tok/s | 0.850x Ollama (minimum 0.8x) |
| p95 end to end | 2,339.8 ms | 1,562.2 ms | 2,178.8 ms | 1.498x Ollama (maximum 1.5x) |
| Correct grounded answers | 27/27 | 27/27 | 27/27 | pass |

`models-rust-ffm.json`, `ollama.json`, and `llama.cpp.json` retain every measured request. Their
SHA-256 values are bound by `qualification.json`.

The separate library-default smoke used the same Models commit, left native quantized decode
disabled, and passed all 9 workload cases with no generation failures. Its
immutable report is `default-correctness/models-rust-ffm.json` (SHA-256 `c332a8cd7819b0d3b300257ad5f9c6f3a3aecd1abdf20cfe00fb380221cc10d3`).

Models: `models@f6252cc07f27dff957c2172895e7f5185e1d7e8b com.integrallis:vectors-core@0.1.21`.
