# Granite 4.1 3B Q4_K_M qualification qualification

The exact 2,099,501,664-byte GGUF artifact with SHA-256
`662b0626cd58f443baea23559b469df6576a81d349649c59413b36a9fb32eb29`
clears `production-rag-model-contribution-v6` at the `PRODUCTION_READY` tier.

The final three-iteration run used Java 25.0.4.1 on Java 25.0.4.1 on a 16-vCPU AMD EPYC 9R14 host (AWS c7a.4xlarge, us-east-1).
Recorded tuning for the performance phase: 16 native workers with 8 for single-token projections (`models.native.kernels.threads=16`, `models.native.kernels.decodeThreads=8`), native quantized decode, native grouped attention, and batched Java attention scores and values (`models.purejava.batchedAttentionScores/Values=true`). Prompt template `granite-documents`.. Prompt template `granite-documents`.
Models answered 27/27 grounded requests correctly;
model answer rate 0.444, model answer correct rate 1.000.

| Metric | Models | Ollama | llama.cpp | Gate result |
| --- | ---: | ---: | ---: | ---: |
| p95 TTFT | 961.4 ms | 626.8 ms | 1,007.6 ms | `PRODUCTION_READY` |
| Median decode | 26.02 tok/s | 29.55 tok/s | 30.42 tok/s | 0.881x Ollama (minimum 0.8x) |
| p95 end to end | 2,282.4 ms | 1,632.4 ms | 2,183.3 ms | 1.398x Ollama (maximum 1.5x) |
| Correct grounded answers | 27/27 | 27/27 | 27/27 | pass |

`models-rust-ffm.json`, `ollama.json`, and `llama.cpp.json` retain every measured request. Their
SHA-256 values are bound by `qualification.json`.

The separate library-default smoke used the same Models commit, left native quantized decode
disabled, and passed all 9 workload cases with no generation failures. Its
immutable report is `default-correctness/models-rust-ffm.json` (SHA-256 `3e1ae2222fba784e02943f430ef13f683682373cc97455380ce24e601111bb92`).

Models: `models@fcd38d31efc54cbd00747f1c2f732208c7f98b5b com.integrallis:vectors-core@0.1.21`.
