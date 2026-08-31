# Qwen3.5 0.8B Q4_K_M qualification

The exact 532,517,120-byte GGUF artifact with SHA-256
`bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517`
clears `production-rag-model-contribution-v5` at the `USABLE` tier.

The final three-iteration run used Java 25.0.4 on a 16-core AMD EPYC 9R14
host. The selected profile combines Java Vector API convolution, gate, SwiGLU,
and attention kernels with the Models-owned Gated DeltaNet FFM recurrence.
Models answered all 27 grounded requests correctly; 15 answers retained
model-written text.

| Metric | Models | Ollama | Gate result |
| --- | ---: | ---: | ---: |
| p95 TTFT | 1,413.0 ms | 595.5 ms | `USABLE` |
| Median decode | 80.61 tok/s | 62.73 tok/s | 1.285x (minimum 0.8x) |
| p95 end to end | 1,648.6 ms | 1,231.7 ms | 1.338x (maximum 1.5x) |
| Correct grounded answers | 27/27 | 27/27 | pass |

`models-rust-ffm.json` and `ollama.json` retain every measured request. Their
SHA-256 values are bound by `qualification.json`.

The separate library-default smoke used the released Models `4934a6c1` code,
left native quantized decode disabled, and passed all 9 workload cases with no
generation failures. Its immutable report is
`default-correctness/models-rust-ffm.json` (SHA-256
`803f6cb0b90365897f32731225b66024a407b35126e588e807b87b40cdc4c4fc`).
