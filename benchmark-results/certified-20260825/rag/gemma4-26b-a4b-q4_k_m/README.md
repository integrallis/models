# Gemma 4 26B-A4B structured-prompt requalification

This directory supersedes the Models candidate and qualification recorded on
2026-08-02. The earlier candidate passed the rendered `gemma4` prompt through
the ordinary-text tokenizer path. The corrected run preserves trusted template
segments through `ModelPrompt`; its first input is 191 tokens, matching the
same-artifact llama.cpp and Ollama controls.

- source: `ggml-org/gemma-4-26B-A4B-it-GGUF`
- revision: `ae4d537a6345467d1c86bb5cc0d4505ff3ebe0f3`
- SHA-256: `88f4a13b0bb95f031a7fad973e10854122fb67ebc34d214d39a2f65053046abc`
- size: 16,796,015,136 bytes
- host: 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Linux amd64
- Models prompt fix: `ac80da9f89d908c1ac4b69cb95fa63c6b21cc890`
- Vectors: `0.1.12` staging candidate

The corrected candidate remains `USABLE`: all 27 requests are correct, 18
retain correct model text, six use validated extractive fallback, and three
correctly abstain. It records 1,861.7 ms p95 TTFT, 12.91 tokens/second median
decode throughput, and 4,318.2 ms p95 end-to-end latency. Policy v5 qualifies
it against the retained same-host llama.cpp and Ollama controls.

`default-correctness/models-rust-ffm.json` records the untuned 9/9 correctness
smoke. `models-rust-ffm.json` records the tuned 27-request run, and
`qualification.json` binds it to the immutable comparator reports.
