# H2O Danube3 500M structured-prompt requalification

This directory supersedes the H2O Danube3 qualification recorded on
2026-08-03. The earlier Models run passed the fully rendered `h2o` prompt to
the ordinary-text tokenizer path, so the model did not receive the same
special-token structure as llama.cpp and Ollama.

The corrected run preserves the trusted template segments through
`ModelPrompt`. Its first input is 208 tokens, matching both native reference
runs over the same prompt bytes and exact artifact:

- source: `h2oai/h2o-danube3-500m-chat-GGUF`
- revision: `73d306884d4f4ed09fa6d146fe3260e3b739dc74`
- SHA-256: `021f78849c5670ecb2aa4cd7c5972eee0a3c9e41e33e5902c408a2ab989f0b43`
- size: 317,877,408 bytes
- host: 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Linux amd64
- Models prompt fix: `ac80da9f89d908c1ac4b69cb95fa63c6b21cc890`
- Vectors: `0.1.12` staging candidate

The tuned backend remains `PRODUCTION_READY`: 27/27 requests are correct,
p95 TTFT is 375.9 ms, median decode throughput is 82.03 tokens/second, and
p95 end-to-end latency is 1,155.5 ms. However, only 3/27 answers retain model
text; 21 use validated extractive fallback and three correctly abstain. The
11.1% model-answer rate fails the 33.3% contribution floor, so this exact
artifact is no longer production-RAG qualified.

`default-correctness/models-rust-ffm.json` records the untuned 9/9 smoke.
`models-rust-ffm.json` records the tuned 27-request run. `qualification.json`
applies policy v5 against the retained same-host llama.cpp and Ollama reports
from the superseded evidence directory and records
`FAILED_MODEL_CONTRIBUTION_GATE`.
