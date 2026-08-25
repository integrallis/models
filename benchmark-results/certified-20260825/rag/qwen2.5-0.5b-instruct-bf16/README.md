# Qwen2.5 0.5B BF16 Safetensors general RAG qualification

This directory contains same-host qualification evidence for the pinned
Hugging Face snapshot of Qwen2.5-0.5B-Instruct:

- revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- primary artifact SHA-256:
  `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`
- primary artifact size: 988,097,824 bytes
- host: 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Linux amd64
- JVM: Eclipse Adoptium Java 25.0.3
- Models prompt-boundary fix: `ac80da9`
- Vectors BF16 batch kernel: `b4b723d`
- reference: PyTorch 2.2.2 and Transformers 4.46.3, loading the same pinned
  local snapshot with remote code disabled

Both reports use the same nine-case general corpus (SHA-256
`4b27eba8f166c84ef19c53de825445a6d0097f9bd8efa20b2d7013f34621f83c`),
top-1 retrieval, ChatML, a 2,048-token context, a 64-token output limit,
greedy generation, one full warmup, and three measured iterations. Prompt
hashes are identical. The Java prompt path additionally has a tokenizer test
against the Transformers-observed 175-token first request, including the exact
ChatML boundary-token IDs.

## Results

| Backend | Tier | p95 TTFT (ms) | p50 decode (tok/s) | p95 E2E (ms) | Correct |
|---|---|---:|---:|---:|---:|
| Models pure Java | USABLE; qualified | 1,692.0 | 14.51 | 3,629.7 | 100% |
| Transformers BF16 CPU reference | OFFLINE reference | 52,271.9 | 2.97 | 61,097.6 | 100% |

Models reaches 4.89 times the reference's median decode throughput and 0.059
times its p95 end-to-end latency. More importantly, 15 of 27 Java attempts are
correct model answers, nine use validated extractive fallback, and three
correctly abstain. This clears the policy's one-third model-contribution floor
with 100% correctness among retained model answers.

The Transformers process is a pinned correctness and artifact-format
reference for a Safetensors bundle that Ollama and llama.cpp do not load. It is
not presented as a production CPU runtime. Models still has to clear the
independent absolute quality and latency gate before the relative comparison is
considered.

## Prompt-boundary regression caught by the comparison

An earlier Java run passed the final ChatML string through the ordinary-text
tokenizer API. That deliberately prevents user text from acquiring special
token privileges, so it also treated trusted template markers as ordinary
text: the first request used 197 tokens instead of the reference's 175 and
generation emitted a malformed `||im_end|>` suffix. The retained candidate
uses `ModelPrompt` segments end to end. Template markers are controls, while
retrieved evidence and the question remain ordinary text.
