# TinyLlama 1.1B Chat Q4_0 general RAG qualification

This directory contains controlled, same-host qualification evidence for the
exact TinyLlama 1.1B Chat v1.0 Q4_0 artifact:

- source: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`
- source revision: `52e7645ba7c309695bec7ac98f4f005b139cf465`
- ModelJar coordinate:
  `org.modeljars.huggingface:thebloke.tinyllama-1.1b-chat-v1.0-gguf.q4_0:1.0.0-q4_0.1`
- SHA-256:
  `da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556`
- size: 637,699,456 bytes
- license: Apache-2.0
- host: dedicated 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Linux amd64
- Models commit: `5c59bdf30ce29b039dfcce461b490f95b929976a`
- ModelJars profile commit used by the selected run: `a2a9120`
- Vectors commit: `c298a0b73970468794c1ba403022e2adc517e57e`
- Models JVM: Eclipse Adoptium Java 25.0.3, HotSpot C2
- native controls: Ollama 0.32.0 and llama.cpp b10012 (`c71854292`)

Every report uses the same artifact bytes, nine general RAG cases, corpus
SHA-256
`4b27eba8f166c84ef19c53de825445a6d0097f9bd8efa20b2d7013f34621f83c`,
top-1 retrieval, the `zephyr` template, a 2,048-token context, a 64-token
output limit, one complete warmup, and three measured iterations. Models,
Ollama, and llama.cpp ran sequentially in separate isolated processes on the
same dedicated host with eight CPU threads. The JDK applies only to Models;
Ollama and llama.cpp used their native runtimes.

## Results

| Backend | Tier | p95 TTFT (ms) | p50 decode (tok/s) | p95 E2E (ms) | Correct |
|---|---|---:|---:|---:|---:|
| Models Rust/FFM v10, explicit profile controls | USABLE | 1,347.5 | 34.18 | 3,194.7 | 100% |
| Models Rust/FFM v10, ModelJar profile | USABLE; qualified | 1,346.7 | 34.26 | 3,188.5 | 100% |
| Ollama 0.32.0 | PRODUCTION_READY | 498.1 | 31.45 | 2,940.9 | 100% |
| llama.cpp b10012 | PRODUCTION_READY | 972.2 | 64.00 | 1,970.1 | 100% |

The marker-driven Models run reaches 108.91% of Ollama and 53.53% of
llama.cpp decode throughput. Its p95 end-to-end latency is 1.084x Ollama and
1.618x llama.cpp. These ratios compare only the sequential process-isolated
runs on the same hardware described above; they are not cross-hardware
performance claims.

All 27 grounded answers were correct. Nine answers retained model text and all
nine were correct, fifteen used validated extractive fallback, and three
correctly abstained. The one-third model-contribution rate meets the production
policy exactly. The reports pin grounding policy
`trusted-title-provenance-statement-anchors-safe-discourse-explicit-abstention-v15`.

The marker-only run supplied no decode, worker-count, batch-size, or attention
system properties. Diagnostics record the exact
`tinyllama_1_1b_chat_v1_0_q4_0_epyc_milan_jdk25_rust_ffm` profile as
`ENABLED`, with no selector mismatch or missing launch argument. The profile
selects eight native workers, native quantized decode, 32-token batched
prefill, and disabled batched attention. All 27 prompt hashes, raw generations,
grounding decisions, evaluations, and raw evaluations match the explicit
control exactly.

`qualification-marker.json` binds the final marker report to both comparator
reports. `CertifiedRagEvidenceTest` recomputes the qualification decision and
the determinism and profile-selection assertions from these retained files.
