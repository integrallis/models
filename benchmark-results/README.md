# Models experiment journal

This is the chronological index for Models compatibility, correctness, and performance work. Raw
reports remain beside the workload that produced them; this page records the question that was
asked, the result that survived review, and how that result changed the product. External engines
appear only as independent oracles and benchmark peers. Production inference remains in-process.

## Evidence rules

- Pin source revisions, model bytes, prompts, runtime versions, and hardware.
- Establish numerical or token equivalence before making a performance claim.
- Pair kernel measurements with a real-model gate when the change can affect inference.
- Retain rejected experiments when they prevent an attractive but incorrect conclusion.
- Promote a model through ModelJars only after the public Models API passes the applicable
  correctness, quality, memory, and latency policy.

## Evolution

| Date | Question | Evidence and finding | Product decision |
| --- | --- | --- | --- |
| 2026-07-18 to 2026-07-24 | Can a Java-owned runtime load GGUF and run useful quantized models without importing an inference engine? | The [launch benchmark index](certified-20260724/README.md) compares pure Java, the Models-owned Rust/FFM kernels, llama.cpp, and Ollama with pinned prompts and artifacts. Vectors reports preserve the preceding compiler and quantized-kernel experiments. | Keep parsing, tokenization, graph execution, KV state, sampling, and generation in Java. Treat native code as a replaceable kernel provider, not a second runtime. |
| 2026-07-24 to 2026-07-26 | Can qualification measure grounded behavior rather than a successful model load? | The dated `certified-20260725` and `certified-20260726` RAG directories retain per-request outputs, independent peers, qualification summaries, and rejected variants across Qwen, Llama, Gemma, SmolLM, code, legal, finance, medical, and transit models. | Make correctness plus performance policy the catalog boundary. A parser or first token is not a qualification. |
| 2026-07-31 to 2026-08-01 | Does reusing unpacked K-quant data across prompt rows improve complete models? | The MiniCPM [Q6_K prefill report](certified-20260801/prefill/minicpm5-1b-q6k/README.md) and [session batching report](certified-20260801/decode/minicpm5-1b-session-batch-long/README.md) connect Vectors register tiles to full-model TTFT, throughput, output hashes, allocation, and session behavior. | Retain only hardware-scoped kernels and session layouts that pass both the primitive and end-to-end gates. |
| 2026-08-02 to 2026-08-03 | Can the same public runtime survive framework and large-model paths? | The [Gemma 4 qualification](certified-20260803/rag/gemma4-26b-a4b-q4_k_m/default-correctness/README.md) grew from a tester-reported Apple Silicon failure into controlled default-correctness evidence. Earlier `certified-20260802` reports exercise plain Java, Spring AI, and LangChain4j. | Add architecture- and hardware-specific regression gates before restoring a qualification; do not infer coverage from smaller demos. |
| 2026-08-24 to 2026-08-25 | Should Models remain GGUF-only? | The [Qwen2.5 BF16 qualification](certified-20260825/rag/qwen2.5-0.5b-instruct-bf16/README.md) exercises a standard Hugging Face config, tokenizer, and sharded Safetensors bundle through pure Java and compares it with Transformers. | Support ecosystem formats when Java can validate and execute them directly. Safetensors became a first-class input; it did not introduce a Python runtime dependency. |
| 2026-08-24 to 2026-08-27 | Can a compact architecture-specific artifact become a complete Java tool-calling model? | The [Needle qualification contract](../models-bench/src/main/resources/tool-qualification/README.md) and [retained report](../models-bench/benchmark-results/certified-20260827/tool-calling/needle2-cact-pure-java.json) cover strict CACT parsing, tokenizer and tensor fixtures, 13 upstream playground cases, constrained arguments, retrieval/confidence heads, and adapter behavior. | Add CACT for the exact Needle 2 graph and expose its capabilities through the same Models APIs. Do not pull the Needle engine or treat CACT as a generic architecture promise. |
| 2026-08-29 | Which FreeToken techniques and formats are worth adopting? | The [FreeToken compatibility audit](freetoken-compatibility-20260829/README.md) records pinned source revisions, dense Qwen3.5, Qwen state snapshots, GPT-OSS Safetensors/MXFP4, rejected prefill layouts, and the next MoE experiments. | Reimplement useful techniques in Java, use external code as an oracle, and require a measured limitation before adding a native shim. |
| 2026-08-29 | Can Java-authored GPU kernels accelerate the existing runtime? | The [A16](../models-accelerator-bench/results/vultr-a16-2q-2026-08-29.md) and [A40](../models-accelerator-bench/results/vultr-a40-4q-2026-08-29.md) reports prove exact Qwen output parity, capacity selection, bounded readiness, and warm Q4 projection/decode gains through TornadoVM. The attention candidate was slower and remains rejected. | Publish an optional Java/Tornado backend for the qualified NVIDIA envelope. Keep Vector API execution as the fallback and require real-device gates for other vendors. |
| 2026-08-30 | Can small new architectures meet the production catalog policy? | [Qwen3.5 0.8B](certified-20260830/rag/qwen3.5-0.8b-q4_k_m/README.md) and [Qwen2.5 3B](certified-20260830/rag/qwen2.5-3b-instruct-q4_k_m/README.md) both answer 27/27 grounded requests correctly. Their qualified profiles use only the narrow Rust/FFM recurrence or quantized-projection kernels needed to meet the latency envelope; library-default Java smokes remain mandatory. | Promote the exact qualified artifacts while retaining pure-Java correctness as the fallback and the next native-removal baseline. |
| 2026-08-30 to 2026-09-01 | Can compact embedding models run through the same Java-native stack? | [All-MiniLM-L6-v2](embedding/all-minilm-l6-v2-q4_k_m.json) and [Granite multilingual](embedding/granite-embedding-107m-multilingual-q4_k_m.json) compare the full pooled vectors with independent llama.cpp references. | Add bidirectional BERT/WordPiece and Granite unigram tokenization in Java, then promote only the checksum-bound artifacts whose equivalence floors pass. |
| 2026-09-01 | Can a 19 MiB cross-encoder reranker preserve an independent reference without a native runtime? | The [MS MARCO MiniLM L6 v2 experiment](reranking/ms-marco-minilm-l6-v2-q4-k-imatrix/README.md) pins exact bytes, pair tokens, segment IDs, ONNX and same-artifact Q4_K logits, rejected conversions, and real LangChain4j/Spring AI adapter gates. Java preserves the top-two order with a 0.101034 maximum delta from ONNX and 0.036392 from the quantized oracle. | Add a first-class Java reranking API and BERT classification head. Carry the missing scalar/vector `erf` forward as a measured JVM request; measure latency and memory in ModelJars before catalog promotion. |

## How the three projects connect

1. **Vectors** asks whether a storage layout or compute primitive is exact and measurably useful on
   a stated JVM and machine.
2. **Models** asks whether those primitives preserve a complete architecture, public API,
   framework adapter, and real workload.
3. **ModelJars** binds the accepted evidence to immutable model bytes and publishes only the
   resulting qualified coordinate.

That order is the project history and the release dependency order. A later optimization may
improve a primitive without changing a catalog artifact; a new artifact may be promoted without a
new Vectors or Models release when the existing runtime already passes its policy.
