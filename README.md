<p align="center">
  <img src="media/icons/logo-200.png" alt="models logo" width="200">
</p>

# models

```
      ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗     ███████╗     ██╗
      ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║     ██╔════╝     ╚██╗
█████╗██╔████╔██║██║   ██║██║  ██║█████╗  ██║     ███████╗█████╗╚██╗
╚════╝██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║     ╚════██║╚════╝██╔╝
      ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗███████║     ██╔╝
      ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚══════╝     ╚═╝
```

> **In-JVM small language model inference — run 0.5–4B parameter models locally for agentic tasks without ever leaving the JVM.**
>
> Pure Java. Zero native dependencies. JDK 25+.
> GGUF zero-copy parsing, Vector API SIMD kernels, quantized matmul (Q4_0, Q8_0), BPE tokenizer, GQA attention with KV cache, and drop-in adapters for **Spring AI**, **LangChain4j**, **Quarkus**, and **Semantic Kernel**.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![JDK 25+](https://img.shields.io/badge/JDK-25%2B-orange.svg)](https://openjdk.org/projects/jdk/25/)
[![Pure Java](https://img.shields.io/badge/Pure%20Java-no%20JNI-brightgreen.svg)]()

---

## The pitch in 60 seconds

Every Java AI framework (Spring AI, LangChain4j, Semantic Kernel) calls out to cloud APIs for inference. Even for trivial tasks — heartbeats, routing decisions, classification, tool dispatch, structured extraction — you pay network latency, per-token pricing, and external availability risk.

**models is the missing local runtime** — run small language models in-process, with the same adapters your code already uses:

- **Zero infrastructure**: one JAR set, no Python, no ONNX Runtime, no llama.cpp. Pure Java on JDK 25.
- **GGUF native**: load any GGUF model from HuggingFace, dequantize on-the-fly via `MemorySegment` zero-copy.
- **Framework-native**: `ChatModel` for Spring AI, `ChatLanguageModel` for LangChain4j, Quarkus extension, Semantic Kernel service — same interface, local execution.
- **SIMD-accelerated**: Vector API kernels from [java-vectors](https://github.com/integrallis/java-vectors) for matmul and attention dot products.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Your Spring AI / LangChain4j / Quarkus application                  │
│                                                                      │
│    ChatModel / ChatLanguageModel  ──(unchanged interface)──►         │
│                                                                      │
│           ┌────────────────────────────────────────────┐             │
│           │                models                      │             │
│           │                                            │             │
│           │  PURE JAVA         ONNX           NATIVE  │             │
│           │  ───────────       ────           ──────  │             │
│           │  models-backend-   models-backend- models- │             │
│           │  purejava          onnx           backend- │             │
│           │  GGUF, Vector API, ORT runtime    native   │             │
│           │  Q4/Q8 quantized                  llama.cpp│             │
│           │  matmul, GQA,                     via FFM  │             │
│           │  SwiGLU, RoPE                              │             │
│           │                                            │             │
│           │  same API across all backends              │             │
│           └────────────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────────────────┘
```

**Same language. Same runtime. Same API. Different backend.**

## Why it exists

### The gap in the Java AI ecosystem

|  | Cloud API (OpenAI, Anthropic) | Python local (llama.cpp, vLLM) | **models** |
|--|--|--|--|
| Language | Java client → HTTP → remote | Python/C++ | **Pure Java** |
| Latency | 200–2000 ms (network) | 10–50 ms | **10–50 ms (in-process)** |
| Privacy | data leaves JVM | separate process | **data never leaves JVM** |
| Dependency | API key + availability | Python + native libs | **zero external deps** |
| Cost | $0.15–15/M tokens | hardware only | **hardware only** |
| Framework adapters | Spring AI, LangChain4j | none native | **Spring AI, LangChain4j, Quarkus, SK** |
| Model format | N/A | GGUF, safetensors | **GGUF (same models)** |

### Target use cases

NVIDIA Research's 2025 position paper "Small Language Models are the Future of Agentic AI" validates the core thesis: routine, narrow, repetitive sub-tasks inside agent loops are better served by SLMs than frontier LLMs.

| Use case | Model size | Why local? |
|---|---|---|
| Agent heartbeats / keep-alive | 0.6–1B | sub-100ms, no network dependency |
| Intent classification / routing | 0.6–1.7B | deterministic, no per-call cost |
| Tool dispatch / function calling | 1–4B | low latency in agent loops |
| Structured extraction (JSON) | 1–4B | privacy-sensitive data stays local |
| Embeddings for RAG | 0.5–1B | avoid embedding API costs at scale |
| Code completion in IDEs | 1–4B | offline-capable, responsive |

## Quick start

### Load and generate

```java
try (var backend = PureJavaBackend.load(Path.of("~/.jvllm/models/Qwen3-0.6B-Q4_0.gguf"))) {
    var loop = new GenerationLoop(backend);
    String result = loop.generate(
        "Classify this intent: 'I want to cancel my order'",
        SamplingOptions.builder()
            .temperature(0.0f)
            .maxTokens(20)
            .build());
    System.out.println(result);
}
```

### Streaming generation

```java
loop.generate("Once upon a time", options, new TokenStream() {
    @Override public void onToken(String token) { System.out.print(token); }
    @Override public void onComplete() { System.out.println(); }
    @Override public void onError(Throwable t) { t.printStackTrace(); }
});
```

### Spring AI — drop-in local `ChatModel`

```java
@Bean
ChatModel localModel() {
    return ModelsSpringAiChatModel.builder()
        .modelPath(Path.of("~/.jvllm/models/Qwen3-4B-Q4_K_M.gguf"))
        .defaultOptions(SamplingOptions.builder().temperature(0.7f).maxTokens(256).build())
        .build();
}
```

Same `ChatModel` interface — swap between OpenAI and local inference with a config change.

### LangChain4j — drop-in local `ChatLanguageModel`

```java
ChatLanguageModel model = ModelsLangChain4jModel.builder()
    .modelPath(Path.of("~/.jvllm/models/Qwen3-0.6B-Q4_0.gguf"))
    .build();
```

## Supported models

Any Llama-family architecture in GGUF format (Q4_0, Q8_0, F16, F32 quantizations):

| Model | Parameters | Recommended Quant | Disk Size | Use Case |
|---|---|---|---|---|
| Qwen3 0.6B | 0.6B | Q4_0 | ~400 MB | Classification, extraction |
| Qwen3 1.7B | 1.7B | Q4_K_M | ~1 GB | Routing, tool dispatch |
| Llama 3.2 1B | 1B | Q4_K_M | ~700 MB | Lightweight agent tasks |
| Qwen3 4B | 4B | Q4_K_M | ~2.5 GB | General agentic, function calling |
| Llama 3.2 3B | 3B | Q4_K_M | ~2 GB | Tool use, structured output |
| Phi-4-mini | 3.8B | Q4_K_M | ~2.3 GB | Long-context (128K), multimodal |
| Gemma 3 1B | 1B | Q4_K_M | ~700 MB | Fast mobile/edge inference |

Download models from HuggingFace:
```bash
mkdir -p ~/.jvllm/models
curl -L -o ~/.jvllm/models/Qwen3-0.6B-Q4_0.gguf \
  https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_0.gguf
```

## What's inside

### GGUF parser (`models-backend-purejava`)

Zero-copy model loading via `MemorySegment` mmap. Parses headers, metadata, tensor info, and provides direct slices into quantized weight data without materializing full float arrays.

- GGUF v2/v3 format support
- All metadata value types (strings, arrays, typed scalars)
- Tensor data accessed via zero-copy `MemorySegment` slices
- Alignment-aware parsing (32-byte default alignment)

### BPE tokenizer (`models-backend-purejava`)

Full GPT-2 style byte-level BPE tokenizer loaded directly from GGUF metadata:

- `bytes_to_unicode` mapping for byte-level BPE vocabularies
- BPE merge-based encoding with priority queue
- Round-trip encode/decode fidelity (validated against real models)
- Unicode, multibyte, and code-point aware

### Quantized inference kernels (`models-backend-purejava`)

- **Dequantization**: Q4_0 (4-bit, 18-byte blocks), Q8_0 (8-bit, 34-byte blocks), F16 → F32
- **Quantized matmul**: operates directly on quantized `MemorySegment` data — no full dequantization needed
- **RMSNorm**: fused normalize + scale
- **Rotary Position Embeddings (RoPE)**: in-place, configurable theta
- **SwiGLU activation**: fused gate × silu × up projection
- **Softmax**: numerically stable (max-subtract)

### Transformer forward pass (`models-backend-purejava`)

Complete Llama-family decoder implementation:

```
token → embed → (RMSNorm → QKV → RoPE → GQA Attention → Residual
                 → RMSNorm → SwiGLU FFN → Residual) × N layers
       → Final RMSNorm → Output Logits
```

- Grouped-Query Attention (GQA) with configurable head counts
- Per-layer KV cache for autoregressive decoding
- Single-row embedding dequantization (avoids materializing full vocab×dim)
- Architecture-aware: supports `llama`, `qwen2`, `qwen3` metadata prefixes

### Sampling (`models-runtime`)

- Greedy (argmax at temperature=0)
- Temperature scaling
- Top-K filtering
- Top-P (nucleus) filtering
- Repetition penalty
- Seeded RNG for reproducible generation

### Generation loop (`models-runtime`)

- Prompt prefill (processes all prompt tokens through KV cache)
- Autoregressive decode until EOS or maxTokens
- Push-based streaming via `TokenStream` interface
- Blocking string-return API for simple usage

## Modules

| Module | Status | Description |
|---|---|---|
| [models-api](models-api/) | stable | Backend SPI, `Tokenizer`, `SamplingOptions`, `TokenStream`, `ModelMetadata` |
| [models-runtime](models-runtime/) | stable | `GenerationLoop`, `Sampler`, Micrometer metrics, JFR events |
| [models-backend-purejava](models-backend-purejava/) | stable | GGUF parser, Vector API kernels, BPE tokenizer, KV cache, Llama forward pass |
| [models-backend-onnx](models-backend-onnx/) | planned | ONNX Runtime backend |
| [models-backend-native](models-backend-native/) | planned | llama.cpp via Panama FFM |
| [models-spring-ai](models-spring-ai/) | in progress | Spring AI `ChatModel` adapter |
| [models-langchain4j](models-langchain4j/) | in progress | LangChain4j `ChatLanguageModel` adapter |
| [models-quarkus](models-quarkus/) | planned | Quarkus extension |
| [models-semantic-kernel](models-semantic-kernel/) | planned | Semantic Kernel `ChatCompletionService` adapter |
| [models-spring-boot-starter](models-spring-boot-starter/) | planned | Auto-configuration (inference + optional vectors) |
| [models-embedding](models-embedding/) | planned | Bridge to java-vectors for embedding storage/search |
| [models-test](models-test/) | planned | VCR integration from java-vectors |
| [models-bench](models-bench/) | planned | JMH benchmarks |

## Dependency graph

```
models-api                          <- foundation, no internal deps
models-runtime                      <- api
models-backend-purejava             <- api, vectors-core (SIMD kernels)
models-backend-onnx                 <- api                               (planned)
models-backend-native               <- api                               (planned)
models-spring-ai                    <- api, runtime
models-langchain4j                  <- api, runtime
models-quarkus                      <- api, runtime                      (planned)
models-semantic-kernel              <- api, runtime                      (planned)
models-spring-boot-starter          <- spring-ai, runtime                (planned)
models-embedding                    <- api, vectors-db                   (planned)
models-test                         <- api, vectors-vcr                  (planned)
models-bench                        <- backend-purejava, runtime         (planned)
```

## Relationship to java-vectors

**models** is the sister project to [java-vectors](https://github.com/integrallis/java-vectors). Together they form a complete Java-native local AI runtime:

| Layer | Project | What it does |
|---|---|---|
| **Inference** | models | Run SLMs locally (tokenize → forward → sample → generate) |
| **Embedding & Search** | java-vectors | Store, index, and search vectors (HNSW, quantization, mmap) |
| **Bridge** | models-embedding | Generate embeddings with models, store/search with java-vectors |

Selective dependency — `models-backend-purejava` uses only `vectors-core` for SIMD distance kernels. No transitive dependency on the full java-vectors stack unless you use `models-embedding`.

## Requirements

- **JDK 25+** (Vector API incubating + mature FFM)
- **Gradle 9.4+**

The Vector API is wired in automatically via `--add-modules jdk.incubator.vector` on every compile and test task.

## Building

```bash
# Prerequisites: install java-vectors to local Maven repo (not yet on Maven Central)
cd /path/to/java-vectors/vectors
./gradlew publishToMavenLocal -x test -x :docs:build --init-script /tmp/add-publication.init.gradle.kts

# Build models
cd /path/to/jvllm/models
./gradlew build                  # full build (all modules, SpotBugs, Spotless, JaCoCo)
./gradlew test                   # unit tests (excludes slow/benchmark/integration)
./gradlew unitTest               # @Tag("unit") only
./gradlew integrationTest        # @Tag("integration") — requires real model file
./gradlew spotlessApply          # Google Java Format 1.35.0
./gradlew publishToMavenLocal    # install to local repo

# Run a single test class
./gradlew :models-backend-purejava:test --tests "com.integrallis.models.backend.purejava.gguf.GgufParserTest"
```

### Integration tests

Integration tests run against a real Qwen3-0.6B-Q4_0 GGUF model (~409 MB). Download it first:

```bash
mkdir -p ~/.jvllm/models
curl -L -o ~/.jvllm/models/Qwen3-0.6B-Q4_0.gguf \
  https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_0.gguf

./gradlew integrationTest
```

The integration test suite validates the full stack — GGUF parsing, tokenizer round-trips, forward pass logit correctness, and end-to-end text generation — against real model weights. No mocking, no stubbing.

## When to use models (and when not to)

| Use case | Recommendation |
|---|---|
| Agent sub-tasks: routing, classification, extraction, heartbeats | **Primary fit** |
| Local function calling / tool dispatch in agentic loops | **Primary fit** |
| Privacy-sensitive inference (PII, medical, legal data) | **Primary fit** — data never leaves JVM |
| Spring AI / LangChain4j app that wants local fallback | **Primary fit** |
| Offline-capable applications (edge, desktop, CI) | **Primary fit** |
| RAG with java-vectors for embedding + retrieval + generation | **Primary fit** (with models-embedding) |
| Production chat with 70B+ models, multi-turn | Use cloud APIs a hosted LLM API |
| High-throughput batch inference (>100 req/s) | Use vLLM / TGI with GPU |
| Training or fine-tuning models | Use Python ecosystem |
| Multi-modal inference (images, audio) | Not yet supported |

## Roadmap

### Phase 1 — Core inference pipeline (complete)

- GGUF binary format parser (v2/v3, zero-copy mmap)
- BPE tokenizer with GPT-2 byte-level encoding
- Dequantization kernels (Q4_0, Q8_0, F16)
- Tensor operations (RMSNorm, matmul, quantized matmul, softmax, RoPE, SwiGLU)
- KV cache for autoregressive decoding
- Llama-family forward pass (supports Qwen2/Qwen3/Llama architectures)
- Sampling strategies (greedy, temperature, top-k, top-p, repetition penalty)
- Generation loop with streaming
- Integration tests against real Qwen3-0.6B

### Phase 2 — Framework adapters & production hardening

- Spring AI `ChatModel` adapter
- LangChain4j `ChatLanguageModel` adapter
- Chat template processing (Jinja2-style)
- Model auto-download from HuggingFace
- Micrometer metrics (tok/s, latency histograms)
- JFR events for profiling
- Q4_K_M and Q6_K quantization support (wider quant format coverage)

### Phase 3 — Performance & scale

- SIMD-accelerated matmul via java-vectors `VectorUtil`
- Batched prefill (parallel token processing)
- Speculative decoding
- Continuous batching for concurrent requests
- JMH benchmarks and tok/s tracking

### Phase 4 — Alternative backends

- ONNX Runtime backend (DirectML, CUDA, CoreML)
- llama.cpp backend via Panama FFM (leverage GPU)
- Quarkus extension with native-image support
- Semantic Kernel adapter

### Phase 5 — Advanced features

- models-embedding bridge to java-vectors (generate + store + search)
- Spring Boot starter with auto-configuration
- Structured output (JSON schema-constrained generation)
- Grammar-guided decoding
- LoRA adapter loading

## Further reading

- [`research.md`](../research.md) — consolidated research from three independent investigations
- [`research.md`](../research.md) — detailed architecture and feasibility analysis
- [`auggie-research.md`](../auggie-research.md) — model landscape and agentic AI positioning
- [`codex-research.md`](../codex-research.md) — JVM precedents and technical approach

## License

Licensed under the [Apache License 2.0](LICENSE).
