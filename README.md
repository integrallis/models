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

> **Experimental in-JVM small-language-model inference for JDK 25.**
>
> Pure-Java core backend. Optional platform bridge modules are isolated. JDK 25+.
> GGUF parsing, vectors-backed F32/Q4_0/Q8_0/Q6_K kernels, tokenization,
> sampling, and a Llama-family forward path are implemented.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![JDK 25+](https://img.shields.io/badge/JDK-25%2B-orange.svg)](https://openjdk.org/projects/jdk/25/)
[![Pure Java](https://img.shields.io/badge/Pure%20Java-no%20JNI-brightgreen.svg)]()

---

> **Project status: pre-alpha.** The first publishable scope is
> `models-api`, `models-runtime`, and `models-backend-purejava`. Framework,
> Apple, ONNX, native, embedding, test, and benchmark modules remain experimental
> or scaffolded and are not part of release `0.1.x`.
> Real-model integration tests download and run the configured Qwen/Qwen-Coder
> GGUF fixtures before passing.

## The pitch in 60 seconds

Most Java AI applications use remote inference services or a separate native
runtime. This project explores a narrower option: small GGUF models loaded and
executed inside a JDK 25 process.

The current implementation is a research-grade local runtime:

- **No native inference runtime in the pure-Java backend**: no Python, ONNX Runtime,
  or llama.cpp for `models-backend-purejava`.
- **GGUF-oriented**: parse GGUF v2/v3 and run the tensor types currently
  supported by the pure-Java backend.
- **Framework integration is emerging**: a LangChain4j `ChatModel` and Spring
  Boot ModelJars auto-configuration are implemented; broader Spring AI support
  remains experimental.
- **Vectors-backed kernels**: mapped F32 and quantized GEMV use `vectors-core`,
  with Panama Vector API SIMD when available and a scalar fallback.

```
application
    │
    ▼
models-runtime ──► models-api ──► models-backend-purejava
                                      │
                                      └── GGUF / F32 / Q4_0 / Q8_0 / Q6_K
```

## Why it exists

### The gap in the Java AI ecosystem

|  | Remote API | Separate local runtime | **models 0.1.x** |
|--|--|--|--|
| Language | Java client → HTTP → remote | Python/C++ | **Pure Java** |
| Process boundary | Network | IPC or local HTTP | **In-process** |
| Runtime dependency | API key + service | Native executable/runtime | **JDK 25** |
| Framework adapters | Commonly available | Runtime-specific | **Not implemented** |
| Model format | Service-defined | Runtime-specific | **Limited GGUF support** |

### Target use cases

The research hypothesis is that routine, narrow tasks can sometimes be served
locally by small models. The current project does not yet provide benchmark or
quality evidence for the use cases below.

| Use case | Model size | Why local? |
|---|---|---|
| Agent heartbeats / keep-alive | 0.6–1B | no network dependency |
| Intent classification / routing | 0.6–1.7B | deterministic, no per-call cost |
| Tool dispatch / function calling | 1–4B | low latency in agent loops |
| Structured extraction (JSON) | 1–4B | privacy-sensitive data stays local |
| Embeddings for RAG | 0.5–1B | avoid embedding API costs at scale |
| Code completion in IDEs | 1–4B | offline-capable, responsive |

## Quick start

### Load and generate

```java
var requirement = ModelJarRequirement.forSource("hf://ggml-org/Qwen3-0.6B-GGUF")
    .versionRange("[3.0.0,4.0.0)")
    .variant("q4_0")
    .backend("pure-java")
    .build();
var registry = ModelJarRegistry.fromClasspath();
new ModelJarInstaller(registry).install(requirement);
try (var backend = PureJavaBackend.load(requirement)) {
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

### Framework adapters

`models-langchain4j` provides `ModelsChatModel`. The Spring Boot starter resolves
ModelJars descriptors and is the foundation for Spring AI auto-configuration.

## Supported models

The tested real-model fixtures are **Qwen3 0.6B Q4_0 GGUF**,
**Qwen3 1.7B Q8_0 GGUF**, **Qwen2.5-Coder 0.5B/1.5B Q4_0/Q8_0 plus 3B Q4_0
GGUF**, **SmolLM2 360M Q8_0 GGUF**, and **TinyLlama 1.1B Chat v1.0 Q4_0 GGUF**,
resolved through ModelJars marker JARs. The 0.5B Qwen2.5, 0.6B Qwen3, and TinyLlama
fixtures also have exact greedy-token reference checks against pinned `llama.cpp`
behavior. The backend accepts Llama/Qwen2/Qwen3 metadata prefixes. Projection
kernels support F32, Q4_0, Q8_0, and Q6_K; embedding rows and small tensors also
support F16. Other architectures, chat templates, long-context quality, and
remaining K-quant formats are not yet claimed.
The larger **Qwen2.5-Coder 7B Q4_0 GGUF** fixture is covered by the strict
`slowTest` path instead of the default PR integration suite.

Resolve, download, and checksum the pinned fixtures through ModelJars:
```bash
./gradlew :models-backend-purejava:downloadQwen306BQ40Model
./gradlew :models-backend-purejava:downloadSmolLm2360MQ80Model
./gradlew :models-backend-purejava:downloadTinyLlama11BChatV10Q40Model
```

## What's inside

### GGUF parser (`models-backend-purejava`)

Zero-copy model loading via `MemorySegment` mmap. Parses headers, metadata, tensor info, and provides direct slices into quantized weight data without materializing full float arrays.

- GGUF v2/v3 format support
- All metadata value types (strings, arrays, typed scalars)
- Tensor data accessed via zero-copy `MemorySegment` slices
- Alignment-aware parsing (32-byte default alignment)

### GGUF tokenizers (`models-backend-purejava`)

GPT-2-style byte-level BPE and Llama SentencePiece tokenizers loaded directly
from GGUF metadata:

- `bytes_to_unicode` mapping for byte-level BPE vocabularies
- BPE merge-based encoding with priority queue
- SentencePiece score-priority merges, dummy-space prefix, BOS/EOS flags, and byte fallback
- Synthetic byte-level, Unicode, ranked-merge, and fallback regression tests
- Unicode, multibyte, and code-point aware

### Quantized inference kernels (`models-backend-purejava`)

- **Dequantization**: Q4_0 (4-bit, 18-byte blocks), Q8_0 (8-bit, 34-byte blocks), F16 → F32
- **Quantized matmul**: operates directly on quantized `MemorySegment` data — no full dequantization needed
- **RMSNorm**: fused normalize + scale
- **Rotary Position Embeddings (RoPE)**: in-place, configurable theta
- **SwiGLU activation**: fused gate × silu × up projection
- **Softmax**: numerically stable (max-subtract)

### Transformer forward path (`models-backend-purejava`)

Implemented Llama-family decoder path:

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
| [models-api](models-api/) | experimental | Backend SPI, `Tokenizer`, `SamplingOptions`, `TokenStream`, `ModelMetadata` |
| [models-runtime](models-runtime/) | experimental | `GenerationLoop` and `Sampler` |
| [models-backend-purejava](models-backend-purejava/) | experimental | GGUF parser, vectors-backed kernels, BPE tokenizer, KV cache, Llama forward pass |
| [models-backend-apple](models-backend-apple/) | experimental | Optional Apple Foundation Models bridge via Java FFM and a tiny Swift C ABI dylib |
| [models-backend-onnx](models-backend-onnx/) | planned | ONNX Runtime backend |
| [models-backend-native](models-backend-native/) | planned | llama.cpp via Panama FFM |
| [models-spring-ai](models-spring-ai/) | scaffold | Spring AI adapter placeholder |
| [models-langchain4j](models-langchain4j/) | experimental | LangChain4j `ChatModel` adapter |
| [models-quarkus](models-quarkus/) | planned | Quarkus extension |
| [models-semantic-kernel](models-semantic-kernel/) | planned | Semantic Kernel `ChatCompletionService` adapter |
| [models-spring-boot-starter](models-spring-boot-starter/) | experimental | ModelJars registry and descriptor auto-configuration |
| [models-embedding](models-embedding/) | experimental | Optional bridge to vectors for embedding storage/search |
| [models-test](models-test/) | scaffold | Planned test-support integration |
| [models-bench](models-bench/) | planned | JMH benchmarks |

## Dependency graph

```
models-api                          <- foundation, no internal deps
models-runtime                      <- api
models-backend-purejava             <- api + vectors-core
models-backend-apple                <- api + optional Apple Foundation Models dylib
models-backend-onnx                 <- scaffold, no dependencies
models-backend-native               <- scaffold, no dependencies
models-spring-ai                    <- scaffold, no dependencies
models-langchain4j                  <- api + runtime + LangChain4j
models-quarkus                      <- scaffold, no dependencies
models-semantic-kernel              <- scaffold, no dependencies
models-spring-boot-starter          <- ModelJars core + Spring Boot
models-embedding                    <- api + vectors-db + vectors-cache-semantic-db
models-test                         <- scaffold, no dependencies
models-bench                        <- scaffold, no dependencies
```

## Relationship to vectors

**models** is a sister project to
[vectors](https://github.com/integrallis/vectors). Low-level SIMD and
MemorySegment-friendly numeric kernels live in `vectors`; model loading,
tokenization, transformer semantics, KV cache, and generation stay in `models`.

| Layer | Project | What it does |
|---|---|---|
| **Inference** | models | Run SLMs locally (tokenize → forward → sample → generate) |
| **SIMD kernels** | vectors-core | JDK Vector API primitives reused by the pure-Java backend |
| **Embedding & Search** | vectors | Store, index, and search vectors |
| **Bridge** | models-embedding | Optional embedding storage/search integration; not published in 0.1.x |

`models-backend-purejava` depends on `vectors-core` for dense GEMV kernels. The
runtime and public API modules remain independent.

## Requirements

- **JDK 25+** (Foreign Function and Memory API)
- **Gradle 9.4+**

## Building

```bash
./gradlew build                  # compile all modules; release modules enforce SpotBugs + JaCoCo
./gradlew test                   # unit tests (excludes slow/benchmark/integration)
./gradlew unitTest               # @Tag("unit") only
./gradlew integrationTest        # downloads, verifies, and runs real-model fixtures
./gradlew spotlessApply          # Google Java Format 1.35.0
./gradlew publishToMavenLocal    # install to local repo

# Run a single test class
./gradlew :models-backend-purejava:test --tests "com.integrallis.models.backend.purejava.gguf.GgufParserTest"
```

### Integration tests

Integration tests resolve immutable model revisions from the ModelJars catalog,
download missing files, verify size and SHA-256, and then execute the real weights:

```bash
./gradlew :models-backend-purejava:integrationTest
```

The suite exercises GGUF parsing, tokenization, finite forward-pass outputs,
sampling, and generation. Qwen2.5-Coder 0.5B Q4_0, Qwen3 0.6B Q4_0, and TinyLlama
1.1B Q4_0 must also match exact greedy token sequences captured from `llama.cpp`
b9960.

The Qwen3 0.6B/1.7B, Qwen2.5-Coder 0.5B/1.5B/3B, SmolLM2 360M, and TinyLlama
1.1B integration tests are strict:
the Gradle `integrationTest` task downloads the model fixtures before the tests
run, and the tests fail if any real model cannot be loaded. CI runs this path in
`.github/workflows/model-integration.yml` with the downloaded GGUF cached under
`~/.jvllm/models`.
The 7B Q4_0 fixture is strict too, but lives under `slowTest` and
`.github/workflows/model-large-integration.yml` so the default PR cache does not
exceed practical GitHub Actions limits.

The KV cache starts with 16 positions and grows geometrically, so loading a
long-context model no longer allocates its full advertised cache. The optional
`models.purejava.maxContextLength` property sets a hard runtime sequence limit
without changing the model metadata reported to callers.

```bash
./gradlew :models-backend-purejava:integrationTest \
  --tests com.integrallis.models.backend.purejava.Qwen25CoderModelJarsIntegrationTest \
  --tests com.integrallis.models.backend.purejava.SmolLm2ModelJarsIntegrationTest \
  --tests com.integrallis.models.backend.purejava.TinyLlamaModelJarsIntegrationTest

./gradlew :models-backend-purejava:slowTest \
  --tests com.integrallis.models.backend.purejava.Qwen25CoderLargeModelJarsSlowTest
```

## When to use models (and when not to)

| Use case | Recommendation |
|---|---|
| Evaluation and development against the tested Qwen3 Q4_0 fixture | Experimental fit |
| Production inference or framework integration | Not yet supported |
| RAG bridge to vectors | Experimental; `models-embedding` provides the optional vectors bridge |
| Production chat with 70B+ models, multi-turn | Use a hosted LLM API |
| High-throughput batch inference (>100 req/s) | Use vLLM / TGI with GPU |
| Training or fine-tuning models | Use Python ecosystem |
| Multi-modal inference (images, audio) | Not yet supported |

## Roadmap

### Phase 1 — Core inference pipeline (implemented, validation limited)

- GGUF binary format parser (v2/v3, zero-copy mmap)
- GPT-2 byte-level BPE and Llama SentencePiece tokenizers
- Dequantization kernels (Q4_0, Q8_0, F16)
- Tensor operations (RMSNorm, matmul, quantized matmul, softmax, RoPE, SwiGLU)
- KV cache for autoregressive decoding
- Llama-family forward pass (supports Qwen2/Qwen3/Llama architectures)
- Sampling strategies (greedy, temperature, top-k, top-p, repetition penalty)
- Generation loop with streaming
- Strict integration tests against real Qwen, Qwen-Coder, SmolLM2, and TinyLlama fixtures

### Phase 2 — Framework adapters & production hardening

- Spring AI `ChatModel` adapter
- LangChain4j `ChatLanguageModel` adapter
- Chat template processing (Jinja2-style)
- Additional ModelJars catalog entries and repository providers
- Micrometer metrics (tok/s, latency histograms)
- JFR events for profiling
- Q4_K_M and additional K-quant support (wider quant format coverage)

### Phase 3 — Performance & scale

- Broader SIMD coverage and kernel benchmarking
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

- models-embedding bridge to vectors (generate + store + search)
- Spring Boot starter with auto-configuration
- Structured output (JSON schema-constrained generation)
- Grammar-guided decoding
- LoRA adapter loading

## Further reading

- [`research.md`](../research.md) — consolidated research from independent investigations
- [`auggie-research.md`](../auggie-research.md) — model landscape and agentic AI positioning
- [`codex-research.md`](../codex-research.md) — JVM precedents and technical approach

## License

Licensed under the [Apache License 2.0](LICENSE).
