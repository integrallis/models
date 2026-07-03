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
> Pure Java. Zero native dependencies. JDK 25+.
> GGUF parsing, Q4_0/Q8_0 kernels, tokenization, sampling, and a Llama-family
> forward path are implemented. Framework adapters remain planned.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![JDK 25+](https://img.shields.io/badge/JDK-25%2B-orange.svg)](https://openjdk.org/projects/jdk/25/)
[![Pure Java](https://img.shields.io/badge/Pure%20Java-no%20JNI-brightgreen.svg)]()

---

> **Project status: pre-alpha.** The first publishable scope is
> `models-api`, `models-runtime`, and `models-backend-purejava`. The Spring AI,
> LangChain4j, Quarkus, Semantic Kernel, ONNX, native, embedding, test, and
> benchmark modules are scaffolding only and are not part of release `0.1.x`.
> Real-model integration tests currently target one local
> `Qwen3-0.6B-Q4_0.gguf` fixture and skip when it is absent.

## The pitch in 60 seconds

Most Java AI applications use remote inference services or a separate native
runtime. This project explores a narrower option: small GGUF models loaded and
executed inside a JDK 25 process.

The current implementation is a research-grade local runtime:

- **No native inference runtime**: no Python, ONNX Runtime, or llama.cpp.
- **GGUF-oriented**: parse GGUF v2/v3 and run the tensor types currently
  supported by the pure-Java backend.
- **Framework adapters are planned**: the current release exposes its own Java
  API and runtime, not Spring AI or LangChain4j implementations.
- **Scalar kernels today**: SIMD matmul is roadmap work, not a current
  performance claim.

```
application
    │
    ▼
models-runtime ──► models-api ──► models-backend-purejava
                                      │
                                      └── GGUF / Q4_0 / Q8_0 / F16 / F32
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
var model = Path.of(
    System.getProperty("user.home"), ".jvllm/models/Qwen3-0.6B-Q4_0.gguf");
try (var backend = PureJavaBackend.load(model)) {
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

Spring AI and LangChain4j adapters are roadmap items. No framework adapter
class is published in `0.1.x`.

## Supported models

The tested end-to-end fixture is **Qwen3 0.6B in Q4_0 GGUF format**.
The backend code accepts Llama/Qwen2/Qwen3 metadata prefixes and implements
F32, F16, Q4_0, and Q8_0 tensor paths. Other architectures, model sizes, chat
templates, long-context behavior, and K-quant formats are not yet claimed.

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

GPT-2-style byte-level BPE tokenizer loaded directly from GGUF metadata:

- `bytes_to_unicode` mapping for byte-level BPE vocabularies
- BPE merge-based encoding with priority queue
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
| [models-backend-purejava](models-backend-purejava/) | experimental | GGUF parser, scalar kernels, BPE tokenizer, KV cache, Llama forward pass |
| [models-backend-onnx](models-backend-onnx/) | planned | ONNX Runtime backend |
| [models-backend-native](models-backend-native/) | planned | llama.cpp via Panama FFM |
| [models-spring-ai](models-spring-ai/) | scaffold | Spring AI adapter placeholder |
| [models-langchain4j](models-langchain4j/) | scaffold | LangChain4j adapter placeholder |
| [models-quarkus](models-quarkus/) | planned | Quarkus extension |
| [models-semantic-kernel](models-semantic-kernel/) | planned | Semantic Kernel `ChatCompletionService` adapter |
| [models-spring-boot-starter](models-spring-boot-starter/) | planned | Auto-configuration (inference + optional vectors) |
| [models-embedding](models-embedding/) | scaffold | Planned bridge to vectors for embedding storage/search |
| [models-test](models-test/) | scaffold | Planned test-support integration |
| [models-bench](models-bench/) | planned | JMH benchmarks |

## Dependency graph

```
models-api                          <- foundation, no internal deps
models-runtime                      <- api
models-backend-purejava             <- api
models-backend-onnx                 <- scaffold, no dependencies
models-backend-native               <- scaffold, no dependencies
models-spring-ai                    <- scaffold, no dependencies
models-langchain4j                  <- scaffold, no dependencies
models-quarkus                      <- scaffold, no dependencies
models-semantic-kernel              <- scaffold, no dependencies
models-spring-boot-starter          <- scaffold, no dependencies
models-embedding                    <- scaffold, no dependencies
models-test                         <- scaffold, no dependencies
models-bench                        <- scaffold, no dependencies
```

## Relationship to vectors

**models** is a sister project to
[vectors](https://github.com/integrallis/vectors). The projects may eventually
provide complementary local inference and vector-search capabilities, but the
bridge module is not implemented or published today.

| Layer | Project | What it does |
|---|---|---|
| **Inference** | models | Run SLMs locally (tokenize → forward → sample → generate) |
| **Embedding & Search** | vectors | Store, index, and search vectors |
| **Bridge** | models-embedding | Planned integration; not implemented or published |

The first published models modules do not depend on vectors.

## Requirements

- **JDK 25+** (Foreign Function and Memory API)
- **Gradle 9.4+**

## Building

```bash
./gradlew build                  # compile all modules; release modules enforce SpotBugs + JaCoCo
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

When the fixture is present, integration tests exercise GGUF parsing,
tokenization, finite forward-pass outputs, sampling, and text generation against
real weights. They do not yet compare logits or generated tokens against a
reference runtime, so numerical correctness is still a release blocker.

## When to use models (and when not to)

| Use case | Recommendation |
|---|---|
| Evaluation and development against the tested Qwen3 Q4_0 fixture | Experimental fit |
| Production inference or framework integration | Not yet supported |
| RAG bridge to vectors | Planned; `models-embedding` is scaffolding |
| Production chat with 70B+ models, multi-turn | Use cloud APIs a hosted LLM API |
| High-throughput batch inference (>100 req/s) | Use vLLM / TGI with GPU |
| Training or fine-tuning models | Use Python ecosystem |
| Multi-modal inference (images, audio) | Not yet supported |

## Roadmap

### Phase 1 — Core inference pipeline (implemented, validation limited)

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

- SIMD-accelerated matmul using the JDK Vector API
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

- [`research.md`](../research.md) — consolidated research from three independent investigations
- [`research.md`](../research.md) — detailed architecture and feasibility analysis
- [`auggie-research.md`](../auggie-research.md) — model landscape and agentic AI positioning
- [`codex-research.md`](../codex-research.md) — JVM precedents and technical approach

## License

Licensed under the [Apache License 2.0](LICENSE).
