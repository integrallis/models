<p align="center">
  <img src="media/icons/android-chrome-512x512.png" alt="models logo" width="200">
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
> GGUF parsing, vectors-backed F32/Q4_0/Q5_0/Q8_0/Q4_K/Q6_K kernels, tokenization,
> sampling, and a Llama-family forward path are implemented.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![JDK 25+](https://img.shields.io/badge/JDK-25%2B-orange.svg)](https://openjdk.org/projects/jdk/25/)
[![Pure Java](https://img.shields.io/badge/Pure%20Java-no%20JNI-brightgreen.svg)]()

---

> **Project status: pre-alpha.** The first publishable scope is
> `models-api`, `models-runtime`, `models-semantic-order`, and
> `models-backend-purejava`. Framework,
> Apple, ONNX, native, embedding, test, and benchmark modules remain experimental
> or scaffolded and are not part of release `0.1.x`.
> Real-model integration tests download and run the configured Qwen,
> Qwen-Coder, SQLCoder, SmolLM2, TinyLlama, DeepSeek-Coder, and MiniCPM GGUF
> fixtures before passing.

The [controlled inference study](INFERENCE_BENCHMARKS.md) compares the same GGUF
bytes through pure Java, llama.cpp, and Ollama and records the current
performance gap and optimization results.

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
- **Compact semantic orders**: WordTour models provide in-process lexical
  neighbors and sparse blurred bag-of-words without tensor inference or a
  vectors dependency.

```
application
    │
    ▼
models-runtime ──► models-api ──► models-backend-purejava
                                      │
                                      └── GGUF / F32 / Q4_0 / Q5_0 / Q8_0 / Q4_K / Q6_K
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
locally by small models. Initial CPU latency evidence is available in the
[controlled inference study](INFERENCE_BENCHMARKS.md); model quality remains a
separate, unproven requirement for the use cases below.

| Use case | Model size | Why local? |
|---|---|---|
| Agent heartbeats / keep-alive | 0.6–1B | no network dependency |
| Intent classification / routing | 0.6–1.7B | deterministic, no per-call cost |
| Tool dispatch / function calling | 1–4B | low latency in agent loops |
| Structured extraction (JSON) | 1–4B | privacy-sensitive data stays local |
| Embeddings for RAG | 0.5–1B | avoid embedding API costs at scale |
| Code completion in IDEs | 1–4B | offline-capable, responsive |
| Lexical expansion / lightweight classification | <1 MB semantic order | instant startup and bounded memory |

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

### Load a semantic-order model

The canonical WordTour marker bundles its 318,552-byte payload, so it needs no
separate model download:

```java
var requirement = ModelJarRequirement.forSource("github://joisino/wordtour")
    .versionRange("[1.0.0,2.0.0)")
    .variant("optimal")
    .backend("semantic-order")
    .build();

WordTour tour = WordTour.load(requirement);
var neighbors = tour.neighbors("concept", 5);
var document = BlurredBagOfWords.encode(
    tour,
    List.of("semantic", "search", "concept"));
```

WordTour lookup is exact and case-sensitive. Local proximity is meaningful;
large rank distance must not be interpreted as evidence that two terms are
unrelated.

### Framework adapters

`models-langchain4j` provides `ModelsChatModel`. The Spring Boot starter resolves
ModelJars descriptors and performance profiles and is the foundation for Spring
AI auto-configuration. `ModelsChatModel.diagnostics()` returns the same backend
plan used by plain Java generation; framework adapters do not select a separate
kernel path.

### Execution planning

`PureJavaBackend` builds one immutable execution plan when the GGUF is loaded.
The planner combines the tensors actually present in every layer, the structured
Vectors runtime capabilities, JVM/compiler identity, and explicit deployment
overrides. Grouped projections, batched prefill, and final-layer prompt pruning stages
are consumed from this plan instead of being re-decided in the forward loop.

```java
try (PureJavaBackend backend = PureJavaBackend.load(model)) {
    BackendDiagnostics diagnostics = backend.diagnostics();
    diagnostics.optimizations().forEach(System.out::println);
}
```

Diagnostics identify enabled, disabled, and unsupported choices, including the
resolved tensor grouping, mixed Q4_K/Q4_K/Q6_K projection eligibility, Q4_0
arithmetic kernel, prefill batch size, final-layer output-row policy, mapped
weights, Vector FMA policy, final-layer K/V-only policy, and persistent row
executor, including the staged Q4_0 FFN schedule when selected.
When the backend is loaded from a `ModelJarDescriptor`, diagnostics also retain
the exact marker coordinate and SHA and assess every measured performance
profile against the current JVM, processor topology, vector width, and startup
arguments. Model-scoped recommendations are applied only when the artifact SHA,
complete runtime selector, deterministic-output evidence, and required launch
arguments all match. A missing compiler flag is reported as requiring a process
restart; Models never pretends to apply JVM startup options after model loading.
`models.purejava.groupedProjections`, `models.purejava.mixedKProjections`,
`models.purejava.q4Kernel`, `models.purejava.prefillBatchSize`,
`models.purejava.finalLayerPrefillPruning`,
`models.purejava.finalLayerKvOnlyPrefill`,
`models.purejava.batchedAttentionScores`,
`models.purejava.batchedAttentionValues`, `models.purejava.stagedQ4Ffn`, and
`models.purejava.stagedQ4Layer` are parsed once per load. Malformed
explicit values fail rather than silently reverting to defaults, and explicit
deployment values override ModelJars recommendations. The Q4 kernel accepts
`widened`, `short-pairwise`, or `unsigned-pairwise`; ordinary loads use
`widened`. Optimized kernels require the corresponding Vectors runtime capability, while automatic
selection remains scoped to an exact measured ModelJars profile. Eligible
mixed-K Q/K/V projections share one Q8_K activation quantization and one row
dispatch. The mixed path remains inactive for every other tensor layout.
Batch-major prefill kernels cover Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, and Q6_K; the
Q5_0 route allows mixed DeepSeek-Coder files to retain batching instead of
degrading the complete prefill plan to one token at a time.

Batched attention-value accumulation is disabled for ordinary loads. An exact
ModelJar performance profile can enable it for a measured artifact and runtime,
or a deployment can select it explicitly with
`-Dmodels.purejava.batchedAttentionValues=true`. The route accumulates cached
value rows through the generic Vectors weighted-row primitive, preserves the
existing summation order, and is reported as `batched-attention-values` in
backend diagnostics.

Exact attention-score batching is also disabled for ordinary loads and selected
only by a matching profile or
`-Dmodels.purejava.batchedAttentionScores=true`. It shares each query-vector
load across two strided key-cache rows through Vectors while preserving the
active provider's independent dot-product result bit for bit. Diagnostics
report the route as `batched-attention-scores`.

Staged Q4_0 feed-forward execution is disabled for ordinary loads and selected
only by a matching profile or
`-Dmodels.purejava.stagedQ4Ffn=true`. Eligible layers require batched prefill,
thread-shareable Q4_0 gate, up, and down weights, parallel GGUF execution, and the
persistent Vectors row executor. The first stage runs gate/up projection, exact
SwiGLU, and Q8_0 preparation before the down-projection stage; diagnostics
report it as `staged-q4-ffn`. Ineligible runtimes keep the established path.

Retained Q4_0 layer execution is independently disabled for ordinary loads and
selected only by a matching profile or
`-Dmodels.purejava.stagedQ4Layer=true`. When the attention output, gate, up, and
down projections are all Q4_0, one four-stage Vectors publication spans output
projection; residual addition, FFN normalization, and Q8_0 preparation;
gate/up projection plus exact SwiGLU; and down projection. The FFN-only schedule
remains available for layers whose attention output uses another format.
Diagnostics report the wider route as `staged-q4-layer`.

Ordinary prefill requests logits only for the final prompt token. For validated
final layers whose attention and FFN projections are uniformly Q4_0 or uniformly
Q8_0, the default plan produces every K/V cache row but runs Q, attention, output
projection, and FFN only for the requested output row. Layouts that qualify only
for FFN pruning keep the narrower exact path. Other tensor layouts remain on the
full-row path until they pass their own controlled gate. Speculative verification
and observer-backed diagnostics request all rows and retain the complete path. Set
`-Dmodels.purejava.finalLayerKvOnlyPrefill=false` to retain only FFN pruning, or
`-Dmodels.purejava.finalLayerPrefillPruning=false` to disable both stages for
rollback or controlled A/B measurement.

## Supported models

The tested real-model fixtures are **Qwen3 0.6B Q4_0, 1.7B Q8_0, and 8B
Q4_K_M GGUF**, **Qwen2.5-Coder 0.5B/1.5B Q4_0/Q8_0 plus 3B Q4_0 GGUF**,
**SmolLM2 360M Q8_0 GGUF**, **TinyLlama 1.1B Chat v1.0 Q4_0 GGUF**,
**DeepSeek-Coder 1.3B Instruct Q4_K_M GGUF**, **MiniCPM5 1B Q4_K_M GGUF**,
and **Qwen2.5-Math 1.5B Instruct Q4_K_M GGUF**, resolved through ModelJars
marker JARs. The DeepSeek fixture validates a mixed Q4_K/Q5_0/Q8_0/Q6_K tensor
file and legacy linear RoPE scaling. MiniCPM5 validates explicit Q/K/V head
widths, 131K context metadata, and its Llama-style byte BPE. Qwen2.5-Math
validates Q6_K token embeddings and a deterministic arithmetic completion. The
0.5B Qwen2.5, 0.6B/8B Qwen3, TinyLlama, DeepSeek, MiniCPM5, and Qwen2.5-Math
fixtures have exact greedy-token reference checks against pinned `llama.cpp`
behavior. The backend accepts Llama/Qwen2/Qwen3 metadata prefixes. Projection
kernels support F32, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, and Q6_K; embedding rows also
support F16 across the same applicable quantized formats. Other architectures,
chat templates, long-context quality, and remaining K-quant formats are not yet
claimed.
The larger **Qwen2.5-Coder 7B Q4_0 GGUF**, **DeepSeek-Coder 6.7B Q4_K_M
GGUF**, **Qwen3 8B Q4_K_M GGUF**, and **DeepSeek-R1-Distill-Qwen-7B Q4_K_M
GGUF**, and **SQLCoder-7B-2 Q5_K_M GGUF** fixtures are covered by dedicated
strict slow-test tasks instead of the default integration suite. The DeepSeek R1
fixture also validates configured BOS handling for byte-level BPE tokenizers.

Resolve, download, and checksum the pinned fixtures through ModelJars:
```bash
./gradlew :models-backend-purejava:downloadQwen306BQ40Model
./gradlew :models-backend-purejava:downloadSmolLm2360MQ80Model
./gradlew :models-backend-purejava:downloadTinyLlama11BChatV10Q40Model
./gradlew :models-backend-purejava:downloadDeepSeekCoder13BQ4KMModel
./gradlew :models-backend-purejava:downloadDeepSeekCoder67BQ4KMModel
./gradlew :models-backend-purejava:downloadMiniCpm51BQ4KMModel
./gradlew :models-backend-purejava:downloadQwen38BQ4KMModel
./gradlew :models-backend-purejava:downloadDeepSeekR1DistillQwen7BQ4KMModel
./gradlew :models-backend-purejava:downloadQwen25Math15BQ4KMModel
./gradlew :models-backend-purejava:downloadSqlCoder7B2Q5KMModel
```

## What's inside

### Semantic orders (`models-semantic-order`)

- UTF-8 newline-delimited cyclic WordTour loading
- Compact binary-search rank index without a permanent term-to-rank hash map
- Deduplicated cyclic neighbor enumeration and shortest cycle distance
- Sparse Gaussian blurred bag-of-words with L1 normalization and distance
- Verified loading of compact payloads bundled in ModelJars
- No dependency on `vectors` or the Java Vector API

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

- **Dequantization**: Q4_0, Q4_K, Q5_0, Q8_0, Q6_K, and F16 storage paths
- **Quantized matmul**: operates directly on quantized `MemorySegment` data — no full dequantization needed
- **RMSNorm**: fused normalize + scale
- **Rotary Position Embeddings (RoPE)**: normal and NeoX layouts, configurable theta, modern and legacy linear scaling
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
| [models-semantic-order](models-semantic-order/) | experimental | Pure-Java WordTour lookup and sparse blurred bag-of-words |
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
models-semantic-order               <- ModelJars core; no vectors dependency
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

`models-backend-purejava` depends on `vectors-core` for dense GEMV kernels.
`models-semantic-order` performs rank lookup and sparse scalar operations and
does not require vectors. The runtime and public API modules remain independent.

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
sampling, and generation. Qwen2.5-Coder 0.5B Q4_0, Qwen3 0.6B Q4_0, TinyLlama
1.1B Q4_0, DeepSeek-Coder 1.3B Q4_K_M, MiniCPM5 1B Q4_K_M, and Qwen2.5-Math
1.5B Q4_K_M must also match exact greedy token sequences captured from
`llama.cpp` b9960.

The Qwen3 0.6B/1.7B, Qwen2.5-Coder 0.5B/1.5B/3B, Qwen2.5-Math 1.5B, SmolLM2
360M, TinyLlama 1.1B, DeepSeek-Coder 1.3B, and MiniCPM5 1B integration tests are
strict: the Gradle `integrationTest` task downloads the model fixtures before
the tests run, and the tests fail if any real model cannot be loaded. CI runs
this path in `.github/workflows/model-integration.yml` with the downloaded GGUF
cached under `~/.jvllm/models`.
Qwen2.5-Coder 7B Q4_0, DeepSeek-Coder 6.7B Q4_K_M, Qwen3 8B Q4_K_M,
DeepSeek-R1-Distill-Qwen-7B Q4_K_M, and SQLCoder-7B-2 Q5_K_M are strict
large-model fixtures. Each has a dedicated test task that resolves, downloads,
checksums, and runs only its model. The DeepSeek, Qwen3, and SQLCoder tests also
match four-token greedy `llama.cpp` b9960 references. CI runs the five tasks as
isolated matrix jobs in `.github/workflows/model-large-integration.yml` so no
runner must cache multiple 4-5 GB files.

The KV cache starts with 16 positions and grows geometrically, so loading a
long-context model no longer allocates its full advertised cache. The optional
`models.purejava.maxContextLength` property sets a hard runtime sequence limit
without changing the model metadata reported to callers.

```bash
./gradlew :models-backend-purejava:integrationTest \
  --tests com.integrallis.models.backend.purejava.Qwen3ModelJarsIntegrationTest \
  --tests com.integrallis.models.backend.purejava.Qwen25CoderModelJarsIntegrationTest \
  --tests com.integrallis.models.backend.purejava.SmolLm2ModelJarsIntegrationTest \
  --tests com.integrallis.models.backend.purejava.TinyLlamaModelJarsIntegrationTest \
  --tests com.integrallis.models.backend.purejava.DeepSeekCoderModelJarsIntegrationTest \
  --tests com.integrallis.models.backend.purejava.MiniCpm5ModelJarsIntegrationTest \
  --tests com.integrallis.models.backend.purejava.Qwen25MathModelJarsIntegrationTest

./gradlew :models-backend-purejava:qwen25Coder7BSlowTest
./gradlew :models-backend-purejava:deepSeekCoder67BSlowTest
./gradlew :models-backend-purejava:qwen38BSlowTest
./gradlew :models-backend-purejava:deepSeekR1DistillQwen7BSlowTest
./gradlew :models-backend-purejava:sqlCoder7B2SlowTest
./gradlew :models-backend-purejava:slowTest # aggregate large-model suite
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
- Strict integration tests against real Qwen, Qwen-Coder, Qwen-Math, SQLCoder,
  SmolLM2, TinyLlama, DeepSeek-Coder, MiniCPM5, and DeepSeek R1 fixtures

### Phase 2 — Framework adapters & production hardening

- Spring AI `ChatModel` adapter
- LangChain4j `ChatLanguageModel` adapter
- Chat template processing (Jinja2-style)
- Additional ModelJars catalog entries and repository providers
- Micrometer metrics (tok/s, latency histograms)
- JFR events for profiling
- Additional K-quant support beyond mixed Q4_K_M files

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
