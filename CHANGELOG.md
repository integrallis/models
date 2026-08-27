# Changelog

All notable changes to models are documented here.

## [Unreleased]

### Added

- Added the first-class `needle2` chat template and tool syntax. It renders the
  official raw-schema prompt contract, system facts, array-wrapped parallel
  calls, and plain user turns for tool results, and recovers generated calls
  through the shared runtime scanner.
- Added schema-derived constrained decoding for Needle 2 tool calls, including
  parallel calls, declared argument types, and the model's refusal form.
- Exposed optional in-model contrastive and confidence heads through the backend,
  runtime-model, and `InferencePipeline` APIs.
- Added bounded, cached tool retrieval using Needle 2's own contrastive head.
  The LangChain4j and Spring AI chat adapters automatically reduce declarations
  larger than five tools before rendering both blocking and streaming requests.
- Added a controlled Needle 2 tool-conformance qualification workload based on
  the upstream playground cases and immutable artifact/source evidence.

### Fixed

- Prevented finite single-call constraints from being applied to Needle 2's
  reasoning-prefixed call arrays.
- Corrected the CACT attention sink and 8-bit KV-cache interpretation, loaded the
  serialized contrastive and confidence heads, and reused compatible prepared
  compact-matrix activations without changing model output.
- Preserved compact JSON tool schemas in the qualification prompt; pretty-printing
  previously changed the prompt token sequence and invalidated the reference.

### Changed

- Production inference remains in process and owned by the Java model graph.
  External engines are benchmark-only correctness/performance oracles; planned
  ONNX support will parse and execute supported operations in Java rather than
  embedding ONNX Runtime.

## [0.3.10] - 2026-08-25

### Added

- Added strict mapped CACT ingestion and a dedicated Needle 2 decoder, including
  the embedded tokenizer, mixed CQ2/CQ4 tensors, official first-token math, and
  structured tool-call compatibility.
- Added strict single-file and sharded Safetensors bundle ingestion, format-neutral
  mapped tensor sources, Hugging Face Qwen 2/Qwen2.5 configuration and tokenizer
  loading, and direct BF16 execution through Vectors 0.1.12.
- Added RAG qualification for supported Hugging Face checkpoint directories and a
  pinned Transformers comparator for formats outside llama.cpp and Ollama.

### Fixed

- RAG benchmarks now preserve segmented model prompts through plain Java, Spring AI,
  LangChain4j, and in-process generation instead of flattening trusted template
  controls into ordinary text.

### Changed

- Raised the Vectors baseline to 0.1.12.

## [0.3.9] - 2026-08-22

### Fixed

- Ordered the Spring Boot starter after Boot's metrics and observation
  auto-configuration on both supported Boot generations. Actuator-created
  meter registries now receive the Spring AI chat and embedding handlers, so
  exact local prompt and completion counts reach
  `gen_ai.client.token.usage` without application wiring.

## [0.3.8] - 2026-08-22

### Fixed

- `models-spring-boot-starter` now registers Spring AI's standard chat and
  embedding meter handlers when the application enables metrics. Exact local
  `Usage` values are therefore exported through the
  `gen_ai.client.token.usage` input/output counters without retaining an
  unrelated hosted-provider starter or pinning the application's Spring AI
  version.

## [0.3.7] - 2026-08-21

### Added

- Added exact prompt and completion token counts to the Models generation
  terminal signal while preserving existing `TokenStream` consumers.

### Fixed

- Spring AI blocking and streaming responses now publish local runtime counts
  through `Usage`, and LangChain4j blocking and streaming responses publish the
  same counts through `TokenUsage`.

## [0.3.6] - 2026-08-21

### Added

- `ModelsSpringAiChatModel` now accepts an explicit logical model name for
  Spring AI response metadata, default options, and observations. ModelJars
  applications can report the selected qualified catalog alias even when a
  GGUF file embeds a generic name such as `Gguf Output`; existing constructors
  continue to use the runtime's model name.

## [0.3.5] - 2026-08-21

### Fixed

- The Spring AI and LangChain4j embedding adapters now serialize access to the
  non-thread-safe local backend. Concurrent corpus ingestion and live queries
  can no longer corrupt shared encoder scratch state or fail in rotary-table
  batching.

## [0.3.4] - 2026-08-21

### Fixed

- `ModelsSpringAiChatModel` now emits Spring AI's standard
  `gen_ai.client.operation` observations for blocking and streaming inference,
  with the local runtime identity attached to request and response metadata.
  The Spring Boot starter passes the application's `ObservationRegistry` into
  its auto-configured adapter, so local chat calls no longer disappear from
  Spring AI metrics.
- Corrected the Spring Boot starter and ModelJars versions shown in the module
  README and generated documentation.

## [0.3.3] - 2026-08-20

### Added

- Added schema-constrained tool-call decoding to the Spring AI and LangChain4j
  adapters. Supported finite JSON Schema argument spaces are compiled into token
  constraints, preventing invalid enumerable calls during sampling; unsupported
  or partially constrained schemas continue through the existing tool-call path.
- Added runtime token constraints, generation-confidence signals, prompt
  visibility planning, and embedding-backed tool selection as framework-neutral
  building blocks.
- Added an embedding equivalence gate to `models-bench`, run with
  `embedding-equivalence --model <artifact.gguf>`. It tests that Models produces
  the same vectors as llama.cpp, for eight pinned probes over the same model
  bytes, exiting non-zero when they diverge. The reference vectors are
  committed, so it runs in seconds and needs no local llama.cpp build.

  Agreement is gated at 0.999 cosine, where a correct run measures 0.99950 and
  mean pooling in place of last-token measures 0.66156. Vector length is gated
  separately at 1e-3: cosine is scale-invariant, so a runtime that skips L2
  normalization agrees with a normalized reference at exactly 1.0.

### Changed

- Raised the Vectors baseline to 0.1.9, including Spring observation wiring and
  semantic-cache response metadata.

### Fixed

- `ModelsSpringAiEmbeddingModel` now emits Spring AI's standard
  `gen_ai.client.operation` observations for direct, document, and batch calls.
  The adapter attaches its pinned model identity and backend dimension, so local
  embedding work remains visible when an application replaces a hosted model.

## [0.3.2] - 2026-08-08

### Added

- `models-router` is now published. It was absent from the release allowlist, so
  every version until now built it and shipped it nowhere: the pretrained task
  classifier, the packaged index, `ModelRouter.discoverLocal()` and the catalog
  discovery added in 0.3.1 were all unreachable from a dependency. The catalog
  SPI itself was never affected, since it lives in the published `models-api`.

## [0.3.1] - 2026-08-08

### Added

- Added the `gemma-embedding` encoder architecture, which makes EmbeddingGemma-300M
  runnable on the pure-Java backend. Bidirectional attention inverts the loop
  nesting rather than changing a mask: every position needs every other
  position's key at the same layer, so the sequence is the unit of work and
  there is no KV cache. Verified against llama.cpp at 0.99956 minimum cosine,
  where forcing causal masking measures 0.57266.

- Added a pretrained task classifier for `models-router`, shipped as a
  quantized index inside the jar. 1929 prompts over ten tasks, 0.9019 accuracy
  on a held-out split, 0.65 MB. Training prompts live in
  integrallis/model-router-corpus; what ships here is the derived index.

- Added `ModelCatalogProvider`, a ServiceLoader SPI in `models-api` that lets
  installed models describe themselves so callers need not hand-write price,
  latency and per-task quality. `ModelRouter.discoverLocal()` consumes it.
  Models without a performance profile for the current hardware are estimated
  from measured peers rather than dropped, because local generation is
  memory-bandwidth bound and so `tokensPerSecond * sizeBytes` is roughly fixed
  on one machine.

- Added `AppleFoundationModelsCatalog`, reporting Apple's on-device model when
  the machine has one. Opt-in via `discoverLocal(true)`: it is present because
  of the hardware, so discovering it by default would make identical code route
  differently on a Mac than in production.

- Added `matryoshkaDimensions` to `GgufEmbeddingBackend`, for models trained
  with Matryoshka Representation Learning. Named for the technique rather than
  called `dimensions` so that truncating a model never trained that way is not
  something a caller reaches for by accident.

### Changed

- Encoder positions within a layer now run concurrently, taking a full router
  index build from 1522 s to 300 s. Bit-exact: the parallel build produces a
  `quantized.bin` with the same SHA-256 as the sequential one.

- Raised the vectors dependency to 0.1.7 for quantized-only collections, which
  store the classifier index as 4-bit codes with no full-precision copy.

## [0.3.0] - 2026-08-05

### Added

- Added tool calling. `ToolSpec` and `ToolCall` describe declarations and
  invocations without introducing a JSON dependency, because argument text is
  carried verbatim rather than parsed. `ToolSyntax` records how each model family
  expresses calls, taken from its published chat template, and drives both
  rendering and recovery so no per-family parser is required. `ChatTemplate`
  gained `render(messages, tools)`, `toolSyntax()`, `supportsTools()`, and
  `canParseToolCalls()`.
- Added tool-call recovery through `ToolCallScanner`, which strips markdown code
  fences, accepts either the `arguments` or `parameters` spelling, and degrades
  to plain text rather than failing a turn on malformed output.
- Surfaced tool calls natively in both framework adapters: Spring AI through
  `AssistantMessage.getToolCalls()`, and LangChain4j through
  `AiMessage.toolExecutionRequests()` with `FinishReason.TOOL_EXECUTION`.
- Added embedding support. `EmbeddingBackend` and `Pooling` define the contract,
  `GgufEmbeddingBackend` implements it over the pure-Java forward pass, and both
  the Llama-family and Gemma 4 decoders can now return the final normalized
  hidden state instead of vocabulary logits. Producing an embedding skips the
  vocabulary projection, so it costs less per token than generating one.
- Added `ModelsSpringAiEmbeddingModel` and `ModelsEmbeddingModel`, letting a
  Spring AI or LangChain4j application keep embeddings inside the JVM.
- Added `Tokenizer.tokenId(String)` for resolving a token id from its exact
  vocabulary text, needed because families disagree on the ids behind identical
  tool-call delimiters.

### Changed

- **Breaking:** moved `EmbeddingBackend` from `com.integrallis.models.embedding`
  to `com.integrallis.models.api`. It is a contract, and leaving it in
  `models-embedding` would have forced every backend implementing it to depend on
  `vectors-db`. No published artifact implemented it.
- `ChatMessage` now carries `toolCalls`. Blank text remains invalid except on an
  assistant turn consisting solely of a tool call. The two-argument constructor
  and the factory methods are unchanged.
- Replaced the sampler's full-vocabulary sort with a bounded-heap top-k
  selection, measured at 19.251 ms to 0.848 ms per sampled token at a
  151,936-token vocabulary. Tie-breaking still prefers the lower token id, so
  seeded output is unchanged.

## [0.2.6] - 2026-08-04

### Added

- Added an owning `InferencePipeline` for coordinated access to structured
  tokenization, model metadata, active context capacity and position, prefill,
  forward-pass logits, reset, checkpoint, rewind, and high-level generation.
- Exposed the runtime-allocated context capacity separately from the maximum
  context length declared by model metadata.

## [0.2.5] - 2026-08-03

### Changed

- Made the release qualification gate exercise the public default Gemma 4
  runtime before any benchmark-only tuning and retain nested benchmark failures.

### Fixed

- Fixed singleton Gemma 4 routed-expert prefill projections by copying
  capacity-sized batched buffers through exact-shape reusable scratch arrays.

## [0.2.4] - 2026-08-02

### Fixed

- Fixed Gemma 4 native batched prefill when MoE routing assigns a single token
  to a Q4_K expert projection by falling back to the Java projection kernel.

## [0.2.3] - 2026-08-02

### Added

- Added `GroundedRagPrompt`, a canonical framework-neutral RAG prompt that
  screens retrieved evidence and withholds rejected context from generation.

## [0.2.2] - 2026-08-02

### Changed

- Qualified Gemma 4 26B-A4B Q4_K_M at the usable tier by parallelizing safe
  prefill regions, vectorizing accumulation, reusing projection scratch space,
  precomputing RoPE values, bounding GELU lookup, and eliminating hot native
  allocations.

## [0.2.1] - 2026-08-02

### Added

- Added Gemma 4 decoder support for hybrid attention, shared and routed MoE,
  mapped experts, the role-aware chat template, and pinned 26B-A4B integration
  fixtures across plain Java, LangChain4j, and Spring AI.
- Added native ABI 4 independent projection dispatch and retained 32-token
  Gemma 4 prefill batching.

### Changed

- Recorded Gemma 4 26B-A4B as integration-tested but not production-qualified;
  all quality gates passed, while p95 TTFT exceeded the usable latency ceiling.

## [0.2.0] - 2026-07-29

### Added

- Added role-aware chat messages and qualified templates for ChatML, Zephyr,
  Llama 3, Gemma, Phi-3, DeepSeek, H2O, and MiniCPM model families.
- Bundled the Apple Foundation Models bridge in `backend-apple` for supported
  Apple Silicon systems, with integrity-checked extraction and caching.

### Changed

- Made `AppleFoundationModelsClient` implement the shared
  `TextGenerationModel` contract for plain Java, LangChain4j, and Spring AI
  applications.
- Updated the reusable Vector API kernel dependency to Vectors 0.1.4.
- Updated onboarding to load qualified artifacts through marker-owned
  ModelJars references while retaining direct GGUF loading as an advanced API.

### Fixed

- Added release verification for the packaged Apple bridge and its native
  artifact metadata.
- Disabled Gradle build caching on the Windows ARM native-kernel job where the
  runner does not provide a stable cache environment.

## [0.1.0] - 2026-07-29

### Added

- Maven Central staging and JReleaser release automation based on the proven
  `integrallis/mfcqi-java` pipeline.

### Changed

- Defined an explicit Maven Central publication allowlist for implemented
  runtime, backend, RAG, embedding, and framework modules.
- Replaced the sibling Vectors composite build with the released Vectors
  dependency so the repository builds and publishes independently.
- Split detailed architecture, module, integration, testing, and support
  material out of the project README.

### Fixed

- Reset the mutable KV cache between independent generations and serialize
  calls sharing one backend.
- Replaced collision-prone BPE merge keys based on `String.hashCode()` with
  full merge-pair keys and added byte-level/ranked-merge regressions.
- Removed unused runtime dependencies and made strict Javadocs, dependency
  locks, staged publications, SBOMs, SpotBugs, and 80% published-module
  coverage part of the release gate.
