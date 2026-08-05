# Changelog

All notable changes to models are documented here.

## [Unreleased]

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
