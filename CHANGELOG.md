# Changelog

All notable changes to models are documented here.

## [Unreleased]

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
