# Changelog

All notable changes to models are documented here.

## [Unreleased]

## [0.3.41] - 2026-09-17

### Added
- Min-p sampling: `SamplingOptions.minP` (default 0 = disabled, validated to [0, 1]) keeps tokens with p >= minP * p_max on the temperature-scaled distribution, applied before top-p (it commutes with top-k). Skipped entirely at 0, so default sampling is byte-identical. Exposed as `integrallis.models.sampling.min-p` and carried from defaults by the LangChain4j and Spring AI adapters.
- Typed stop reasons: `StopReason` (`EOS`, `STOP_SEQUENCE`, `CONSTRAINT_COMPLETE`, `REPETITION_LOOP`, `CANCELLED`, `MAX_TOKENS`) is delivered through the new `TokenStream.onComplete(GenerationUsage, StopReason)` (its default forwards to `onComplete(GenerationUsage)`) and recorded in `GenerationMetrics.stopReason()` on the sequential, speculative and continuous-batching paths. Model-chosen ends win over truncation on the same token.
- Cancellation: `TokenStream.isCancelled()` (default false) is polled after every emitted token. The LangChain4j streaming model hands handlers a `StreamingHandle` backed by it, and disposing a Spring AI streaming subscription cancels generation.
- Finish reasons in adapters: LangChain4j maps EOS/stop sequence/constraint to `STOP`, the token limit to `LENGTH`, and repetition loops and cancellation to `OTHER` (tool calls keep `TOOL_EXECUTION`); Spring AI generation metadata carries `STOP` / `LENGTH` / `REPETITION_LOOP` / `CANCELLED` plus the exact reason under `stopReason`. Engines that report no reason still produce no finish reason.
- Opt-in repetition-loop detector: `SamplingOptions.repetitionLoopDetection(new RepetitionLoopDetection(maxSpan, minRepeats, minLoopTokens))` stops generation with `REPETITION_LOOP` once the generated tokens end in an exactly periodic run (period <= maxSpan, >= minRepeats copies, >= minLoopTokens tokens). At most maxSpan comparisons per token, independent of generation length. Stops are counted in `GenerationLoop` / `RuntimeTextGenerationModel.repetitionLoopStops()` and `ContinuousBatchingMetrics.repetitionLoopStops()`. Off by default; Spring Boot `integrallis.models.sampling.repetition-loop.*`.
- `Tokenizer.endOfGenerationTokenIds()` exposes the resolved end-of-generation set for diagnostics.

### Fixed
- Hugging Face checkpoints now stop on every `eos_token_id` declared in `config.json` and `generation_config.json` (integer or list), not only the tokenizer's single EOS plus vocabulary heuristics; a list-valued `eos_token_id` in a Qwen 2 or GPT-OSS `config.json` no longer fails to parse.
- Named performance cliffs: `PerformanceCliff` names the reasons a fast path was not taken, and each is reported from the branch that takes the slower path, at most once per process, as a `com.integrallis.models.PerformanceCliff` JFR event (category `Models`, with stack trace) and as `performance-cliffs` / `performance-cliff.<id>` entries in `PureJavaBackend`, `SopranoBackend` and `RustFfmBackend` diagnostics. Reasons: `vector-api-unavailable`, `vector-width-capped`, `persistent-executor-not-used`, `q4-pairwise-kernel-unsupported`, `batched-prefill-unsupported-tensor-type`, `row-by-row-projection`, `fused-grouped-attention-not-wired`, `native-grouped-attention-unavailable`, `native-gated-delta-net-unavailable`, `native-poll-budget-unsupported`. Diagnostics are unchanged when no cliff has been reported. Provocation and observed event counts per reason: `benchmark-results/2026-09-16-loader-cliffs/README.md`.
- Structural test that fails when a Vector API `VectorSpecies` is a method, constructor or lambda parameter, or a field that is not `static final`. backend-java has no violations; the pinned vectors-core 0.1.22 has one (`PanamaConstants#preferredSpecies`, called only from a static initializer), recorded in an explicit allow-list as a vectors follow-up.

### Fixed
- GGUF and Safetensors loaders assert every tensor's size: the byte length implied by the type's block layout (or dtype width) and the shape must fit between the tensor's start and the next tensor in offset order, or the end of the file. Short, overlapping and overrunning regions now fail with the tensor name, the expected byte count and the bytes available. Previously a GGUF tensor whose range overlapped the next tensor but stayed inside the file was accepted and read its neighbour's bytes. All 36 GGUF and 20 Safetensors files on the development host still parse.

## [0.3.40] - 2026-09-16

### Changed
- Native worker pool: workers poll for the next dispatch for a time budget before they park (`models.native.kernels.pollMillis`, default 5 ms; reported as `native-kernel-poll-millis`). Parking after 4,000 spin rounds cost a third of decode on a shared 16-vCPU host (15.6-16.6 to 24.0-26.0 tok/s, 4.3x fewer context switches) and about 2% on dedicated cores (c7a.4xlarge, 25.5 to 26.1); 5 ms keeps the whole gain and a fifth of the idle tail of 25 ms. The chunk-stealing experiment was measured on dedicated cores (-10% decode, -25% prefill) and removed.
- Native backend: the vectors persistent executor, which the Java side still drives for its parallel ops, is parked when the native kernel pool opens (vectors 0.1.22, `VectorUtil.setGgufPollMillis(0)`; reported as `java-executor-poll-millis`). With both pools polling, prefill on the native backend fell from 126 to 40 tok/s and context switches rose from 0.2 M to 24 M per run; parked, it matches the previous release exactly.
- vectors 0.1.22: the pure-Java backend's executor polls at every stage barrier before parking, measured +45% decode on dedicated cores (6.4 to 9.4 tok/s) and up to +80% on shared vCPUs.

## [0.3.39] - 2026-09-16

### Added
- Qwen3 8B Q4_K_M qualification through the owned Rust/FFM kernel: an independent pinned llama.cpp greedy-token oracle through the FFM kernel, a repeatable named native fixture task, and tool-conformance CLI support for the `rust-ffm` backend; 14/14 tool suite plus the conversational tool-result follow-up pass on the pinned artifact, with the matched controlled-host benchmark recording p95 TTFT 13.98 s to 1.83 s. No external inference runtime is introduced. (#177)
- Java-native support for Apache-2.0/MIT-licensed local embedding artifacts: Nomic Embed Text v1.5 F16 and BGE Small EN v1.5 F16, qualified against pinned llama.cpp a582222 oracles across eight probes (minimum cosine 0.9999999 and 0.9999988); the Nomic Q4_K_M sibling stays excluded at 0.9957 against the 0.999 admission gate. (#178)

## [0.3.38] - 2026-09-16

### Fixed
- Native worker pool: a worker counted out of one generation that had not yet reached the idle wait when the active worker count grew again treated the finished generation as new, ran it against the grown count, and decremented the completion counter of the generation published after it, so the caller could return before every partition of the next job was written (rows of a projection left unwritten). Reproduced on two pinned CPUs at about 6% of runs with the pool's own toggling test; the published job word now carries the partition count the job was published with, and workers judge their membership against that count. 0/400 failures after the fix on the same pinned loop.
- Activated LoRA on Llama-family GGUF graphs: llama.cpp's converter stores the query and key projection rows in the interleaved (adjacent-pair) rotary layout, but the adapter loader added the `q_proj` / `k_proj` low-rank updates in Hugging Face row order, so on Granite (and any other non-NeoX graph) those two updates landed on the wrong rows within each head while the value, output and FFN updates were right. `ActivatedLoraAdapter.Architecture` now carries the head counts and an `interleavedRotaryRows` flag, and the loader permutes the query and key `lora_B` rows into the GGUF layout at load time. Found with IBM's PEFT reference on the dequantised Q4_K_M weights: on a SQuAD case whose base answer is "kick back", the runtime produced that span instead of the label; it now produces the reference's `"answerable"` and its hidden states track the reference to within the activation-quantisation regime. Qwen adapters (NeoX layout) were never affected.
- Fused grouped-query attention is Granite-only; every other architecture keeps the head-by-head attention loop its pinned greedy oracles were recorded on (the fused kernel's lane reductions and vector exponential flipped a near-tie Qwen3 0.6B token on the Rust arm).

### Added
- `granite` RAG prompt template in `models-rag-bench` so Granite 4.x artifacts can run the controlled RAG qualification harness with the byte-exact Transformers envelope.
- `granite-documents` RAG prompt template: evidence in the Granite 4.x `<documents>` system block, bare question as the user turn, byte-exact with the Transformers chat template; `GraniteDocumentsPrompt` moved to `models-runtime` (public, Jackson-free, Jackson-parity tested) so both bench modules share one renderer.

- Added the Granite decoder graph (embedding, attention, residual, and logit scalars) and the
  activated-adapter runtime: a fail-closed Safetensors activated-LoRA loader, exact activation
  boundaries at the adapter's invocation tokens, physically shared immutable KV prefixes between
  the base and activated branches, `ActivatedToolCallingModel` with shared and recomputed prefix
  strategies, Rust/FFM activated loaders, and Spring AI and LangChain4j tool loops over one base
  cache lineage. The runtime is a capability; it qualifies no adapter by itself.
- Routed single-session prompt prefill through the ragged session-batch path, so shared-prefix
  preparation and every `TextGenerationSession` prompt batch their matrix products. Identity was
  proven on a nano model, on the pinned Granite 4.1 3B GGUF against llama.cpp b9960, and by the
  full backend-java suite; the activated path's per-case time fell by roughly 7x on the Rust arm.
- Added the answerability qualification runners for an upstream RAG specialist: the frozen
  two-dataset window runner with a Transformers prompt-byte oracle, the 4,096-token long-context
  retention gate, a Granite documents template for the prefix-sharing crossover, the fail-closed
  upstream adapter packager, and the ModelJars component report assembler.

### Changed
- Attention in the Llama-family forward pass is partitioned over the GGUF worker pool: single-token decode over query heads, batched and independent-session prefill over batch rows. Same arithmetic and order; Granite 4.1 3B (40 heads of 64) had been bounded by the serial loops. Granite also uses the vector swiGlu and a vector FMA for its residual multiplier.
- Fused grouped-query attention: `GroupedQueryAttentionKernel` scores and accumulates every query head of a KV group in one pass over the cached keys and values, bit-identical to the head-by-head vectors-core kernels; the forward pass attends per KV-head group with one thread-local score buffer per head.
- `models.native.kernels.decodeThreads` limits the native workers that take rows on single-token projections while batched prefill keeps the whole pool (new additive kernel export `jmodels_kernels_context_set_active_threads`, capability bit 19, ABI unchanged).
- Native grouped-query attention for single-token decode on Granite: the Rust kernel computes one query row over up to two cached key/value spans through a zero-copy critical downcall (`jmodels_grouped_attention_f32_with_context`, capability bit 20), partitioned over KV heads on the native pool; `models.native.groupedAttention=false` disables it. Idle native workers now sleep on a separate condition so a smaller decode partition wakes only the workers that take rows. Granite's rmsNorm scale loop and swiGlu are vectorised with tier-stable arithmetic.

### Fixed

- Mapped the `dbrx` GGUF pre-tokenizer to the Llama-3 split pattern with ordinary merge ranking;
  an unmapped name previously skipped pre-tokenization entirely for Granite 4.x. Every BPE
  pre-tokenizer pattern now treats U+00A0 as whitespace, as the published Rust regexes and
  llama.cpp do. All 620 rendered Granite 4.1 documents prompts became byte- and token-identical
  to Transformers 4.57.1.
- Rendered Granite's `<documents>` and `</documents>` template markers as control tokens; they
  are special tokens in Granite 4.1 and had been placed inside a text segment.

### Experiment

- Opened the Granite 4.1 3B answerability hybrid candidate: the catalog base plus IBM's upstream
  Apache-2.0 Granite RAG Library activated adapter, sharing the base prefix by physical array
  identity. Gates 1 and 2 passed on real weights; the identity precondition and the 310-case
  window are running on a bounded host. No hybrid is qualified by this release.

### Corrected

- Withdrew the projected Qwen router's hybrid qualification. Its measured latency and correctness
  remain useful semantic-routing evidence, but it shares neither KV tensors nor model state and
  therefore does not satisfy the cache-sharing composition objective. The retained report was
  also absent from the Models revision cited by the downstream evidence URL. No hybrid recipe is
  currently qualified.

## [0.3.37] - 2026-09-11

### Added

- Added per-member virtual-conversation context projection, including full-history, current-turn,
  and tool-result-to-user policies. Physical members still retain independent exact prompt/KV
  state; the projection controls only the semantic messages rendered for that member.

### Experiment

- Measured Qwen3 0.6B Q4_0 chat plus Qwen3 1.7B Q8_0 tool selection as a projected virtual model.
  All 36 turns passed across six counterbalanced fresh JVMs on a dedicated eight-vCPU host. Median
  end-to-end time improved from 53.166 seconds to 35.151 seconds (33.88%), while median peak RSS
  increased from 2,404,032 KiB to 3,270,444 KiB because both models remain resident. This result
  was subsequently withdrawn as a hybrid qualification because it did not transfer cache state.

## [0.3.36] - 2026-09-08

### Changed

- Flattened compatible prompt positions from independent Llama-family sessions into physical
  projection batches while preserving each session's causal attention boundary and KV cache.
  Pinned MiniCPM5 and Qwen3 profiles improved median aggregate prompt throughput by 5.81% and
  14.12%, respectively, with matching output hashes and no increase in median peak RSS.

## [0.3.35] - 2026-09-08

### Added

- Added the Hammer 2.1 tool protocol and a safe parser for its mixed JSON, Python-literal, and
  tagged tool-call output. Qualification evidence is retained, but no Hammer model is promoted.
- Added a reproducible virtual-model qualification harness that verifies conversation continuity,
  tool selection, opaque tool-result handling, exact recall, latency, and peak memory across fresh
  JVM processes.

### Fixed

- Applied GGUF YaRN scaling from the checkpoint metadata, restoring exact long-context inference
  for architectures that declare a scaled original context length.
- Made the Pages artifact name unique per workflow attempt so a failed documentation deployment can
  be retried without creating an ambiguous duplicate artifact.

### Qualification

- Rejected the Qwen3 0.6B chat + Qwen3 1.7B tool composite after it passed all 36 correctness turns
  but was 20.55% slower at median end-to-end latency than the single-model control on an isolated
  8-vCPU host. The runtime APIs remain available; no preconfigured hybrid artifact is published.

## [0.3.34] - 2026-09-07

### Added

- Added opt-in ragged prompt prefill across independent generation sessions. Supported backends
  share physical transformer passes while retaining each session's position, KV cache, final
  logits, and continuation state.
- Added an explicit backend capability signal and fail-fast scheduler validation, preventing an
  unsupported backend from silently turning the requested optimization into sequential work.

### Fixed

- Preserved the complete independent-session batching contract through the Rust/FFM backend
  wrapper selected by qualified ModelJars runtimes.

### Documentation

- Documented the model- and machine-specific qualification boundary, including exact-continuation
  MiniCPM and Qwen counterexamples and the measured latency-versus-memory tradeoff.

## [0.3.33] - 2026-09-07

### Added

- Added opt-in continuous batching for high-level generation sessions. Concurrent requests share
  one physical model step while retaining independent prompt-prefix and KV state, token constraints,
  streaming callbacks, and per-request generation metrics.
- Added bounded admission, a configurable idle batch-formation window, and scheduler-level batch
  utilization metrics.
- Preserved the backend prefill contract in bounded scheduling chunks and required an explicit
  batch size, since controlled MiniCPM and Qwen profiles prove that capacity alone does not predict
  a throughput gain.

### Documentation

- Documented how virtual conversations can use one continuous scheduler per physical member and
  why batches never cross model identities.

## [0.3.32] - 2026-09-07

### Added

- Added `VirtualChatModel`, which presents capability-specific local models as one role-aware
  conversation while retaining a separate generation session, chat template, and exact prompt/KV
  prefix for each physical member.
- Added prefill-only generation-session preparation and optional background catch-up for inactive
  virtual-model members, with cache and phase measurements.
- Added `VirtualChatRouter`, connecting virtual conversations to the adaptive router's task
  classifier, policies, session affinity, cache evidence, and runtime feedback.
- Added canonical initial history when opening a virtual session, allowing system instructions and
  prior messages to be installed without generating an artificial turn.

### Documentation

- Documented virtual-model construction, routing, tool turns, switch telemetry, background-prefill
  resource tradeoffs, and the boundary between shared semantic context and model-specific KV.

## [0.3.31] - 2026-09-06

### Added

- Added complete pure-Java DeBERTa-v2 cross-encoder inference from multi-file Safetensors
  checkpoints, including the Hugging Face Unigram tokenizer, precompiled normalization,
  disentangled relative attention, ContextPooler, and classification head.
- Added real-checkpoint plain Java, LangChain4j, and Spring AI reranking gates for the pinned
  `mixedbread-ai/mxbai-rerank-xsmall-v1` artifact.

### Changed

- Parallelized independent DeBERTa token rows while retaining deterministic scores and a
  deployment override through `models.deberta.threads`.
- Expanded immutable F16 weights once into the existing F32 execution kernels after direct mapped
  F16 experiments showed a material throughput regression.

### Documentation

- Documented Safetensors reranking, artifact preparation, framework usage, the six-pair
  Transformers oracle, and controlled cold-load and throughput evidence.

## [0.3.30] - 2026-09-06

### Added

- Added explicit high-level text-generation sessions over batch-capable backends. Each session
  keeps an independent context, exact prompt-prefix history, and generation metrics while sharing
  one loaded model.
- Added standard GGUF BERT rank pooling and `cls.*` / `cls.output.*` sequence-classifier tensor
  aliases while preserving compatibility with previously corrected reranker artifacts.
- Added checksum-bound plain Java, LangChain4j, and Spring AI gates for a corrected TinyBERT L2
  Q8_0 artifact plus a local-artifact performance experiment.

### Documentation

- Documented conversation-scoped generation sessions and corrected the composition research plan:
  KV state is isolated by model and adapter identity unless equality of the produced K/V
  activations is demonstrated.
- Recorded the rejected incomplete TinyBERT community conversions, corrected conversion hashes,
  Transformers equivalence, and the measured low-latency pure-Java reranking tier.

## [0.3.29] - 2026-09-05

### Added

- Added a framework-neutral text-to-speech API with immutable PCM audio, incremental audio
  callbacks, speech controls, and PCM16 WAVE encoding.
- Added complete in-process execution of standalone Soprano 1.1 GGUF artifacts: byte-level
  tokenization, autoregressive acoustic generation, vocoder, inverse STFT, and 32 kHz mono output.
- Added true streaming synthesis whose emitted chunks reproduce the blocking waveform exactly.

### Changed

- Prepared Hugging Face-named BF16 language-model projections as owned F32 execution matrices and
  reused the Vectors prepared-F32 abstraction for the Soprano vocoder.
- Added a narrowly scoped Models-owned Rust/FFM Q8 projection kernel for qualified CPU profiles;
  model parsing, graph execution, state, sampling, vocoder, streaming, and audio remain Java-owned.
- Batched 12 stable acoustic frames per streaming vocoder call and bounded Soprano's native worker
  recommendation at six, while preserving explicit deployment overrides.

### Fixed

- Applied repetition penalties once per unique prior token, matching the reference sampler and
  preventing duplicated history from multiplying the penalty.
- Preserved per-block accumulation order in the native Q8 kernel so it matches the Java projection
  and deterministic Soprano waveform.
- Removed repeated overlapping one-token vocoder work from incremental synthesis; the final Q8 and
  pure-Java BF16 profiles now pass the production streaming-latency policy.

### Documentation

- Added text-to-speech API guidance, immutable artifact and oracle evidence, controlled streaming
  latency gates, and the measured packed-dot-product request for JVM implementors.

## [0.3.28] - 2026-09-04

### Fixed

- Detect C1-only and interpreted-only JVM launches before pure-Java model loading, avoiding an
  apparent inference hang under Spring Boot `bootRun` and IntelliJ's default Spring Boot launch
  optimization.
- Added an exact Spring context regression for the `ChatClient` builder-customizer, conversation
  memory, typed tool, and Qwen tool-result continuation path.

### Documentation

- Documented the Gradle, Maven, and IntelliJ settings that retain the optimizing C2 compiler for
  in-process inference during development.

## [0.3.27] - 2026-09-03

### Added

- Added adaptive model routing across mixed local and hosted fleets, with hard capability filters,
  composable scorers, live availability and queue signals, EWMA outcome feedback, failure cooldowns,
  and bounded session affinity.
- Added a provider-neutral `ModelFleet` execution layer. Applications supply their own local or
  hosted clients, so the router does not introduce a provider SDK or external inference runtime.
- Added Spring AI and LangChain4j routed chat adapters for blocking and streaming requests. Both
  preserve the framework-native request, and streaming fallback is limited to failures before the
  first response is emitted.

### Fixed

- Made the Qwen 3.5 rewind/reset regression test portable across equivalent SIMD reduction orders.

### Documentation

- Added a routing guide covering fleet construction, live feedback, session affinity, failover,
  framework adapters, and why KV or prefix state is never shared across different models.

## [0.3.26] - 2026-09-03

### Added

- Added family-neutral tool-calling templates and parsing for Qwen/Hermes, SmolLM3, MiniCPM,
  Llama, and Needle protocols, with schema-aware argument validation and control-token isolation.
- Added a retained 14-scenario qualification harness covering tool selection, typed arguments,
  crowded tool sets, refusal, parallel calls where supported, and tool-result continuation.
- Added real-weight Spring AI and LangChain4j round-trip gates for Qwen3 1.7B, including typed Java
  tool execution followed by a grounded conversational answer without a custom renderer.

### Documentation

- Published checksum-pinned evidence for six small tool models. Qwen3 1.7B and Needle 2 passed;
  Qwen3 0.6B, MiniCPM5 1B, SmolLM3 3B, and Llama 3.2 3B remain unqualified.

## [0.3.25] - 2026-09-02

### Added

- Added pure-Java execution for Meta MobileMoE-S QAT Safetensors checkpoints, including the
  official tokenizer and chat template, top-k routed and shared experts, packed signed-INT4 G32
  weights, batched prefill, and grouped projections.
- Added a prepared Q8 runtime layout that keeps the source checkpoint mapped while reusing the
  existing Java Vector API Q8 kernels. The direct packed-INT4 layout remains available with
  `-Dmodels.mobilemoe.runtimeLayout=packed-int4` for lower-memory environments.
- Added official-checkpoint tokenizer, full-logit, KV-state, layout, public-backend, and
  default-configuration integration gates.

### Documentation

- Recorded the complete MobileMoE optimization history, including the rejected BF16 expansion,
  retained packed-INT4 and prepared-Q8 JMH measurements, immutable model identity, 27-request
  production qualification, and the resulting scaled signed-INT4 JVM request.

## [0.3.24] - 2026-09-02

### Added

- Added typed, fluent Spring AI tool-result renderers for Needle 2. Spring's serialized callback
  result is converted back to the registered Java type inside the adapter before application code
  renders the assistant response; raw serialized results now require an explicit opt-in.

### Fixed

- Prevented Needle 2's private reasoning, empty tool array, and end-of-turn markers from leaking
  into `ChatClient.content()` when no declared tool applies.

### Documentation

- Updated the Spring AI tool example to return a human-facing weather answer from a typed Java
  result and documented explicit raw-result and no-applicable-tool behavior.

## [0.3.23] - 2026-09-01

### Added

- Added a framework-neutral reranking API and pure-Java BERT cross-encoder execution with
  WordPiece sentence pairs, token-type embeddings, exact-GELU feed-forward layers, and corrected
  dense-tanh classification heads.
- Added LangChain4j `ScoringModel` and Spring AI `DocumentPostProcessor` adapters with pinned
  real-model integration tests.
- Added a checksum-bound MS MARCO MiniLM L6 v2 Q4_K imatrix equivalence gate against ONNX and the
  same artifact executed by an independent oracle.
- Added a three-process cold-load and warm pair/batch latency experiment for the six-document
  second-stage reranking envelope.

### Documentation

- Documented reranking usage, framework integration, the rejected uncorrected conversion, retained
  numerical evidence, and the missing standard scalar/vector `erf` as a measurable JVM request.

## [0.3.22] - 2026-08-31

### Added

- Added pure-Java T5/SentencePiece unigram tokenization for GGUF artifacts.
- Qualified Granite Embedding 107M Multilingual Q4_K_M on the existing bidirectional BERT encoder,
  with pinned English, accented Latin, and Japanese tokenization plus a 0.999746 minimum
  llama.cpp embedding cosine.

### Documentation

- Documented Granite multilingual embeddings, unigram tokenization, pooling, and equivalence
  evidence.

## [0.3.21] - 2026-08-31

### Fixed

- Prevented Needle 2 protocol markers and JSON closing delimiters from being consumed inside
  generated string arguments.
- Made Needle 2 Spring AI action selection deterministic and verified the reported hyphenated
  weather tool end to end with the exact `88252` argument and serialized Java result.

### Changed

- Added the Spring AI zipcode case as a mandatory CACT qualification regression and versioned the
  policy so earlier evidence cannot qualify the corrected runtime.

### Documentation

- Documented deterministic Needle action selection and the exact real-model Spring AI release
  gate.

## [0.3.20] - 2026-08-30

### Added

- Added Java Vector API kernels for SwiGLU, Qwen3.5 gate transforms, and fused causal depthwise
  convolution with SiLU.
- Added a Models-owned ABI 5 Gated DeltaNet recurrence kernel behind the exact-profile
  `models.native.gatedDeltaNet` switch; model graph and session state ownership remain in Java.
- Added JMH coverage for the fused Java kernels and the Java/native recurrence paths.

### Changed

- Qualified Qwen3.5 0.8B Q4_K_M and Qwen2.5 3B Instruct Q4_K_M against Ollama under the unchanged
  production RAG policy, retaining all request-level evidence and performance profiles.
- Updated to Vectors 0.1.18 for the fused Java/Panama inference kernels.

### Documentation

- Documented the native recurrence boundary, deployment switches, qualified Qwen results, and
  exact retained evidence.

## [0.3.19] - 2026-08-30

### Added

- Added pure-Java BERT-family embedding execution with WordPiece tokenization, learned position and
  token-type embeddings, bidirectional attention, GELU feed-forward layers, and configurable
  pooling.
- Added pinned All-MiniLM-L6-v2 Q4_K_M qualification against llama.cpp, including exact-token and
  eight-probe vector equivalence gates.
- Added real-model Spring AI and LangChain4j embedding adapter integration coverage for MiniLM.

### Documentation

- Documented the BERT embedding architecture, supported metadata contract, and qualified MiniLM
  runtime path.

## [0.3.18] - 2026-08-30

### Added

- Added Java-native loading and execution for official GPT-OSS Hugging Face checkpoints, including
  MXFP4 routed experts, attention sinks, YaRN rotary factors, and the GPT-OSS tokenizer path.
- Added first-class GPT-OSS Harmony tool calling with the current recipient/channel wire format and
  verified Spring AI and LangChain4j adapter coverage.
- Added pinned official-checkpoint generation, full-logit oracle, and isolated prefill qualification
  gates for GPT-OSS 20B.

### Changed

- Updated to Vectors 0.1.17 for the measured MXFP4 projection hot-loop improvement.
- Documented GPT-OSS as a compatibility implementation rather than a qualified catalog model until
  its measured Java-native throughput meets the publication gate.

## [0.3.17] - 2026-08-30

### Changed

- Added execution-planned, batched prefill for dense Qwen3.5 GGUF models while preserving exact
  session, rewind, continuation, and pinned llama.cpp token results.
- Reworked Gated DeltaNet recurrence over contiguous Vector API row operations and kept all
  recurrence workspaces session-owned across prefill and decode.
- Planned Qwen3.5 from its loaded tensor topology, including auxiliary Gated DeltaNet projections,
  rather than assuming a Llama-shaped mapped graph.
- Updated to Vectors 0.1.15 for the exact Q5_K two-query tail optimization.

## [0.3.16] - 2026-08-29

### Added

- Added pure-Java execution for dense, text-only Qwen3.5 GGUF models, including their hybrid
  full-attention/Gated DeltaNet graph and grouped value-head layout.
- Added pinned 0.8B and 4B Q4_K_M compatibility gates against llama.cpp token oracles and
  FreeToken recurrence fixtures.

### Changed

- Reused Qwen3.5 decode state and Gated DeltaNet workspaces across tokens, reducing sampled
  transient `float[]` allocations without changing deterministic output.
- Documented the Qwen3.5 support boundary and the measured FreeToken compatibility audit.

## [0.3.15] - 2026-08-29

### Added

- Added the optional `backend-tornado` module. It compiles Models-owned Java Q4_0 projection
  kernels for capacity-qualified NVIDIA GPUs, prepares reusable prefill and decode plans during
  readiness, and falls back to the Vector API when acceleration is unavailable.
- Added standard `ServiceLoader` discovery through `PureJavaBackend.loadAutomatic(...)`, allowing
  ModelJars and direct Models applications to select an optional accelerator without coupling the
  core Java backend to its implementation.

### Changed

- Removed redundant Needle 2 vocabulary projections during prefill while preserving exact logits.
- Completed the Spring AI Needle 2 tool-result follow-up path for blocking and streaming
  `ChatClient` calls, including typed `@Tool` results registered through `defaultTools(...)`.

## [0.3.14] - 2026-08-28

### Fixed

- Preserved Spring AI tool declarations by advertising `ToolCallingChatOptions`, and executed the
  callback/follow-up loop on both blocking and streaming `ChatClient` paths across Spring AI 1.1
  and 2.0.
- Added artifact-capability validation so ModelJars callers receive a clear error when attempting
  tool calling with a model that was not qualified for it.

### Documentation

- Documented Spring AI tool execution, downloadable-model memory and device behavior, and the
  current Apple Foundation Models tool-calling limitation.

## [0.3.13] - 2026-08-28

### Added

- Exposed runtime-owned generation measurements through `InferencePipeline` and
  `RuntimeTextGenerationModel`, including tokenization, prompt preparation, prefill, time to first
  token, decode, total duration, token usage, prompt-cache reuse, and decode throughput.

## [0.3.12] - 2026-08-28

### Fixed

- Made mapped model and Gemma 4 expert-cache memory compatible with GraalVM Native Image while
  retaining deterministic shared-arena cleanup on the JVM. Native executables use automatically
  managed, cross-thread arenas so Vector API support remains available during in-process inference.

## [0.3.11] - 2026-08-27

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
  the upstream playground cases and immutable artifact/source evidence. Reports
  include the JVM maximum heap alongside processor and physical-memory evidence.

### Fixed

- Kept the `com.integrallis:models` facade dependency-free beyond Models API/runtime
  by replacing the new tool-schema Jackson dependency with a bounded in-process
  Java parser.
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
- Raised the Vectors baseline to 0.1.13 for reusable prepared rotated-codebook
  activations in the Java CACT path.

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
