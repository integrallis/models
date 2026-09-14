# Model intake, portable formats, and cache-composition research

## Decision summary

ModelJars should expand from GGUF in two directions at once:

1. **Use SafeTensors Hugging Face repositories as the primary upstream intake format.** This is
   how Models gains faithful access to original weights, architecture configuration, tokenizer,
   chat template, generation defaults, adapter metadata, and shard indexes.
2. **Keep the shipped execution stack owned and closed.** An imported package becomes a Models
   artifact only after its architecture runs through an owned Java path or a deliberately narrow,
   owned platform bridge. It must not become a wrapper around an inference server, an arbitrary
   Python runtime, or a pluggable third-party backend.

The immediate catalog-expansion priority is not every good public model. It is the intersection of
an open or acceptable license, an architecture Models can execute or implement next, a useful
specialized workload, reproducible upstream files, and an evaluation that matches that workload.

The hybrid work establishes a useful boundary: **physical KV-prefix sharing is solved mechanically
for an activated adapter over the same base**, but Qwen3 1.7B tool-routing quality is not release
quality. A hybrid release must wait for a new independently trained candidate. It must not reuse
or retune against the exposed V20/V22 decision screens.

## Non-negotiable product constraints

| Constraint | Product consequence |
| --- | --- |
| Java-native is the reference path | Every new architecture begins with a Java implementation and deterministic oracle tests. |
| Rust is a narrow temporary crutch | A Rust/FFM component may accelerate a demonstrated hotspot, never own the graph, tokenizer, sampling, cache state, model selection, or generation lifecycle. |
| No external inference server | Ollama, llama.cpp server, vLLM, SGLang, Python, and Docker may be oracle or benchmark peers only. They are never prerequisites for a ModelJars application. |
| No user-pluggable backend SPI | Models owns a sealed set of execution paths. A platform bridge is selected internally from a descriptor and capability policy, not supplied by an application. |
| Provenance before convenience | Every artifact pins upstream revision, every downloaded file hash, source license/notice, importer version, architecture mapping, tokenizer/template hashes, and qualification report. |
| Sensitive data stays local when declared | A request can require an in-process model; a future policy/redaction layer will make that decision from audited data classification rather than pretending the router can infer PII from arbitrary text. |

## Candidate assessment

### Immediate qualification queue

These are the candidates that merit a bounded experiment first. “Ready” means ready for an
evidence-producing intake/compatibility experiment, not that it is already qualified for release.

| Priority | Candidate | Why it matters | Current obstacle | First bounded experiment |
| --- | --- | --- | --- | --- |
| P0 | **Qwen3 8B Q4_K_M** | Best same-family successor for the rejected 1.7B hybrid; Apache-2.0, native tool format, 128K class context, and an existing Qwen3 execution path. | Memory/latency and tool quality on the owned Java decoder; a fresh specialist adapter must be trained from new audited data. | Base Java conformance + tool/no-tool screen; only then train a new activated adapter and freeze a disjoint hybrid screen. |
| P0 | **SmolLM3 3B Q4_K_M** | A small Apache-2.0 general model with documented tool calling and explicit think/no-think control. It is already generally qualified in the current catalog, which lowers runtime risk. | Tool behavior needs its own template, parser, and end-to-end Spring/LC4J qualification; chat qualification is not tool qualification. | Deterministic tool/no-tool and multi-turn result-consumption suite with the native template. |
| P0 | **LFM2-1.2B-RAG GGUF** | A deliberately small RAG-oriented model: a strong counterpoint to generic chat models if it proves grounded-answer quality and low latency. | LFM2 architecture/tokenizer/template compatibility and Liquid’s LFM license must pass audit. It is not a proxy for embedding quality. | Parser/load conformance, source-template reproduction, then a compact RAG truthfulness/retrieval-answer screen. |
| P0 | **S1-mini (Superwhisper)** | A 0.6B Qwen3-derived, English ASR text-normalizer. It gives the audio catalog an immediately useful, CPU-feasible task without first solving waveform generation. | Exact required system prompt, control line, and disabled-thinking template are part of correctness; it is not a general chat or TTS model. The naming clause must be retained. | Java template conformance plus a held-out transcript-normalization set, greedy decoding, and empty-output handling. |
| P1 | **Phi-4-mini-instruct** | MIT-licensed 3.8B reasoning/instruction model with a documented tool prompt format and an official int4 ONNX release. It is the strongest first ONNX decoder candidate. | Phi architecture and long-context decoder support; ONNX generation requires cache I/O and sampling support, not merely `session.run`. | First prove the SafeTensors Java loader; independently assess the ONNX graph package and Java owned cache/sampling integration. |
| P1 | **IBM Granite 4.1 8B** | Apache-2.0, enterprise-friendly licensing and a useful business/code/RAG profile. | Granite architecture correctness, template/tool behavior, and real Java performance have not been evidenced by its catalog listing. | GGUF baseline conformance, then SafeTensors package import and an enterprise RAG/tool screen. |
| P1 | **Gemma 4 E4B IT** | A relatively small multimodal, Apache-2.0 model with native function-calling claims; a good bridge to text/audio/image work. | E4B is not “Gemma 4 text only”: multimodal tokenization, audio/vision processors, and QAT quant behavior are separate correctness work. | Text-only parity first; then independently qualify modality processors. Do not claim unified multimodal support from text success. |
| P1 | **EuroLLM 1.7B** | Already qualified for general multilingual use in the current ModelJars catalog; useful local router/chat candidate for European languages. | No tool or domain qualification yet; tiny model capacity must be measured rather than assumed. | Multilingual no-tool/call control and short grounded RAG evaluation. |
| P1 | **Qwen3 0.6B + S1-mini adapter family** | Same base family makes activated-adapter cache sharing technically possible and the task is narrow enough for a credible hybrid demonstration. | S1-mini is a transcript normalizer, not a general tool expert; the product should expose it as a pipeline stage, not pretend it is a general virtual model. | Physical cache sharing for a fixed shared system prefix, paired with normalization accuracy and a base-chat isolation test. |

### Valuable, but not first in the line

| Candidate | Assessment | Recommendation |
| --- | --- | --- |
| **DeepSeek-R1-Distill-Qwen-7B** | MIT and useful for reasoning. Distilled reasoning output can be verbose and is not evidence of reliable tool selection. Its Qwen2 lineage means a different runtime family from Qwen3. | Qualify as a reasoning route after Qwen3 8B or Phi; use structured reasoning/answer tests, not chain-of-thought extraction. |
| **Llama 3.1 8B Instruct** | Mature architecture, broad ecosystem, and useful baseline; the license is Llama 3.1, not Apache/MIT. | Keep as a comparison and general-chat qualification candidate; license review is a release gate. |
| **Ministral-3-3B-Instruct-2512** | Attractive compact multimodal/tool candidate with very long context. | Defer until the Mistral-3 multimodal package and vision processor are supported. Do not use a text-only GGUF result to imply full model support. |
| **Gemma-3n-E2B-IT** | Mobile-oriented multimodal model with selective activation and a Gemma license. | A later Apple/Android multimodal track. Its encoders and modality token budget make it unsuitable as a quick text-only catalog expansion. |
| **Fin-R1 7B** | Domain-specialized finance can tell a compelling SLM story, but financial factuality and compliance errors are costly. | Candidate only with a curated, timestamped finance RAG benchmark and an explicit “not financial advice” product boundary. |
| **HuatuoGPT-o1 7B** | Medical specialization is interesting, but its intended-domain risks require a much higher safety bar. | Research-only until source/license, clinical evidence, safety policy, and medical evaluation are independently audited. Not a casual catalog promotion. |
| **SQLCoder-7B-2 Q5_K_M** | Strong focused SQL use case and a familiar Llama-family decoder. CC-BY-SA-4.0 has redistribution obligations. | Qualify with SQL execution correctness and safety controls (read-only/default sandbox); surface its license conspicuously. |
| **Dolphin variants** | “Dolphin” is a family of community fine-tunes, not a single stable provenance or safety profile. | Do not intake by brand. Admit a specific immutable upstream revision only after license, data provenance, template, safety behavior, and task evaluation are known. |
| **Qwen2.5-Coder 3B / 7B** | Strong practical coding candidates and existing Qwen2 support. | Continue code qualification; 7B belongs on a larger-model hardware tier. Use repository-level tests, not code-completion anecdotes. |
| **DeepSeek-Coder 6.7B** | Useful historical coding baseline. | Keep as a comparison candidate; prioritize newer Qwen coder variants unless it wins an owned benchmark. |
| **TinyLlama, SmolLM2 360M, MiniCPM5 1B** | Excellent footprint demonstrations but limited headroom for reliable tool use and complex RAG. | Keep for cost/latency tiers, classification, and narrow transforms; do not overmarket them as general agents. |

### Desert Ant Labs: what to learn, what not to import blindly

Desert Ant’s catalog is a strong design reference for narrow on-device products: one model per task,
small downloads, clear capability boundaries, benchmark disclosure, and explicit platform behavior.
Its available models cover word alignment, enhancement, spoken-language ID, PII redaction,
speech recognition, text tagging, and structured extraction. They are not candidates for the
ModelJars public catalog under the current terms: the license is source-available rather than
open source, requires product attribution and device-count telemetry, and prohibits standalone
model or SDK redistribution. A ModelJar is necessarily a standalone redistributable model
artifact. [Desert Ant catalog](https://desertant.com/models/)
[Desert Ant license](https://license.desertant.com/1.0)

The direct lessons for Models are:

- **Adopt task-scoped qualifications.** Align is a useful example: it is a timing refiner, not an
  ASR engine; its claimed 0.7 MB Core ML footprint and fallback semantics are explicitly scoped.
  [Align model page](https://desertant.com/models/align/)
- **Measure end-to-end audio behavior.** Clear reports codec, mastering, chunked I/O, model size,
  and device-specific throughput, not a single abstract “tokens per second” number.
  [Clear model page](https://desertant.com/models/clear/)
- **Do not wrap or redistribute their SDK.** Desert Ant is a benchmark and product-design
  reference only unless the vendor offers terms explicitly permitting standalone ModelJar
  distribution without required telemetry. Their current Core ML/ONNX artifacts may be used for
  controlled interoperability research, never as a released ModelJar.

The initial Desert Ant-inspired JVM work should therefore be small, reproducible models in our own
catalog: language ID, PII redaction/classification, VAD, ASR post-processing, and reranking. These
also directly improve safe local routing.

## Audio and speech opportunity

`audio.cpp` is a valuable reference implementation, not a runtime dependency. Its model matrix
shows Qwen3-TTS 0.6B/1.7B variants, TTS, ASR, VAD, diarization, codecs, and voice conversion; it
also documents that GGUF alone does not make an audio architecture compatible. That last point is
essential: tensor names, graph structure, metadata, audio preprocessing, and codec stages must
all be implemented per family. [audio.cpp supported models](https://github.com/0xShug0/audio.cpp)

The first audio release sequence should be:

1. **ASR text post-processing: S1-mini.** It is a 596M unique-parameter Qwen3-0.6B derivative
   designed solely to normalize raw English ASR text. Its upstream card specifies the exact
   prompt/control contract, greedy decode, `enable_thinking=false`, input limit, license naming
   obligation, and 7,519-case evaluation. This is immediately tractable with the existing Qwen3
   Java decoder. [S1-mini model card](https://huggingface.co/superwhisper/s1-mini)
2. **Audio feature pipeline.** Add owned WAV decode/resample, mel/log-mel generation, chunking,
   streaming timestamps, and deterministic waveform/feature fixtures. This benefits ASR, VAD,
   language ID, enhancement, and TTS.
3. **Qwen3-TTS as a multi-component family.** Import its text frontend, autoregressive acoustic
   model, codec/vocoder, voice reference assets, and generation configuration as one verified
   package. Start with 0.6B base speech, then custom-voice/design variants. No “TTS supported”
   catalog claim until intelligibility, speaker similarity where applicable, duration, streaming,
   audio safety, and real-time factor are qualified on each target.
4. **Only then consider waveform acceleration.** The useful `audio.cpp` findings are session
   lifetime, staged graph reuse, chunked I/O, GPU convolution hot spots, and per-route precision
   qualification—not importing a C++ engine. Its own results warn that lower precision can change
   output quality or even fail on particular paths. [audio.cpp quantization notes](https://github.com/0xShug0/audio.cpp)

## Formats and package policy

### 1. SafeTensors/Hugging Face repository layout — next

SafeTensors is the right upstream format because it is a safe, zero-copy tensor container rather
than a pickle serialization. It also supports partial tensor access, which is important for shards
and future device placement. [SafeTensors documentation](https://huggingface.co/docs/safetensors/main/index)

The unit of import is a **repository manifest**, not a `.safetensors` file:

```text
upstream revision
  ├── config.json
  ├── generation_config.json                 (optional)
  ├── tokenizer.json / tokenizer.model / vocab files
  ├── tokenizer_config.json / special token maps
  ├── chat_template.jinja                    (or template in tokenizer config)
  ├── model.safetensors | model-00001-of-N.safetensors
  ├── model.safetensors.index.json           (when sharded)
  ├── adapter_config.json + adapter weights  (when present)
  ├── preprocessor / processor files          (modalities)
  └── LICENSE, NOTICE, model card provenance
```

The importer must verify that every tensor referenced by the index is present exactly once, map
each tensor to an owned architecture schema, reject unknown required tensors, and retain every
source file hash in the derived ModelJar manifest. It must parse known tokenizer/template formats
without evaluating remote code or arbitrary Jinja. A deliberately small supported template subset
with golden upstream-rendering tests is safer than silently approximating a template.

**Do not accept raw `.pt`, `.pth`, or `.bin` checkpoints as product inputs.** They are commonly
pickle-based and can execute code during deserialization. A separately documented, isolated,
explicitly trusted conversion tool may eventually generate SafeTensors for a developer, but no
ModelJars application or CI qualifier should open a pickle checkpoint.

### 2. ONNX — next, but split encoders from decoders

ONNX is the best second format for encoders, rerankers, classifiers, audio front ends, and selected
small generation models. The official Java bindings are in Maven Central and provide CPU artifacts
for Windows/Linux/macOS x64 plus CUDA artifacts for Windows/Linux. [ONNX Runtime Java](https://onnxruntime.ai/docs/get-started/with-java.html)

The important distinction is architectural:

- An **encoder ONNX** graph is a highly credible early target for owned execution or a sealed,
  internal ONNX bridge: input tensors in, embeddings/logits out.
- An **autoregressive decoder ONNX** needs a causal cache contract, position state, token sampling,
  stop handling, tool parsing, and often past/present KV I/O. Loading one `model.onnx` is not
  generation support.

Phi-4-mini is therefore an excellent ONNX discovery candidate: Microsoft publishes CPU/mobile and
GPU int4 optimized graph packages under MIT, including model configuration. It should first be
used to drive an ONNX package conformance harness, not advertised as a complete model runtime.
[Phi-4-mini ONNX model card](https://huggingface.co/microsoft/Phi-4-mini-instruct-onnx)

Models should not expose arbitrary third-party “execution providers” as an application plugin
surface. We can use ONNX Runtime as an oracle and, if later justified, keep a sealed Models-owned
integration whose supported device paths are explicit and tested. The public API remains Models,
not `OrtSession`.

### 3. Core ML — a first-class Apple target, not a universal catalog format

Core ML is the correct way to reach Apple CPU, GPU, and Neural Engine for models that have a real
Core ML graph. The source package still begins as SafeTensors/ONNX/Hugging Face assets; an Apple
build step creates a platform artifact whose exact compiler, deployment target, and input/output
schema must be pinned.

Core ML should be represented in the catalog as a **platform variant of one logical model**:

```text
logical model + semantic version
  ├── Java/GGUF reference artifact
  ├── Apple Core ML compiled artifact (platform, OS, compiler fingerprint)
  └── qualification matrix per artifact
```

The sealed Models Apple bridge selects it only when the package, OS, hardware capability, prompt
processor, and qualification all match. It must fail back to the owned Java path; it may not claim
cross-platform portability.

### 4. ExecuTorch `.pte` — mobile deployment after the importer foundations

ExecuTorch is valuable for Android/mobile because it distributes `.pte` deployment programs and a
Java/Kotlin AAR with mobile acceleration options. It is not simply “another file extension”: a
`.pte` package binds graph operators, delegates, preprocessing, model configuration, and device
assumptions. It belongs after SafeTensors package/provenance and ONNX encoder work, as a sealed
Android deployment lane rather than an open backend mechanism. [ExecuTorch Android documentation](https://docs.pytorch.org/executorch/stable/using-executorch-android.html)

### 5. TensorRT plans and compiled engines — not a catalog format

TensorRT plans are tied to TensorRT version, CUDA version, GPU architecture, driver behavior, and
often optimization profiles. A thin Rust shim cannot make them portable; it merely creates a
second external runtime dependency. Treat plans as private benchmark/deployment build products,
never downloadable ModelJars artifacts. The same applies to arbitrary accelerator binaries.

## KV cache and composition research

### What is physically shareable

A KV cache contains per-layer key/value projections, not a generic representation of text. It can
be aliased without recomputation only when all of the following match:

- exact model weights and architecture;
- exact tokenizer/token IDs and all rendered bytes/tokens up to the shared boundary;
- exact position/RoPE/scaling state and attention-mask semantics;
- exact K/V projection state for every shared layer; and
- exact cache dtype, layout, quantizer, and block geometry.

“Same family” is insufficient: Qwen3 0.6B and 1.7B differ in shape and weights, so they cannot
share K/V blocks. Two ordinary LoRA adapters also cannot share a prefix if either changes K/V
projections before that prefix. Activated LoRA is special only when the adapter is provably
inactive over an immutable prefix; after activation, the suffix is separate. This is why the
current hybrid experiment correctly checks object/block identity rather than inferring sharing
from a routing decision.

### Adopt the useful system ideas, not their external runtime

PagedAttention divides KV cache into fixed token blocks and maps logical request blocks onto
physical blocks, enabling refcounts and prefix sharing. Its original evaluation reported 2–4x
throughput improvement at comparable latency for serving workloads; the mechanism itself is
directly relevant to a JVM cache allocator. [PagedAttention paper](https://arxiv.org/abs/2309.06180)

SGLang’s RadixAttention makes the matching structure explicit: a radix tree finds and retains the
longest reusable prompt prefix. That is the right ownership/eviction model for a local multi-model
router, but it should be implemented in Models as an in-process `PrefixBlockStore`, not delegated
to an SGLang server. [SGLang paper](https://openreview.net/attachment?id=VqkAKQibpq&name=pdf)

KIVI shows that K/V quantization is asymmetric: keys want per-channel grouping and values
per-token grouping. Its reported result is a useful hypothesis—lower memory can increase feasible
batch size—but it must be re-evaluated per architecture and workload, especially tool calls and
long-context retrieval. [KIVI paper](https://arxiv.org/abs/2402.02750)

LMCache demonstrates useful concepts—tiered storage, chunking, optional serialization,
compression, encryption, and explicit cache operations—but its daemon is not part of the Models
product architecture. Its warning that quantized external cache storage trades bytes for a small
quality loss is precisely why Models needs an owned evaluation gate. [LMCache compression notes](https://docs.lmcache.ai/mp/serde.html)

### Proposed owned `KVCacheEnvelope` v1

Create an internal, versioned envelope before attempting disk persistence or cross-process reuse:

```text
KVCacheEnvelopeV1
  modelFingerprint            # weights + architecture + importer revision
  tokenizerFingerprint        # tokenizer + special tokens + template renderer
  promptPrefixHash            # token IDs, not merely UTF-8 input text
  positionState               # RoPE/scaling/window state
  attentionSemantics          # causal/sliding/bidirectional and masks
  cacheLayout                 # layers, KV heads, head dimension, block token count
  storageEncoding             # f32/bf16/fp16/q8/... plus scale/zero layout
  activationFingerprint       # adapter identity and exact activation boundary
  safetyDomain                # tenant, data class, TTL, remote-transfer prohibition
  blocks[]                    # immutable refcounted physical block handles or serialized payloads
```

Rules:

- An envelope mismatch is a cache miss, never a conversion attempt.
- In-memory same-process aliasing is phase one; encrypted disk persistence and RAM/GPU movement
  are later phases.
- Any lossy cache encoding needs a model/workload-specific quality profile and a reversible
  fallback to higher precision.
- Cache keys are tenant- and policy-scoped. Never let a prefix containing one customer’s data
  become a reusable block for another customer merely because text hashes collide or match.
- Metrics must expose matched tokens, physical bytes saved, cache-hit TTFT reduction, dequant time,
  and correctness deltas—not only hit rate.

### “Reasoning-core extraction” is not a safe product primitive

There is no general way to cut a reasoning-only subset out of a trained transformer and expect it
to retain useful reasoning. Hidden states, attention projections, MLPs, positional encoding, and
the output head are jointly trained. Generic weight slicing or copying an internal “reasoning
layer” would be ungrounded engineering.

The legitimate alternatives are:

| Technique | What it can do | What it cannot do |
| --- | --- | --- |
| Distillation | Train a smaller router, reasoning model, or domain model from evaluated behavior. | Reuse the parent model’s KV blocks across different weights. |
| Activated LoRA/adapters | Share immutable base-prefix blocks when activation begins after the prefix. | Make an underperforming base/tool specialist accurate by cache policy alone. |
| Speculative decoding | Let a small draft propose tokens and a larger target verify them. | Make the draft KV physically valid for the target; target verification still owns correctness. |
| Early exit | Reduce work if a model is trained/evaluated with trustworthy intermediate heads. | Be safely bolted onto an arbitrary released checkpoint. |
| MoE routing | Activate fewer experts inside one model. | Permit K/V sharing across unrelated models or expert layouts. |
| Semantic summary/cache | Compress old conversation into explicit text or structured state. | Preserve exact token-level behavior like a KV cache. |

## Router: from “choose a model” to a privacy-aware local execution planner

The router should make an auditable decision per request, not heuristically chase a single score:

```text
request
  -> local PII / policy classifier and redaction
  -> task and modality classifier
  -> capability filter (tools, JSON, embeddings, rerank, vision, audio, context)
  -> hardware fit (RAM/VRAM/Apple capability, supported ISA, cached model residency)
  -> exact-prefix/KV reuse lookup
  -> local quality-cost-latency ranking
  -> local model / hybrid route / explicit remote escalation
  -> observed outcome + bounded feedback record
```

The decision record should include: policy version; whether remote use was allowed; detected data
class; redaction result; model candidates rejected and why; cache match length/bytes; predicted vs
actual TTFT, TPOT, completion tokens, cost, and quality signal; plus the model/artifact hashes.
This makes optimization explainable and makes a corporate deployment able to prove that sensitive
content was kept local.

The target policy makes remote APIs a last tier, not a default fallback. It must explicitly permit
them, specify approved vendors/models/regions, cap cost and latency, and define whether raw
prompt, redacted prompt, retrieved excerpts, or only a local summary may leave the process. A
refusal to escalate is a valid router outcome.

### Router audit, 2026-09-14

The current router is a useful, tested selection layer rather than a blank slate. It already:

- filters for context, configured quality/cost/TTFT floors, live availability and explicit
  application filters before it scores;
- scores measured quality, cost, latency, reliability, locality, model residency, queue depth, and
  *per-model* retained prompt-prefix evidence;
- preserves an active tool/provider-state turn on its current healthy model and binds fallback
  execution to application-owned local or hosted clients without importing any provider SDK; and
- retains bounded session affinity, but never presents one model's KV tensors as usable by another.

This round adds two previously missing hard constraints: `RoutingRequirements` carries exact
declared capabilities (for example `tool-calling`) and `RoutingDataBoundary.LOCAL_ONLY` keeps one
request in process even when the fleet policy otherwise permits remote models. `DiscoveredModel`
now preserves capabilities separately from quality/task tags, so catalog discovery does not
silently collapse “tool-capable” into a vague `tool-use` label. The capability and boundary tests,
including provider-neutral hosted-client selection, are part of the router's unit suite.

It is intentionally **not** yet a universal cache-mobility system. Cache affinity is an observed
token count keyed by the physical model, not proof that blocks can move. The next runtime work is
an internal, non-public `KVCacheEnvelopeV1` plus block ownership/fingerprints; only then can a
same-base activated adapter or another independently proved pair share a prefix. There is also no
automatic PII classifier yet: an application or separately qualified local policy layer must set
the data boundary, rather than relying on unsafe prompt-text guessing.

## Execution roadmap

### Next 30 days

1. Implement and test a SafeTensors repository manifest/parser with shard-index validation,
   memory-mapped tensor access, tokenizer/template fingerprints, and an allowlisted architecture
   mapping. Start with BERT/MiniLM and Qwen3 fixtures.
2. Qualify S1-mini as a text-normalization audio stage through the existing Qwen3 Java path.
3. Run owned Java GGUF baseline screens for Qwen3 8B, SmolLM3 tool calling, LFM2 RAG, and Granite
   4.1 8B; release only independent passes.
4. Define `KVCacheEnvelopeV1`, fixed-size block allocator, refcounts, radix-prefix index,
   tenant scope, and cache correctness tests. Keep storage in-process and lossless initially.
5. Start the Qwen3 8B hybrid experiment with fresh audited training/validation data and a new
   frozen development screen. The rejected Qwen3 1.7B policy/adapter does not transfer.

### Following 30–60 days

1. Bring up ONNX for encoder/reranker/classifier packages, then Phi-4 mini decoder feasibility.
2. Add a sealed Apple Core ML lane for specifically qualified encoders/audio components.
3. Add KV precision experiments: BF16/FP16 baseline, then Q8 and KIVI-inspired asymmetric modes,
   evaluated on retrieval, tool calls, long context, and exact generation—not perplexity alone.
4. Add router policy enforcement, PII-redaction gates, local-hardware profiling, and cost/quality
   decision traces.
5. Begin Qwen3-TTS package reconnaissance and feature-pipeline implementation; do not schedule a
   catalog release until all components and waveform gates exist.

## Sources

1. [Hugging Face, “SafeTensors documentation.”](https://huggingface.co/docs/safetensors/main/index)
2. [Hugging Face, “Models on the Hub.”](https://huggingface.co/docs/hub/models)
3. [Microsoft, “Phi-4-mini-instruct ONNX.”](https://huggingface.co/microsoft/Phi-4-mini-instruct-onnx)
4. [Microsoft, “Phi-4-mini-instruct model card.”](https://huggingface.co/microsoft/Phi-4-mini-instruct)
5. [Qwen, “Qwen3-8B model card.”](https://huggingface.co/Qwen/Qwen3-8B)
6. [Google, “FunctionGemma 270M model card.”](https://huggingface.co/google/functiongemma-270m-it)
7. [Hugging Face, “SmolLM3 deployment and tool-calling example.”](https://huggingface.co/docs/microsoft-azure/foundry/examples/deploy-smollm3)
8. [Superwhisper, “S1-mini model card.”](https://huggingface.co/superwhisper/s1-mini)
9. [Liquid AI, “LFM2-1.2B-RAG-GGUF model card.”](https://huggingface.co/LiquidAI/LFM2-1.2B-RAG-GGUF)
10. [Desert Ant Labs, “Models.”](https://desertant.com/models/)
11. [Desert Ant Labs, “Align.”](https://desertant.com/models/align/)
12. [Desert Ant Labs, “Clear.”](https://desertant.com/models/clear/)
13. [Desert Ant Labs, “Source-Available License.”](https://license.desertant.com/1.0)
14. [0xShug0, “audio.cpp.”](https://github.com/0xShug0/audio.cpp)
15. [ONNX Runtime, “Java.”](https://onnxruntime.ai/docs/get-started/with-java.html)
16. [PyTorch, “ExecuTorch Android.”](https://docs.pytorch.org/executorch/stable/using-executorch-android.html)
17. Kwon et al., [“Efficient Memory Management for LLM Serving with PagedAttention.”](https://arxiv.org/abs/2309.06180)
18. Zheng et al., [“SGLang: Efficient Execution of Structured Language Model Programs.”](https://openreview.net/attachment?id=VqkAKQibpq&name=pdf)
19. Liu et al., [“KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.”](https://arxiv.org/abs/2402.02750)
20. [LMCache, “KV Cache Compression.”](https://docs.lmcache.ai/mp/serde.html)
