# `models-composite` — a virtual model assembled from smaller ones

Status: EXPERIMENTAL (virtual runtime implemented and measured 2026-09-07; no packaged recipe).
Origin: memory project register rows 5j/5o; the workbench chat-tier question — "what local
configuration beats the wire?" — kept converging on composition rather than a single bigger model.

## 1. The idea in one paragraph

`VirtualChatModel` owns one canonical role-aware conversation and a policy over physical parts. A
selector dispatches each semantic turn to a chat or tool specialist, each part renders the shared
messages with its own template, and each part retains an independent `TextGenerationSession` and
KV lineage. `VirtualChatRouter` supplies classifier, policy, affinity, and feedback integration.
This cannot correctly implement the raw `TextGenerationModel` contract because that contract
receives an already-rendered prompt; re-rendering per model requires semantic messages. Packaging a
measured recipe behind a catalog identity remains a later phase.

## 2. What the research established (read from the code, this repo, 2026-09-04)

- `TextGenerationModel` is streaming-first (`generate(prompt, options, TokenStream)`) with
  blocking wrappers, but its rendered prompt is the wrong abstraction for members with different
  chat templates. The virtual facade therefore owns canonical `ChatMessage` history.
- **Constrained decoding is public** in `models-runtime`: `ConstrainedTextGenerationModel`,
  `TokenConstraint`, `JsonSchemaConstraint`, and `ToolCallTokenConstraints`. A composite can invoke
  the capability directly; a constraint does not belong in provider-neutral `SamplingOptions`.
- The ordinary pipeline retains one exact prompt-prefix lineage across calls. The merged
  `TextGenerationSession` API adds one high-level session per conversation over the existing
  low-level `InferenceSession`, sharing loaded weights while isolating KV, retained prefixes,
  lifecycle, and metrics.
- No adapter/LoRA loading or swap API exists. Sharing a base's weights would not make adapter KV
  interchangeable: adapter-modified hidden states change later-layer K/V activations even if K/V
  projection matrices themselves are untouched.
- Raw KV remains model-specific. Activated LoRA creates one exact exception by leaving a prefix on
  the base model before adapter invocation. Calibrated cross-model mappers are a separate,
  approximate research path and are not implemented by the virtual runtime.
- The catalog SPI (`ModelCatalogProvider` → `DiscoveredModel{id, local, tags, contextWindow,
  costs, sizeBytes, Performance, quality, successRate}`) is ServiceLoader-discovered, and the
  router consumes it via `CatalogDiscovery`. The router **selects, never calls** — so a composite
  registered in the catalog is routable today with zero router changes.

Literature precedent for the composition levels (read, not measured): specialist function-callers
(Gorilla, xLAM), token-level collaboration (Co-LLM; speculative decoding as the speed variant),
cross-attention weight composition (CALM), shared-base multi-adapter serving (S-LoRA, Punica),
base-aligned prefix reuse ([Activated LoRA](https://arxiv.org/abs/2512.17910)), and calibrated
cross-model KV translation ([closed-form within-family](https://arxiv.org/abs/2608.03893) and
[universal context-reuse](https://arxiv.org/abs/2608.30963) proposals).

## 3. Implemented shape

```
models-runtime/
  VirtualChatModel          semantic facade and independent physical sessions
  PromptPrefillMetrics      measured prefill-only catch-up

models-router/
  VirtualChatRouter         adaptive selector and runtime-feedback bridge
```

Key design commitments:

1. **The facade is honest.** Every response names the physical member, route reason, task, switch
   boundary, parsed tool calls, and generation metrics; a composed answer is never unattributable.
2. **Dispatch is a policy SPI, not an if-chain.** The default `Selector` follows declared
   capabilities; `VirtualChatRouter` adds the bundled classifier and adaptive routing policies.
3. **Constraints ride the member.** A member may bind an optional `ConstraintFactory`, evaluated
   with the current tools for each selected turn.
4. **Implemented members are local sessions.** Mixed local/hosted fleets remain supported by
   `ModelFleet` and the framework routed adapters, but hosted state has not been folded into the
   stateful virtual-session API.
5. **Budget accounting at the seam.** The virtual response exposes the physical member's prompt,
   completion, cache-read, cache-write, TTFT, and decode measurements where they are known.

## 4. Required seams in existing modules (small, additive)

### 4.1 Constraint plumbing (already available)
`ConstrainedTextGenerationModel.generate(prompt, options, stream, constraint)` is the public seam.
A composite part can bind a `TokenConstraint` factory to its delegate and invoke this capability
without changing `SamplingOptions` or moving the composite into `models-runtime`.

### 4.2 Catalog entry for virtual models (future)
A future `CompositeCatalog` can be another `ModelCatalogProvider` on the classpath. `sizeBytes` =
sum of parts; `Performance` = measured per recipe (the ladder below), never derived by guessing;
`quality` keys per task ("chat", "tool-calling") measured per recipe. No recipe is registered now.

### 4.3 Explicit conversation state (released in 0.3.30)
`InferencePipeline.openGenerationSession()` opens independent high-level generation state over one
loaded, batch-capable backend. Each conversation owns its exact prompt prefix, KV state, metrics,
reset, and close lifecycle. Calls remain serialized because backend scratch is shared. This is the
state boundary required to measure turn dispatch honestly; nothing in phases 0–1 requires LoRA.

### 4.4 Phase 2 — adapter loading and cache-safe switching (`models-runtime`)
Add LoRA adapter loading/swap over a shared base, initially with one KV lineage per adapter. Record
adapter identity in every cache key and measure weight savings, switch overhead, catch-up prefill,
and duplicate KV memory. Cross-adapter KV reuse is a separate gated experiment: the actual K/V
activations must match every layer under full recomputation. Restricting adapters to Q + MLP is not
sufficient because changed hidden states feed later K/V projections.

### 4.5 Phase 3 — measured context mobility (`models-accelerator-bench` first)
Do not add a public cache-import API first. Screen immutable artifacts for layer count, KV-head
count, per-head key/value dimensions, RoPE, and tokenizer identity; then capture test-only
unrotated K/V and fit a pair-specific mapping. Only a held-out quality and latency win can justify a
runtime SPI. The current screening tool always reports `directReuseSafe=false`.

## 5. Built-in recipes and their gates (nothing ships un-measured)

| recipe | parts | gate |
|---|---|---|
| `chat-local-constrained` | one small instruct model; tool part = same model + JSON-schema constraint | beats the same model unconstrained on tool-call correctness at equal latency (expected: deletes the malformed-call class) |
| `chat-local-dispatch` | tool specialist (≤1B) + prose model (1.5–3B), turn dispatch | beats `chat-local-constrained` after charging separate KV and model-switch catch-up prefill, or it dies |
| `chat-local-adapters` (phase 2) | shared base + specialist adapters with isolated KV | beats two fully loaded models on memory without losing latency or quality |
| `chat-local-activated-adapter` (research) | base model + adapter activated after explicit invocation tokens | reuses only the proven base-aligned prefix and beats isolated adapter catch-up with equivalent output |
| `chat-local-kv-translated` (research) | pair-specific calibrated mapper between two matched-KV models | beats native target prefill while retaining declared held-out target quality; every translated handoff is observable |

Measurement protocol: the workbench chat ladder already defined in the memory project — five canned
memory interactions (store, search, prose), tool-call correctness, phase timings, prompt-cache
read/write tokens, cold/warm/switch-back latency, peak RSS/native memory, and selected model for
each turn. Run against the measured hosted round-trip. Same harness, new arms.

### 5.1 Qwen dispatch screen, 2026-09-07

Qwen3 0.6B Q4_0 for chat plus qualified Qwen3 1.7B Q8_0 for tools passed all five turns through the
pure-Java backend. Independent prefix evidence behaved as designed: the first tool switch was cold
without catch-up, and both switch-backs reused only the selected member's own cached tokens.

Eager background prefill reduced the first tool switch from 57.953 s to 44.601 s at the default 12
GGUF workers, but raised the five-turn total from 104.514 s to 107.272 s by competing with foreground
chat. Six-worker prefill beat its matched six-worker control, but not the best no-background result.
The semantic facade proceeds; this pair does not become a built-in recipe. Raw reports and the
rejected after-turn scheduler are retained by the Memory composition harness.

The two Qwen artifacts also passed a metadata-only calibration screen: both have 28 layers, 16
query heads, 8 KV heads, 128-wide keys/values, the same RoPE base, and an identical tokenizer
vocabulary. Their hidden widths differ and their KV values are not directly reusable. This makes
them a useful pair for a test-only calibrated transfer experiment, not a product feature or recipe.
The first small Java ridge screen found strong key structure but poor held-out value reconstruction
(0.367–0.576 cosine; 0.819–0.987 relative L2), so its single-layer and raw-correlation top-two
mappers were rejected. No cache-transfer SPI is authorized by that result.

## 6. Why this belongs in the models project

The router made model *selection* infrastructure; this makes model *assembly* infrastructure, and
the two compose: a recipe is a candidate, so the router can choose between "one big local model",
"a composite of two small ones", and "the wire" on measured cost/latency/quality like any other
choice. Downstream semantic-chat consumers can use one virtual session while retaining
physical-model provenance in each response. A future packaged recipe would apply the ModelJars
thesis one level up: models as dependencies, then compositions as dependencies.

## 7. Landscape (swept 2026-09-04, docs/source level; discovery search unavailable — found-no-evidence, not verified-absent)

Every ingredient of "composition as a dependency-managed package" exists in the wild; no system
has them together.

| property | who has it | why it falls short |
|---|---|---|
| named/versioned/installable model package | Ollama, LM Studio Hub, HF, Docker | single model |
| package referencing another **package** | LM Studio `model.yaml` `base:` chain | *variant substitution* (GGUF vs MLX), not assembly |
| runtime two-model dependency declared in an artifact | vLLM `speculators` (`verifier.name_or_path` in config.json) | unpinned, draft/verify only, weights-with-pointer |
| two models inside one registry artifact | **Ollama `DRAFT`** (undocumented, v0.30.0+, OCI layer) | vendors a *local file*; registry references explicitly refused |
| declarative multi-model graph | KServe InferenceGraph, Seldon Pipeline, Ray Serve, NVIDIA llm-router | cluster resources, not packages |
| pinned model deps + library deps in one versioned artifact | **BentoML Bento** (`models: [name:version]`) | imperative composition, no transitive model resolution |
| multi-model composition behind **one model facade** | **OpenRouter** `openrouter/fusion`, versioned presets | hosted, account-scoped, closed |
| build-time recipe culture with codenames | HF **mergekit** | weight *merging* — produces new weights, not runtime composition |

**Conclusion:** a recipe package whose declared dependencies are themselves model packages,
resolved transitively, presenting one model facade — no evidence found anywhere. Two nearest
misses show how a competitor could close it: OpenRouter adding a package path, or Docker Compose
OCI publishing adding a resolver. Public claims must say "we found no prior art", never "none
exists".

**Proven combos worth seeding as recipes** (read in docs/papers, none measured here — each gets
the tool-fidelity ladder before a codename ships): the RedHatAI speculator↔verifier matrix
(EAGLE-3/DFlash over Llama-3.1-8B/70B, Qwen3-8B/14B/32B, gpt-oss-20b/120b); LM Studio's
same-vocabulary pairs (Llama-3.1-8B ← Llama-3.2-1B; Qwen2.5-14B ← Qwen2.5-0.5B); vLLM's
cross-vocabulary TLI pairing (Qwen3-8B ← SmolLM2-135M); Mixture-of-Agents (6 proposers x 3
layers, aggregator matters — WizardLM proposes well and aggregates badly); RouteLLM's published
routers; Plano/archgw's 4B orchestrator beside task agents.

## 8. Open questions

- Streaming across dispatch: a turn routed mid-stream cannot switch parts; dispatch is
  per-generate-call (fine for chat turns; recorded limitation).
- Framework adapters need an explicit conversation/session ownership contract; a singleton Spring
  or LangChain4j model cannot silently infer which concurrent caller owns retained state.
- A packaged recipe needs a ModelJars-owned lifecycle that opens every pinned part and closes the
  virtual session before its runtimes; the current builder deliberately receives session factories.
- Router `quality` keys for packaged recipes: measured per recipe on which benchmark? (Tool-fidelity
  protocol for "tool-calling"; a small chat eval for "chat" — decide when the ladder runs.)
