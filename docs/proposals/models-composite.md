# `models-composite` — a virtual model assembled from smaller ones

Status: PROPOSAL (runtime assumptions re-audited 2026-09-06; no composite built).
Origin: memory project register rows 5j/5o; the workbench chat-tier question — "what local
configuration beats the wire?" — kept converging on composition rather than a single bigger model.

## 1. The idea in one paragraph

A `CompositeModel` implements `TextGenerationModel` — same `modelName()`, `diagnostics()`,
`generate(prompt, options, stream)` contract as any backend — but inside it is a *policy over
parts*: a turn classifier that dispatches to a prose specialist or a tool-calling specialist, a
token constraint applied when structure is required, and (phase 2) adapter swapping over one
shared base with adapter-isolated KV state. Cross-adapter KV reuse is a later experiment requiring
proof that the produced K/V activations are equal. To every consumer — Spring AI adapter, workbench,
memory-server, and critically the **router** — it is just another model with an id, a performance
profile, and quality scores. Composition becomes a *catalog entry*, not an application concern.

## 2. What the research established (read from the code, this repo, 2026-09-04)

- `TextGenerationModel` is streaming-first (`generate(prompt, options, TokenStream)`) with
  blocking wrappers — a facade a composite can implement without API changes.
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
- The catalog SPI (`ModelCatalogProvider` → `DiscoveredModel{id, local, tags, contextWindow,
  costs, sizeBytes, Performance, quality, successRate}`) is ServiceLoader-discovered, and the
  router consumes it via `CatalogDiscovery`. The router **selects, never calls** — so a composite
  registered in the catalog is routable today with zero router changes.

Literature precedent for the composition levels (read, not measured): specialist function-callers
(Gorilla, xLAM), token-level collaboration (Co-LLM; speculative decoding as the speed variant),
cross-attention weight composition (CALM), shared-base multi-adapter serving (S-LoRA, Punica).

## 3. Module shape

```
models-composite/
  CompositeModel            implements ConstrainedTextGenerationModel — the facade
  TurnPolicy                SPI: classify(prompt) -> route          (default: router's task classifier)
  CompositePart             a named part: delegate model + optional TokenConstraint + ChatTemplate
  CompositeRecipe           declarative assembly: parts, policy, fallback order, budget
  CompositeCatalog          implements ModelCatalogProvider — publishes recipes as DiscoveredModels
  recipes/                  built-in recipes (see §5)
```

Key design commitments:

1. **The facade is honest.** `modelName()` names the recipe (`composite:chat-local-v1`);
   `diagnostics()` reports every part's backend diagnostics plus which part served the last call —
   a composed answer must never be unattributable (the same provenance instinct as everywhere else
   in this family).
2. **Dispatch is a policy SPI, not an if-chain.** Default `TurnPolicy` = the router's bundled
   task classifier (already shipped, 0.90 held-out); a recipe may override with a rule (e.g. "the
   prompt's tail contains tool schemas ⇒ tool part").
3. **Constraints ride the part, not the call.** A `CompositePart` binds its delegate to an
   optional `TokenConstraint` factory (e.g. tool part + `JsonSchemaConstraint` over the declared
   tools). Malformed structured output becomes impossible by construction on that part.
4. **Parts are models, so parts can be composites or hosted.** A recipe may name a hosted
   delegate for one role — composition and routing then nest naturally (a PRIVACY_STRICT policy
   at the router level still filters the whole composite by its `local` flag, which is only true
   if every part is local).
5. **Budget accounting at the seam.** The composite counts tokens per part per call and exposes
   them through diagnostics — the numbers a spend ledger (e.g. the Forge's) needs, produced where
   they are known.

## 4. Required seams in existing modules (small, additive)

### 4.1 Constraint plumbing (already available)
`ConstrainedTextGenerationModel.generate(prompt, options, stream, constraint)` is the public seam.
A composite part can bind a `TokenConstraint` factory to its delegate and invoke this capability
without changing `SamplingOptions` or moving the composite into `models-runtime`.

### 4.2 Catalog entry for virtual models (none needed)
`CompositeCatalog` is just another `ModelCatalogProvider` on the classpath. `sizeBytes` = sum of
parts; `Performance` = measured per recipe (the ladder below), never derived by guessing;
`quality` keys per task ("chat", "tool-calling") measured per recipe.

### 4.3 Explicit conversation state (merged, pending release)
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

## 5. Built-in recipes and their gates (nothing ships un-measured)

| recipe | parts | gate |
|---|---|---|
| `chat-local-constrained` | one small instruct model; tool part = same model + JSON-schema constraint | beats the same model unconstrained on tool-call correctness at equal latency (expected: deletes the malformed-call class) |
| `chat-local-dispatch` | tool specialist (≤1B) + prose model (1.5–3B), turn dispatch | beats `chat-local-constrained` after charging separate KV and model-switch catch-up prefill, or it dies |
| `chat-local-adapters` (phase 2) | shared base + specialist adapters with isolated KV | beats two fully loaded models on memory without losing latency or quality |
| `chat-local-kv-compatible` (research) | specialization proven to preserve every layer's K/V activations | beats isolated adapter state on switch latency with token-equivalent handoffs |

Measurement protocol: the workbench chat ladder already defined in the memory project — five canned
memory interactions (store, search, prose), tool-call correctness, phase timings, prompt-cache
read/write tokens, cold/warm/switch-back latency, peak RSS/native memory, and selected model for
each turn. Run against the measured hosted round-trip. Same harness, new arms.

## 6. Why this belongs in the models project

The router made model *selection* infrastructure; this makes model *assembly* infrastructure, and
the two compose: a recipe is a candidate, so the router can choose between "one big local model",
"a composite of two small ones", and "the wire" on measured cost/latency/quality like any other
choice. Downstream (memory-server chat, workbench, Forge deliberators) then consume compositions
without knowing they are composed — which is exactly the modeljars thesis applied one level up:
models as dependencies, now compositions as dependencies.

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

## 8. Open questions (for the build phase)

- Streaming across dispatch: a turn routed mid-stream cannot switch parts; dispatch is
  per-generate-call (fine for chat turns; recorded limitation).
- ChatTemplate mismatch between parts (CHATML vs others): the recipe owns per-part templates;
  the composite's `ModelPrompt` handling must re-render per part, not share rendered text.
- Where does tool-schema knowledge live? The constraint factory needs the tool JSON schemas at
  call time — likely via `ModelPrompt` metadata rather than string parsing (needs an API look).
- Router `quality` keys for composites: measured per recipe on which benchmark? (Tool-fidelity
  protocol for "tool-calling"; a small chat eval for "chat" — decide when the ladder runs.)
