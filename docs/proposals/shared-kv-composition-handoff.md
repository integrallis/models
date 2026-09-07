# Handoff: two small models in one chat — state, switching, and composition

**Status:** runtime assumptions audited against Models 0.3.31. Rungs 1–2 were measured downstream
on 2026-09-06; Qwen3 1.7B passed while Qwen + Needle2 fell to 2/5. A second pair and background
catch-up were measured on 2026-09-07: Qwen3 0.6B chat + Qwen3 1.7B tools passed 5/5 with isolated
KV lineages, but background prefill did not beat the best demand-catch-up total. The virtual-model
API is implemented on `feat/hybrid-model-composition`; no preconfigured composite recipe is yet
authorized by performance evidence.

Origin: a user question on 2026-09-04 — *"can two small models serve the chat function, is there a
model composition precedent, where one model deals with the prose and another deals with tool
calling for example, while sharing context, KV cache, etc?"* Recorded as row **5o** in
`memory-server-refactor-plan.md`, with the artifact-packaging half as row **5p**.

---

## 1. The physical constraint that shapes everything

**Raw KV from one independently trained model cannot be directly installed in another model.** The
cache is a tensor of per-layer key/value projections; its shape is tied to layer count, head count
and head dimension, and its *values* are tied to the specific weights that produced them. Matching
shapes make a transfer experiment possible; they do not make the values interchangeable.

There are three honest choices at a routing boundary:

1. Keep independent KV state for each model or adapter identity.
2. Recompute the missing context when switching, while retaining each model's last exact prefix.
3. Reuse KV only after proving that the state which produced every cached K/V activation is
   identical.
4. Translate one model's KV into another model's representation with a calibrated mapper, then
   accept it only if the complete quality and latency gates pass. This is approximate transfer, not
   shared raw state.

Sharing a base model's weights saves weight memory. It does **not** by itself make two adapters' KV
caches interchangeable.

Two recent research paths make that distinction important:

- **[Activated LoRA (aLoRA)](https://arxiv.org/abs/2512.17910)** keeps the prefix before explicit
  invocation tokens aligned with the base model. That prefix has the same effective model state and
  can be reused exactly; ordinary LoRA does not have this invariant.
- **Cross-model KV translation** fits per-head mappings between source and target caches. A 2026
  [closed-form ridge approach](https://arxiv.org/abs/2608.03893) reports useful results only for
  some matched-KV pairs, with sharp failures on others. A newer [universal context-reuse
  proposal](https://arxiv.org/abs/2608.30963) also reports cross-family and cross-tokenizer
  translation, but remains research evidence rather than a runtime contract.

Models implements neither mechanism today. `VirtualChatModel` keeps independent exact state and
never silently substitutes translated activations.

### Corrected assumption: Q + MLP-only adapters are not enough

The earlier plan claimed that adapters restricted to Q projections and MLP layers would preserve an
exact KV cache because K/V weights do not change. That claim is false in a multilayer transformer.
Changing Q or the MLP changes a layer's output hidden state. That changed hidden state becomes the
input to the next layer, so the next layer's K/V activations change even when its K/V matrices are
unchanged. Layer-zero K/V may match while later-layer K/V does not.

The cache key therefore includes the effective model state, including adapter identity, unless a
stronger invariant over the produced activations is demonstrated. Output-head-only specialization
may preserve transformer KV, but its usefulness for prose/tool specialization is an experiment, not
an assumption.

Models 0.3.31 has no adapter/LoRA loader or swap API, so adapter experiments remain blocked. It does
provide one high-level `TextGenerationSession` per conversation over the existing low-level
`InferenceSession`, with isolated prompt/KV lineage and metrics.

---

## 2. Prior art, by coupling level

All read at documentation or source level on 2026-09-04. None reproduced.

| Level | Systems | What they actually do |
|---|---|---|
| **Turn dispatch** | Gorilla, xLAM | 1B-class *tool specialists* reported to beat larger generalists at function calling. Each turn goes to one model; no sharing. |
| **Token-level** | Co-LLM (learned deferral), speculative decoding | One model defers individual tokens to another. Speculative decoding is the latency-oriented variant of the same shape. |
| **Weight-level** | CALM | Cross-attention between two frozen models — composition inside the network, not at the serving layer. |
| **Shared-base multi-adapter** | S-LoRA, Punica | Many LoRAs served over one base with batched adapter application. This shares base weights; cache compatibility still requires an adapter-aware policy. |
| **Base-aligned adapter prefix** | Activated LoRA | Defers adapter activation until invocation tokens, preserving exact base-model KV before that boundary. |
| **Translated cross-model KV** | closed-form ridge mapping; universal context-reuse layer | Derives target-shaped KV from source state. This can avoid target prefill, but it is approximate and pair/calibration dependent. |

The gap: turn-dispatch systems retain separate state and pay catch-up prefill when a specialist has
not seen the latest turns; token-level systems are about speed or quality, not role specialisation;
S-LoRA-class systems solve *serving many adapters*, not *composing two specialists within one
conversation*.

---

## 3. The measured ladder — cheapest rung first

Written this way deliberately. **Rung 1 may capture most of the value and does not need two models
at all.** Do not skip it to get to the interesting engineering.

### Rung 1 — one small model + grammar-constrained tool decoding
Constrain decoding to a grammar that only admits well-formed tool calls. This **deletes the
malformed-call failure class outright** rather than reducing it. Hypothesis: most of what looks like
"small models are bad at tool calling" is actually malformed output, not wrong intent.

*Cost:* no second model, no second prefill.
*Decides:* whether composition is needed at all.

*Runtime status:* ready. `ConstrainedTextGenerationModel`, `TokenConstraint`,
`JsonSchemaConstraint`, and `ToolCallTokenConstraints` are public in 0.3.29.

### Rung 2 — turn-dispatch pair
A classifier routes each turn to prose-model or tool-model. **The dispatcher already exists**: the
router ships a bundled task classifier at 0.90 held-out accuracy (row 5j).

*Cost:* separate model weights, KV memory, and prompt-prefix lineages. A switch requires the newly
selected model to catch up on conversation tokens added since it last ran. It need not blindly
prefill the whole transcript on every turn if its own session is retained, but it cannot reuse the
other model's KV.
*Decides:* whether role specialisation beats a single model, before paying for cache engineering.

*Runtime status:* measured for two candidate pairs. `ModelFleet` bound both local clients and the
session API supplied isolated state and model-specific prefix metrics. The Qwen/Needle pair failed
2/5 and is rejected. Qwen3 0.6B chat + Qwen3 1.7B tools passed 5/5: the tool member read 0/258 tokens
on its first demand switch, while each switch-back reused only that member's retained prefix.

### Rung 3 — independent sessions with background catch-up
Prefill an inactive member's current semantic history while the selected member generates. The
prepared KV remains owned by the inactive member; a later switch reuses that member's exact prefix.

*Measured result:* 5/5 at both default 12-worker and six-worker settings. At 12 workers, first tool
switch latency fell from 57.953 s to 44.601 s, but cold chat rose from 13.091 s to 26.136 s and the
five-turn total rose 2.6%. Six-worker background prefill beat its matched six-worker demand control
by 10.5%, but remained 4.5% slower than the best result, 12 workers without background work. An
after-turn scheduler was worse on the back-to-back workload and was rejected.

*Decision:* expose prefill and opt-in background catch-up with measurements; keep it off by default
and do not package this pair as a catalog composite.

### Rung 4 — shared-base adapter pair with isolated KV
Two specialists as adapters over one loaded base. The base weights are shared, while each adapter
keeps its own KV lineage. Measure adapter-switch overhead, duplicate KV memory, and catch-up prefill
before considering a more aggressive cache policy.

*Blocked on:* adapter load/swap support in the Models runtime and qualified specialist adapters.
*Decides:* whether sharing weights is worthwhile even without sharing KV.

### Rung 5a — base-aligned adapter prefix
Implement an activated-adapter boundary so tokens before the invocation marker execute with the
base model's effective weights. Reuse only the exact base-aligned prefix; after activation, the
adapter owns a separate KV lineage.

*Gate:* token-exact logits or a documented numerical tolerance at every handoff position, followed
by unchanged task quality. A latency win cannot waive correctness.

### Rung 5b — calibrated cross-model KV translation
For pairs with matched KV geometry and identical tokenization, capture unrotated per-layer K/V,
select predictive source layers, and fit an offline per-head mapper. Compare translated handoff with
the target's native prefill over held-out prompts and the five-turn tool-fidelity protocol.

The first screened pair is Qwen3 0.6B to Qwen3 1.7B. Both qualified GGUFs have 28 layers, 16 query
heads, 8 KV heads, 128-wide keys and values, a 1,000,000 RoPE base, and an identical 151,936-token
vocabulary. Their hidden widths differ (1,024 versus 2,048), and `directReuseSafe=false`; the match
authorizes calibration work only.

*Gate:* mapper time must beat native target catch-up, target logits and task quality must remain
inside a declared held-out tolerance, and every response must report that approximate transfer was
used. A pair-specific mapper never becomes a general compatibility claim.

*Preliminary result:* Java-only single-layer, per-head, and raw-correlation top-two ridge screens
were run on 256 calibration and 80 held-out tokens. Keys showed useful structure, but held-out value
cosine remained 0.367–0.576 with relative L2 error 0.819–0.987 across the sampled layers. Those
cheap mappings are rejected. A larger predictive top-k reproduction is required before attempting
to install a complete translated cache.

---

## 4. Protocol — how to judge any rung

Use the **same 5-interaction tool-fidelity protocol** as the chat-model speed ladder (row 5j's
workbench arm), so results are comparable to work already done. Per rung, record:

- tool-call validity rate (well-formed and schema-correct)
- task completion over the 5 interactions
- tokenization, prompt preparation, prefill, **time to first token**, decode, and total wall clock
- prompt-cache read/write tokens for each model-specific session
- cold start, same-model warm turn, first switch, and switch-back measurements
- peak process RSS and backend/native memory, since duplicate KV and adapter state are the tradeoff
- selected model/adapter and switch reason for every turn

**Every arm must be run by us.** Vendor benchmark numbers for Gorilla/xLAM are a reason to look, not
evidence about our regime.

---

## 5. The packaging half (row 5p) — relevant if any rung wins

Neither measured pair cleared both correctness and end-to-end performance gates, so none of the
packaging below is authorized by the current evidence. It remains a design candidate for a future
pair that clears both.

The direction already set: a composite is **pure metadata, lighter than a model marker**, shipped as
`org.modeljars.composite:<codename>` whose **Maven dependencies ARE its parts**, so the dependency
resolver becomes the composition-graph resolver.

Landscape swept 2026-09-04 at docs/source level (discovery search was unavailable, so this is
*found-no-evidence*, **not** verified-absent):

- **LM Studio** `model.yaml` — registry, revisions, package-to-package `base:` references, but for
  *variant substitution*, not composition.
- **vLLM speculators** — a genuine runtime two-model dependency (`verifier.name_or_path` in
  `config.json`), but unpinned and draft/verify only.
- **Ollama** — an undocumented `DRAFT` instruction (v0.30.0+) packing two models into one OCI
  manifest, but it vendors a local file and **explicitly refuses registry references**.
- **BentoML** — `models: [name:version]` pinning, imperative composition, no transitive resolution.
- **KServe / Seldon / Ray / NVIDIA llm-router** — graphs as cluster resources, not packages.
- **OpenRouter `openrouter/fusion`** — multi-model deliberation behind one slug with versioned
  presets; hosted, account-scoped, closed.

**No evidence found anywhere of a recipe package whose declared dependencies are themselves model
packages, resolved transitively, behind one model facade.** Nearest competitive closes: OpenRouter
adding a facade-to-package path, or Docker Compose OCI publishing gaining a resolver.

Licensing stance already decided: **recipes open by default** — a JVM artifact is decompilable, so
the moat is qualification (pinned parts, measured per-platform profiles, scores under named
protocols), not secrecy. Commercial tier is **open topology, licensed parts**.

Seed combinations to try (all read, none measured): the RedHatAI speculator↔verifier matrix
(EAGLE-3 / DFlash across Llama-3.1-8B, Qwen3-8B/14B/32B, gpt-oss-20b/120b).

---

## 6. What would make me drop this

- Rung 1 closes most of the tool-fidelity gap → composition is machinery for its own sake.
- Rung 2 shows no role-specialisation benefit → there is nothing for adapter composition to make
  cheaper.
- Shared weights do not offset adapter-switch and duplicate-KV costs → stop at turn dispatch.
- No useful adapter design can preserve a base-aligned prefix → delete rung 5a, not the correctness
  gate.
- A calibrated mapper misses quality or end-to-end latency gates → reject that pair and retain its
  negative evidence; matching tensor shapes are not a reason to ship rung 5b.

State which of these happened. A negative result here is worth as much as a positive one and costs
far less than building the packaging layer first.
