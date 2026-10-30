# EdgeRock 4B: what serving it actually requires

Scope established from the artifact, not the paper. Every statement below is read from the real
GGUF header via `tools/gguf_describe.py`, or from a path:line in this repository.

## Correcting the earlier record

EdgeRock was triaged on 2026-10-29 and recorded as **blocked, "no GGUF exists"**. That was wrong.
What was actually observed was `HTTP 401` on the two linked GGUF repositories, and a 401 is access
denied, not absence. `hkust-aaron/EdgeRock-4B-GGUF` was created 2026-10-17 and carried "Publish
validated EdgeRock runtime conversion" the same day -- twelve days before that triage.

The half of the triage that still holds: upstream llama.cpp has no `edgerock`. The GGUF says so
itself, `edgerock.runtime = "llama.cpp with included EdgeRock RMA patch"`. So the verdict "we
cannot serve this today" was right; the stated reason was not, and a wrong reason is what stops
anyone revisiting.

## The artifact

`hkust-aaron/EdgeRock-4B-GGUF`, 8 quantisations, `EdgeRock-4B-Q4_K_M.gguf` at 2.94 GB. GGUF v3,
546 tensors, 60 metadata keys.

| key | value |
| --- | --- |
| `general.architecture` | `edgerock` |
| `edgerock.block_count` | 32 |
| `edgerock.embedding_length` | 2560 |
| `edgerock.feed_forward_length` | 9216 |
| `edgerock.attention.head_count` / `_kv` | 16 / 4 |
| `edgerock.attention.key_length` / `value_length` | 256 / 256 |
| `edgerock.attention.window` | 1024 |
| `edgerock.full_attention_interval` | 4 |
| `edgerock.attention.recurrent_layers` | `[1,1,1,0, 1,1,1,0, ...]` |
| `edgerock.ssm.{conv_kernel,state_size,group_count,time_step_rank,inner_size}` | 4 / 128 / 16 / 32 / 4096 |
| `edgerock.memory.{capacity,admission_rank,commit_chunk,writer_intermediate}` | 1024 / 256 / 256 / 2560 |
| `edgerock.rope.{freq_base,dimension_count,dimension_sections}` | 1e7 / 64 / `[11,11,10,0]` |
| `edgerock.context_length` | 262144 |
| tokenizer | `gpt2` BPE, `pre = qwen35`, vocab 248320 |

32 blocks = **24 recurrent + 8 full-attention**, consistent with the mask and
`full_attention_interval = 4`. The 8 memory layers sit on the attention layers.

Licence: EdgeRock Model License 1.0. Commercial use, SaaS, redistribution and fine-tuning are
granted royalty-free; the single restriction forbids baking weights into non-reprogrammable
silicon (mask ROM), which no software runtime does. Attribution and full-licence inclusion are
required, and the supplement may not be presented as an Apache-only grant.

## The backbone is one we already serve

This is the finding that changes the cost. EdgeRock's `config.json` names
`fixed_memory_parent: "original_qwen3_5_text"`, and the tensors bear it out. Of its **37 distinct
tensor roles, 22 are already read** by `backend-java/.../qwen35/` -- the whole gated-delta-net
recurrence (`ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt.bias`, `ssm_norm`, `ssm_out`),
the whole attention block, the dense feed-forward, and both norms.

`Qwen35Config.fromMetadata` (`Qwen35Config.java:137`) reads every key EdgeRock publishes, because
it already resolves them under an `architecture + "."` prefix for three families
(`ARCHITECTURES = List.of("qwen35", "qwen35moe", "qwen3next")`, `:65`). EdgeRock's absent
`expert_*` keys default to 0, which is the dense path.

Vectors needs less than expected: `PanamaVectorUtilSupport` already provides
`causalDepthwiseConv1dSilu` and `sigmoidAndScaledSoftplus`, the two GDN primitives, both hardened
for tier-stability on 2026-10-29.

## The gap, measured

Fifteen tensor roles, all the memory module, 8 layers each:

```
memory.input_norm.weight          memory.q_proj.weight      memory.q_norm.weight
memory.k_proj.weight              memory.k_norm.weight      memory.v_proj.weight
memory.o_proj.weight              memory.read_gate_weight   memory.read_gate_bias
memory.read_scale                 memory.admission_down.weight (2560x256)
memory.admission_up.weight (256x4) memory.write_down_proj.weight
memory.write_gate_proj.weight     memory.write_up_proj.weight
```

Shapes imply the mechanism: a low-rank admission scorer (2560 -> 256 -> 4), a gated read over a
1024-slot store with 16 read gates and per-gate scales, and a writer at
`writer_intermediate = 2560` committing in chunks of 256. The paper (arXiv:2610.10118) is the
authority for the exact recurrence; the shapes constrain it but do not determine it.

Three further gaps that are not tensors:

1. **Sliding-window attention at 1024.** `edgerock.attention.window` is not a key
   `Qwen35Config` reads, and qwen35 has no windowed attention path.
2. ~~**`ssm_ba.weight`.**~~ **Resolved, not a gap.** `Qwen35Weights.java:198` decides with
   `file.hasTensor(prefix + "ssm_ba.weight")` -- "by what the file carries, not by the
   architecture name, because the fused form is a tensor-layout choice rather than a different
   computation". EdgeRock's split `ssm_alpha`/`ssm_beta` is the already-supported form.
3. **mRoPE `dimension_sections [11,11,10,0]`.** The multimodal rope layout. Text-only inference
   may be able to ignore it; that must be verified rather than assumed.

## Why this is worth doing, in one line

A model whose state is worth keeping between requests is the first catalog candidate where a
long-lived JVM is an architectural advantage and not a packaging convenience: 24 recurrent layers
of SSM state plus a 1024-slot memory at 262144 context, held off-heap across turns rather than
rebuilt per call. The measured TTFT gap that rejected 20 quality-passing models on 2026-10-29 is
the same lever.

## Qualification: the two-arm rule still applies, via an existing policy

llama.cpp and Ollama cannot serve `edgerock`, so there is no stock comparator. The catalog already
solves this: `production-rag-publisher-reference-v1`
(`modeljars/tools/rag-publisher-reference-gate.mjs`) qualifies against a **pinned publisher fork**
-- it does so today for Bonsai against `PrismML-Eng/llama.cpp@e8fd209c`, with hash-verified reports
and exactly one comparator.

EdgeRock fits that mechanism directly, and ships its own reference runtime in the GGUF repository
(`edgerock-native-linux-sm90.tar.gz`). Pin it, run both arms on the same artifact sha, and the
two-arm discipline is preserved. The gate currently hard-codes Bonsai's id and artifact, so it
needs generalising to a table of (model, pinned reference) pairs -- contained, not a new protocol.

## State, 2026-10-30

Measured here unless marked otherwise.

**EdgeRock-4B-Q4_K_M runs through our decoder.** Greedy from `"The capital of France is"`,
16 tokens, library defaults: `" Paris. Paris is known for its historical landmarks and
attractions. One of the most"`. Logits finite at every step; vocabulary 248320, resolved from
`tokenizer.ggml.tokens` because the file publishes no `vocab_size` under either prefix.
Reproduce with `:backend-java:edgeRockCompatibilityTest
-Dmodels.fixtures.edgeRock4bGguf=<path>`.

**That run is not evidence the memory works.** A memory module returning zeros leaves a plain
Qwen3.5 hybrid, which also produces fluent text, so the continuation above cannot distinguish the
two and must not be cited as if it could. The second test in that task drives one real memory
layer with real weights and measures L1 = 5.94 for layer 3's contribution after one commit.

**The anchors contribute exactly nothing before the first commit**, and that is correct, not
broken: the read precedes the write, the read sees only the committed selection, and `commit()`
fires when the pending chunk reaches `commit_chunk` = 256. The first version of that test drove a
single token and would have reported a dead layer that was merely early. Consequence for
qualification: RAG prompts here run 171-379 tokens, so the feature is barely exercised at all, and
a flat result from it is "no data", not "no effect". The publisher's own long-memory cases, at 9664
and 10462 input tokens, are the ones that exercise it.

### Why this became a fourth name rather than a new decoder

The published GGUF is **dual-keyed**: a complete `qwen35.*` metadata set beside its own
`edgerock.*`, identical values, and every one of its 546 tensors carries a name
`Qwen35Weights` already resolves. The publisher's own llama.cpp patch does the same thing from the
other side -- `patch_llama.py` maps `edgerock` to `LLM_ARCH_QWEN35` and extends
`src/models/qwen35.cpp`. Two independent implementations reached the same decision.

Read off the artifact, not assumed: `attn_q` is `[2560, 8192]`, twice the query width, because
`attn_output_gate` packs the gate into it; `attn_gate` exists only on the 24 recurrent layers;
there is no `output.weight`, so the head ties.

### Corroborated independently

Our snapshot refusal was reached before reading the patch, and the patch agrees: it forces
`COMMON_CONTEXT_SEQ_RM_TYPE_FULL` for `edgerock`, because its context cannot be partially removed.
`captureLinearState` therefore refuses on a model with anchors -- a snapshot carries convolution
and recurrent state only, so restoring one would rewind the recurrence while the bank kept anchors
written past the restore point. `rewind()` is unaffected: it clears the banks and replays.

### A defect this work found in our own reader

`attention.window` was read with `getUint32().orElse(0)`, and `getUint32` strictly filters on
`Uint32Value`. A width stored as int32 would have been dropped to 0 and served as **full attention
with nothing reported**. Absent and present-but-unreadable are now different answers.

## The reference, and what it costs to use

The GGUF repository ships its own backend, and the licences are permissive: the patch is
Apache-2.0 (`LICENSE-CODE`), llama.cpp MIT (`backend/llama-cpp-LICENSE`). Rule 6 is satisfied --
and nothing from it is copied into our runtime; it is used only as an external oracle.

- `backend/llama-cpp-runtime.patch`, against llama.cpp
  `62318cd57dc63900ecb153272c0b10f78550855d`, plus `backend/edgerock-rma.h` (to `src/`) and
  `backend/edgerock-rma-graph.inc` (to `src/models/`). All 9 hunks apply cleanly at that commit
  and the CPU build succeeds; verified here on 2026-10-30.
- The prebuilt `edgerock-native-linux-sm90.tar.gz` is Linux x86-64 + CUDA SM90 only, so any other
  platform must rebuild. Their published receipts are CUDA runs with `--n-gpu-layers 99`, so their
  numbers are GPU numbers and are **read, not ours**.
- `receipts/*.json` carry the exact server command, four cases with expected answers, outputs and
  timings. Two are long-memory cases at 9664 and 10462 input tokens. `Modelfile.Q4_K_M` gives the
  template: ChatML with a forced `<think>` opener, `temperature 0`.

## Order of work

1. ~~Prove the config parses the real header with `edgerock` admitted, and the loader's tensor
   expectations against the real tensor list.~~ Done; `ssm_ba` answered -- the split
   `ssm_alpha`/`ssm_beta` form is the one already supported.
2. ~~Refuse rather than mis-serve.~~ Superseded. The memory module exists, so the blanket refusal
   is gone and three narrower ones replace it: the graph will not construct unless anchors are
   present exactly when the architecture declares them; batched prefill clamps to one token
   because it has no memory wiring; grouped decisions are refused because the bank is not
   branch-scoped.
3. ~~Implement the memory module, with a parity test against the reference.~~ Done at layer
   scope -- `EdgeRockMemoryLayerParityTest` against a dumped reference layer. **Full-model parity
   is still open.**
4. ~~Sliding-window attention.~~ Done, as a shifted base offset into the position-contiguous key
   cache, so `attendHead` is unchanged. mRoPE is **not** implemented; see below.
5. **Pin greedy output against the patched llama.cpp** built above. This is the correctness gate
   and it is the next thing to run.
6. **Qualify**: candidate `pure-java` against that same build as the pinned publisher reference,
   plus the 9/9 library-defaults smoke, then generalise
   `rag-publisher-reference-gate.mjs` off its hard-coded Bonsai pair to a table of
   (model, pinned reference) pairs.

## Still unverified

- **mRoPE.** The file publishes `rope.dimension_sections [11, 11, 10, 0]` and the reference config
  sets `mrope_interleaved`; we apply plain NeoX over `rope.dimension_count`. For text-only input
  the three sections are expected to carry the same position and collapse to exactly that, which
  is why the model loads rather than being refused. Coherent output over 16 tokens is weak support
  and nothing more -- a rope error can stay invisible at short context. Step 5 settles it.
- **Full-model parity.** Layer-scope parity passes; whole-model does not exist yet.
- **Prefill throughput.** EdgeRock is clamped to one token per step. Real cost, not yet measured.

## What this document does not claim

No number here is measured by us. The paper's figures -- and the 10-31x and 99.1% KV-reduction
claims seen in its abstract -- are **read, not reproduced**, and none may be repeated as ours. The
decision to build is justified by the architecture and the licence, not by a self-reported score.
