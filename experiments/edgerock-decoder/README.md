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
- **`backend/ollama-runtime.patch` does not add EdgeRock support**, despite the name. It and
  `backend/ollama-cluster-config.patch` are byte-identical (verified with `cmp`), 79 lines, and
  they do one thing: replace `os.UserHomeDir()` with `os.UserConfigDir()` across `auth/auth.go`,
  `cmd/cmd.go`, `cmd/config/config.go` and `envconfig/config.go`. That is an HPC convenience so
  Ollama keeps its config under `XDG_CONFIG_HOME`, and it touches no inference code at all.
  Ollama serves EdgeRock only because the shipped tarball bundles the patched llama.cpp as its
  runner (`lib/ollama/llama-server`). Stock Ollama cannot serve the architecture, and rebuilding
  the Ollama patch against `d1ed3fd1` as the README suggests buys nothing for EdgeRock on its own.
  **Consequence for qualification:** there is no cheap stock-Ollama comparator. The comparator is
  the patched llama.cpp, through the publisher-reference route the policy already supports.
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

## Where qualification stands: a measured relative-gate failure

Same box (darwin, 12 logical / 6 physical cores), same model file
(`EdgeRock-4B-Q4_K_M`), both arms at `--ctx-size 16384`, both near 850% CPU.

| model | `pure-java` | `rust-ffm` | patched llama.cpp, `-t 8` | ratio on `rust-ffm` |
| --- | --- | --- | --- | --- |
| VibeThinker-1.5B Q4_K_M (control) | 3.10 tok/s | 11.98 tok/s | 22.9 tok/s | 0.523 |
| EdgeRock-4B Q4_K_M | 1.06 tok/s | **4.06 tok/s** | 7.2 tok/s | **0.564** |

The `pure-java` column is not the route to the catalog and never was: 65 of 68 catalog entries
qualify on `rust-ffm` and the three `pure-java` entries are all grandfathered from
`certified-20260902`. The control is what established that -- EdgeRock's `pure-java` ratio of 0.147
is slightly *better* than an unrelated model's 0.135, so that figure is what the Java decoder does
against llama.cpp on this box and is not a cost of the memory layers.

**EdgeRock first gained nothing from `rust-ffm`: 0.93 tok/s against 1.06 pure-Java, slightly worse,
while the control gained 3.9x.** That isolated the cause exactly. `QuantizedProjection` called
`TensorOps.ggufMatmul` directly instead of going through the `GgufBatchedMatrixKernel` seam, and
for a K-quant that runs on the calling thread -- `packedProjectionThreads` only parallelises the
MXFP4 and PTQ1_0 float-activation kernels. Nine projections on each of eight memory layers were
single-threaded while the rest of the graph used the native kernels, so roughly a tenth of the
arithmetic per token was setting the decode rate. Putting them on the seam gives **4.06 tok/s**.

Worth being precise about what that did and did not need: no new Rust. ABI 7 already exposes the
Q4_K/Q5_K/Q6_K projections, grouped dispatch, attention and Gated DeltaNet recurrence this model
uses. The fix was Java plumbing.

0.564 is **not a pass.** The gate measures a same-host comparator through the RAG harness, not a
bare `llama-cli`, and the control's own bare ratio of 0.523 is likewise below 0.80 while models on
that route do qualify -- so the harness-measured comparator is evidently slower than bare
`llama-cli` by a margin that matters. 0.564 is suggestive of qualifying territory and the real
number has not been measured.

The two routes agree numerically, which is a precondition for calling any of this a speedup: layer
3's contribution after one commit is 6.200681824 on the Java route and 6.200961286 on the native
one, 4.5e-5 relative. Not bit-for-bit, because the kernels reduce in a different order.

Both probes (`:backend-java:decodeRateProbe`, `:backend-native:nativeDecodeRateProbe`) require an
explicit context, and the pure-Java one runs `PureJavaBackend.load(Path)` because that is what
`InProcessGenerationClient` calls for the `pure-java` arm.

Two corrections to earlier readings in this experiment, both from measuring instead of inferring:
an earlier EdgeRock figure of 1.9 tok/s rested on a decode token count estimated from the output
text rather than counted, and the clean probe says 1.06; and a single CPU sample of 232% would have
supported a parallelism explanation, where the sustained figure is ~850% on both arms, so the gap
is per-core efficiency.

### A correction to an earlier reading of "the shim"

`backend-native` does **not** embed llama.cpp, Ollama or any other engine (`backend-native
/README.md:5-9`): it accelerates selected GGUF kernels behind a versioned C ABI, while the graph,
tokenizer, KV-cache ownership, sampling and generation stay in Java. So the publisher's llama.cpp
patch is an oracle and a reference, never something to be carried into the shim, and the Java
decoder in this experiment is the production path rather than a stepping stone to one.

### What is left before a listing

1. **Prefill.** EdgeRock is still clamped to one token per step, because batched prefill has no
   memory wiring and no windowed attention. This is the remaining large cost and it lands on TTFT,
   which the relative end-to-end ceiling of 1.50 measures directly.
2. **The real gate numbers**, from the RAG harness with a same-host comparator, not from these
   probes.
3. **The long cases.** Now affordable at 4.06 tok/s where they were hours.
4. **Workload choice.** EdgeRock emits a `<think>` block, and reasoning models have failed the
   grounding gate here before; the catalog's 49 `general` qualifications are the precedent to
   follow rather than a RAG grounding workload.

What has been recovered so far: `QuantizedProjection` was calling the short `ggufMatmul` overload,
which allocates four arrays and re-quantizes the activation per call -- about 288 allocations per
token across nine projections on eight memory layers. Giving it reusable scratch, as
`Qwen35ForwardPass` already does, took decode on the arithmetic case from 214.7s to 99.8s
(**2.15x**) with an unchanged answer.

Not a prefill rate: the 0.8 tok/s figure over 40 tokens is dominated by first-use session
allocation, since the KV caches at ctx 16384 are ~1 GB. It is slower per token than decode, which
is the tell. A real prefill number needs a long prompt, and EdgeRock is additionally clamped to one
token per step.

## The next prefill win: batch the memory read, keep only the stage sequential

Observed while running the long cases: prefill holds about 400% CPU where decode sustains ~850%.
`memoryBatched` walks the batch row by row, and each row issues nine native matvecs at batch size
one, so a 9664-token prefill makes on the order of 700k single-row kernel calls while the backbone
around them runs batched.

Most of that does not need to be sequential, and the reason is specific: **the read sees only the
committed selection, never the pending chunk** (`MemoryRead.read` takes `bank.selection()`). So
within any span of rows containing no commit, every row reads an identical bank. Staging does not
change what a read sees; only `commit()` does.

That splits the layer cleanly:

- Batchable, because they are pure functions of the row's hidden state: `memory.input_norm`,
  `q_proj` with its norm and gate and rope, `o_proj`, the writer SwiGLU
  (`write_gate`/`write_up`/`write_down`), `k_proj`, `v_proj`, and both admission projections.
- Batchable within a commit-free span: the read itself, since the committed selection is fixed.
- Genuinely sequential, and cheap: `bank.stage()`, which is array copies plus the commit trigger.

So the batch is processed in sub-spans delimited by commit boundaries --
`min(remaining, commitChunk - staged())` -- batching everything but the stage inside each span.
With `commit_chunk` at 256 and prefill batches at or below that, most spans are the whole batch.

This matters for the gate rather than for tidiness: TTFT is what the relative end-to-end ceiling of
1.50 measures, and the patched llama.cpp reports 23.1 tok/s on the prompt for this model.

## Provenance for a publisher-reference entry

`RagPublisherReference` is the enum that admits a llama.cpp comparator at all, and it carries one
entry today (`BONSAI2_PTQ1_0`). Its `repository`/`revision` pair works for Bonsai because the
publisher ships a *fork*, so the fork's revision pins the implementation. EdgeRock ships a patch
against upstream instead, so a faithful entry has to pin the patch as well as the commit -- which
is a real schema gap, not a formality: the same upstream commit without the patch does not serve
the architecture.

Measured on 2026-10-30, and identical on the bench pod and locally (both arms must read the same
bytes or `sameWorkload` excludes the comparison):

| item | value |
| --- | --- |
| artifact | `EdgeRock-4B-Q4_K_M.gguf` |
| artifact sha256 | `67c354bcb2abe0cba407fbfe8e7f7cc100dd44458b6fbdfee95ae22f3aaa2c6a` |
| artifact bytes | 2936478592 |
| upstream | `https://github.com/ggml-org/llama.cpp` |
| revision | `62318cd57dc63900ecb153272c0b10f78550855d` |
| `llama-cpp-runtime.patch` sha256 | `6a4c32359d88c84a8cee2a6920847220da61d7931d406dc2c0970f0da9dbf20f` |
| `edgerock-rma.h` sha256 | `4018dd0a6d8368673eb9019df4cbde6721213b0d1ecc2063407bd3eb1ab4a058` |
| `edgerock-rma-graph.inc` sha256 | `35a01ac0aa9c68d40055d844354d785383c5a4a8851efa436970ea6fd00adfbb` |

## Measured on the bench pod: RunPod `cpu3c`, 16 vCPU AMD EPYC 7702P, AVX2, ctx 16384

The laptop was the wrong instrument, and the runbook says so outright -- "do not burden the local
Mac with model downloads, oracle runs, or benchmarks". Two long cases were lost to a 30-minute
cap there before this moved.

| arm | decode | note |
| --- | --- | --- |
| ours, `pure-java` | 3.20 tok/s | |
| ours, `rust-ffm` | **7.14 tok/s** | 16 threads |
| patched llama.cpp | 9.0 tok/s | 16 threads, prompt 20.6 tok/s |

Decode ratio **0.79** against the 0.80 floor, before any prefill work. The earlier 0.147 in this
document came from the laptop's `pure-java` arm and was the wrong arm on the wrong hardware.

Still not a pass, for the reason that has not changed: the gate measures both arms through the RAG
harness with `sameWorkload` matching -- workload, corpus, cases, template, topK, max tokens,
context, **threads**, grounding policy -- and `--publisher-reference` is what admits the llama.cpp
arm. These are probe numbers, not gate numbers.

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
