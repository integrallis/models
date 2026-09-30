# Changelog

All notable changes to models are documented here.

## [Unreleased]

### Corrected

- **The 0.3.48 and 0.3.49 qualification claims overstate model quality.** Both releases state that
  fourteen models scored 1.000 on every quality metric. `correctAnswerRate` scores the **pipeline**:
  under the grounding policy these runs used, an answer that fails citation screening is replaced with
  text extracted from the retrieved document, and the replacement is what is scored. Across all 14
  models and 378 attempts, **69% were answered by `EXTRACTIVE_FALLBACK`**, 20% by the model, 11% by
  retrieval abstention.

  Applying `RagProductionQualificationPolicy` -- this repository's own gate, at a 1/3 model-answer
  floor and 90% correctness among those answers -- **five of the fourteen qualify**: `lfm2`,
  `qwen3-30b-a3b-instruct-2507` and `qwen3-coder-30b-a3b` (`qwen3moe`), `kat-coder-v2.5` (`qwen35moe`),
  and `gemma-4-E4B`. `qwen3next`, `deepseek2`, `gemma3n`, `gpt-oss` and `mistral3` have **no qualifying
  model**.

  Nothing in the released artifacts is affected, and no catalogue record was ever contributed from
  these runs. The full correction, including the per-model table and the raw model output, is in
  `release-evidence/CORRECTION-2026-09-29.md`.

- **The summary now publishes what makes that visible.** `RagBenchmarkSummary` gains
  `rawCorrectAnswerRate`, `modelAnswerRate`, `modelAnswerCorrectRate` and `extractiveFallbackRate`, and
  the CLI prints the last three beside `correct=`. They existed only inside `runs[]` before, which is
  why a summary could report a perfect score for a model that contributed nothing and no reader had
  anything to contradict it. `RagStatisticsTest` pins that exact case.

### Fixed

- **gemma3n never applied its final logit softcap.** The reference's graph ends in SCALE, TANH, SCALE
  around the output projection -- `30 * tanh(logit / 30)` -- and gemma3n's published header declares no
  softcapping key, so the reference's hparams default of 30 applies. Our decoder applied nothing, while
  gemma4 has applied its own since it shipped. The formula now lives in `TensorOps.softcap` with gemma4
  delegating to it, so there is one implementation.

  Confirmed numerically against a dump of the reference graph: raw `-15.6992` leaves as `-14.4074`, and
  `30 * tanh(-15.6992 / 30)` is `-14.41`.

  **This is not why gemma3n produces corrupted text.** The transform is monotonic, so it cannot change
  which token a greedy decode picks, and the generated text is byte-identical before and after -- which
  is also why nothing noticed its absence. It matters for any consumer reading a probability, a
  temperature or a logprob from these logits.

- **GELU could read past the end of the tanh table.** `ArrayIndexOutOfBoundsException: Index 65537 out
  of bounds for length 65537`, from `TensorOps.tableTanh` inside `gelu`, which killed a gemma3n
  qualification run 456 seconds in. The table holds `TANH_TABLE_SIZE + 1` entries so the last
  interpolation has a neighbour to read, but a value one float ulp below the table's limit still scales
  onto that last slot -- adding `10.0f` rounds it to `20.0f` and `20.0f * scale` is exactly the size --
  so the read went one past the end.

  Shared code: any architecture whose feed-forward uses GELU could reach it. No existing measurement
  changes, because the only inputs affected previously threw. The test reproduces the exact exception,
  and hits the boundary directly -- a sweep through plausible activations was written first and passed
  against the crashing version, because the boundary is one ulp wide.

- **A reasoning trace was scored as if it were the answer; grounding policy is now v21.** Screening a
  `<think>` block fails on the block's own terms -- it reasons aloud, so it states things the retrieved
  documents do not support, the whole output is rejected, and the answer that followed is never
  examined. Several models answered correctly after a trace and were reported as contributing nothing.
  A closed `<think>`, `<thinking>` or `<reasoning>` block is now removed before screening, with
  `rawText` still carrying the full generation. `POLICY_ID` becomes v21 because this changes which
  decision a completion receives, and records written under v20 keep their meaning rather than being
  reinterpreted.

  An explicit refusal for *unterminated* traces was written and then removed: probing the policy showed
  the decision is `EXTRACTIVE_FALLBACK` with or without it on every input tried, because ordinary
  screening already rejects such text. Code whose effect cannot be demonstrated is a claim, not a
  safeguard.

- **The output token cap defaults to 256, not 64.** The campaign never chose 64 -- it inherited the CLI
  default, and at 64 most attempts were truncated. A grounded answer here is one short sentence plus a
  bracketed source id, and a model that reasons first needs room for both, so the old default could not
  express a passing answer for a whole class of models. 256 is a floor that lets them finish, not a
  tuned optimum. Every report records the cap it ran with, so older records stay interpretable.

- **DeepSeek-V2 models were prompted with the DeepSeek-V1 format.** The harness rendered
  `### Instruction:` / `### Response:`, while DeepSeek-Coder-V2-Lite-Instruct's own
  `tokenizer.chat_template` in the published GGUF uses `User: ` / `Assistant: `. Shown the `###`
  markers, the model continued the pattern and emitted `###` for entire completions -- which read as a
  broken decoder until the template was checked. A new `deepseek-v2` template fixes it; `deepseek` is
  left exactly as it was, because the V1 coder models are already qualified against those markers with
  their greedy oracles pinned on them.

  **The template was not the cause of its garbage output, and this entry originally implied it was.**
  Re-run on the corrected template with a 256-token cap, the model emits pure newline tokens for the
  whole completion: `modelAnswerRate` 0.000, `truncatedAnswerRate` 1.000. `###` under one prompt and
  `\n` under another are the same degeneracy. The `deepseek2` decoder is defective on real weights; the
  template fix is right and independent.

### Notes

- **`deepseek2`, `gemma3n` and `qwen3next` emit incoherent output that prompting does not explain.**
  `deepseek2` emits only newlines once correctly prompted, which is the strongest evidence of the three.
  `gemma3n` produces corrupted text at 0.0 raw correct; `qwen3next` degenerates into repetition at
  11.1%. The `gemma` template used matches gemma3n's own turn markers read from the published GGUF, and
  `chatml` matches Qwen's, so a decoder defect is the leading explanation -- stated as a suspicion, not
  a diagnosis. Evidence in `release-evidence/CORRECTION-2026-09-29.md`.

### Notes

- **`gpt-oss` from a GGUF is qualified on published weights**, measured after 0.3.49 shipped and so not
  claimed by it. `unsloth/gpt-oss-20b-GGUF` Q4_K_M (11.6 GB), on `models@0.3.49+rerun11-fbe951608665`:
  **1.000 on all six quality metrics over 27/27 attempts with zero failures**, 63.7 s ttft p50 and
  361.6 ms per token, 11.6 GB peak RSS. It ran on the `gpt-4o` o200k pre-tokenizer split added in
  0.3.48, which until this run had no catalogue model exercising it.

  That leaves `mistral3` as the only one of the eight new architectures with no published-weights run
  behind it.

  **The latency is the honest headline.** 361.6 ms per token is the parallel MXFP4 kernel from 0.3.49
  at work -- the same model on 0.3.48 sat at 8% CPU and had not finished a run after 100 minutes,
  where this completed in 55 -- but it is still several times slower per token than a K-quant model of
  comparable size, because the second half of the MXFP4 gap is untouched. The K-quant kernels quantize
  the activation to Q8 and reduce in integer arithmetic; MXFP4 still multiplies in scalar floats. That
  kernel is the next piece of work, and it has to be proven identical to the scalar one before it can
  be adopted.

## [0.3.49] - 2026-09-29

### Fixed

- **MXFP4 projections use every core instead of one.** Every other quantization hands its whole matrix
  to vectors-core's `gguf*BatchDotProduct`, which parallelises above a one-mebibyte threshold. MXFP4
  alone looped rows on the calling thread, so a model whose weights are MXFP4 decoded on a single core
  while the rest of the catalogue used all of them. Found on a 16-vCPU qualification worker running
  gpt-oss from a GGUF, sitting at a flat **8% CPU -- about one core** -- where K-quant models pegged the
  box.

  **Measured 5.15x on 12 cores** on the shape gpt-oss actually uses (one expert, 2880x2880, 4.2 MiB of
  weights): 11.60 ms to 2.25 ms per projection. Not 12x, because the kernel dequantizes as it reads and
  is bandwidth-bound rather than compute-bound.

  Two models in the catalogue touch MXFP4: gpt-oss from a GGUF, where every expert is MXFP4 and the
  effect is the whole model, and Qwen3-Next, where only the shared-expert gate and up are, so the
  effect is a slice of each layer. **No published number is retracted.** `Qwen3-Coder-Next`'s 241 ms
  per token was measured on exactly the code that shipped in 0.3.48; it will be faster on this
  release, and re-measuring it is a new measurement epoch rather than a correction.

  Rows are split, never reductions. Each output element is its own dot product over its own row, so
  thread count cannot move a bit -- asserted bit-identical to the serial path rather than close, which
  in a backend that pins float reduction order on purpose is the only assertion worth making.

  The first version of this crashed instead of being slow, which the tests caught: a `MemorySegment`
  from a confined arena is readable only by the thread that allocated it, so handing its rows to a pool
  thread throws `WrongThreadException` from inside the dot product. Parallelism is now gated on the
  segment actually being reachable from another thread -- the same property the execution planner
  already gates thread sharing on -- and a confined-arena projection stays on the calling thread at any
  size.

  The scalar float activation is unchanged and still deliberate: the K-quant kernels quantize the
  activation to Q8 and reduce in integer arithmetic, and an MXFP4 kernel that does the same has to be
  proven identical to this one first. This release fixes the thread factor only.

## [0.3.48] - 2026-09-29

### Changed

- **Fused grouped-query attention is now available to every grouped-query model, off by default.**
  The fused kernels read each cached K and V row once per KV head instead of once per query head,
  which on a decode step bound by cache bytes is one pass instead of `groupSize`. They were always
  parameterised by group size, head stride and scale and were never Granite-specific.

  **It is off by default because it is not token-preserving, which we measured rather than assumed.**
  The fused path agrees with the head-by-head loop to 1.4e-6 relative on the attention output, flat in
  context length (3.8e-7 at 64 cached rows, 1.4e-6 at 1024 and 4096) — but greedy decoding is a
  discrete argmax, so a perturbation that small still flips a token whenever the top two candidates
  sit inside it. Swapping only the attention kernel under Granite on the RAG workload changed **2 of 9
  generated answers from byte-identical prompts** (a comma became a semicolon; one clause was
  reworded) at unchanged `correctAnswerRate`, `retrievalRecall` and `abstentionAccuracy`. The Rust
  native grouped-attention kernel diverges less — 7.3e-7 relative, bit-identical at a single cached
  position — and still flipped those same two answers.

  **And the gain does not justify that cost.** Measured on a group-8 model (qwen2.5-coder-3b,
  1838-token context, four counterbalanced arms): **2.89 vs 2.74 tok/s decode, a 5.5% gain**. The two
  `true` arms agreed to 0.3% and each path's checksum was bit-reproducible, so the measurement is tight
  — it is the *prediction* that was wrong. A bytes-per-token model said 1.45x, assuming the head-by-head
  path's `groupSize` re-reads of each K and V row all come from DRAM. They do not: one group's KV
  working set is under a megabyte per layer at this context, so it is L2/L3 resident and fusion saves
  cache traffic rather than memory traffic. The Rust native grouped-attention kernel is the better
  candidate if an attention change is ever adopted — 15% decode measured on Granite, and half the
  divergence — but it flips the same answers.

  A qualification record is a claim about the bytes a model produced, and a published one cannot be
  retracted. So each architecture keeps the route its records were measured on: Granite the native
  kernel with the Java fused loop as its fallback, everything else head-by-head. Adopting a faster
  route is a **new measurement epoch** — every pinned greedy oracle re-run and the tier band
  re-derived on it — not a default flip. `models.purejava.fusedGroupedAttention=true` asks for it by
  name, and the choice is recorded as the `fused-grouped-attention` optimization decision so a record
  always says which route produced it.

- **New performance tier `PERFORMANCE_REDUCED`, split out of `OFFLINE`.** `OFFLINE` was conflating
  "too slow for this SLO" with "cannot be used at all", and the difference turned out to be most of the
  catalogue. Measured on the small/medium qualification campaign: of 28 models tiered `OFFLINE`, **every
  one** passed retrieval recall, citation recall, abstention accuracy and correct-answer rate — none
  were broken, they were slower than the USABLE latency bound. Five were rejected on the tail alone,
  with a ttft median under 2000ms at 95% confidence and only 3–7 of 27 requests over it;
  `gemma_3_4b_it` missed by **90ms (4.5%)** while holding better than 2x margin on retrieval, tpot and
  end-to-end, and landed in the same tier as a model 3x over.

  A `PERFORMANCE_REDUCED` model is shippable with a slower latency expectation, which for a
  small/medium-first catalogue is a product category rather than a failure. Quality failures still
  outrank latency, so a model that is both slow and ungrounded reports `FAILED_QUALITY` — returning
  "works, just slower" for a model producing wrong answers would be worse than the old tier was.

  `OFFLINE` is retained and no longer produced: 316 certified records carry it and a published
  qualification cannot be retracted. Those records are unaffected — `performanceTier` is a stored field,
  not a recomputation — and they stay re-readable, since an `OFFLINE` record whose quality gates passed
  is a `PERFORMANCE_REDUCED` record under the current taxonomy, derivable from the stored summary
  without re-running anything.

- The `fused-grouped-attention-not-wired` performance cliff is now
  **`fused-grouped-attention-disabled`**. It used to mean the fused kernel served Granite only; it now
  means the faster route was available and not taken. The id changed rather than being kept stable
  because records carrying the old one were written when it meant something else —
  `benchmark-results/2026-09-16-loader-cliffs/README.md` is one of them.

### Fixed

- **Four defects that only a real published model could expose, each found on a fleet worker after the
  weights had already been downloaded.** All four were invisible to the unit suite for the same
  reason: the fixtures were uniform on the axis that broke.

  - The execution planner crashed with a `NullPointerException` on the first mixture-of-experts model
    to reach the fleet. `ModelTopology` decided thread-shareability by dereferencing the dense
    feed-forward's first matrix, which is null on a routed layer. It now walks every segment a layer
    actually owns. The regression test needs an auto arena, because with a confined one the
    shareability check short-circuits false before it ever reaches the null.
  - Gemma 4 E4B would not load: its shared-expert hidden dimension **varies by layer** and the loader
    read one key for all of them. Scratch buffers are now sized from the maximum and indexed per
    layer — kept deliberately separate, since a buffer sized from one layer's width is an
    out-of-bounds write rather than an exception when another layer is wider.
  - Gemma 4 E4B, again, was rejected with `Tensor not found: blk.24.attn_v.weight`. It declares
    `shared_kv_layers=18` over 42 layers, so layers 24-41 attend to an earlier layer's cache and carry
    no `attn_k`, `attn_k_norm` or `attn_v` of their own. The forward pass already skipped those
    projections for such a layer; the loader was demanding tensors nothing would have read. Every
    fixture had given all three to every layer *because* the loader demanded them, so the shape a real
    file has could not be expressed in a test.
  - `ggufMatmul` handled F16 only in its batched form, which a decoder running one token at a time
    cannot reach. Gemma 4 E4B publishes `per_layer_model_proj` as F16 where E2B publishes BF16, and
    that projection runs per token.

- The same model produced **different output text on different machines**. Every vectorised float
  reduction in the model path sized itself from the host — `SPECIES_PREFERRED` capped by
  `vectors.maxBits` — and a float reduction's last bits are decided by how many partial sums it keeps
  and in what order it folds them. In a retrieval score that is invisible. In a transformer it
  decides which token is selected, so it changes the generated text. `PinnedReduction` pins the shape
  at eight lanes and now backs `rmsNorm`, the F32 row score, the batched F32 path and the attention
  reduction. It is **bit-identical to what it replaces at the 256-bit default**, so x86 hosts running
  the default see no change; ARM and narrower or wider configurations do.

- CUDA K-quant and attention kernels are now bit-exact against the CPU control, and G1 token parity
  passes on device (640 tokens identical, both stages routed). Three defects, all the same mistake of
  validating a device kernel against a plausible reference instead of the path the gate compares it
  to: Q6_K folded the way the scalar path folds rather than the way the Panama control does; `expf`
  approximated the exponential instead of transcribing the CPU's own polynomial, argument reduction
  and rounding; and the attention reduction folded in a different order.

- Completed deadlines were never removed from the routing timer queue, so a long-lived
  `RoutingExecution` accumulated them.

- Spring AI did not surface budget overruns as terminal accounting failures.

- LangChain4j applied SDK parameters around provider public APIs rather than through them.

### Added

- **Eight decoder architectures the pure-Java backend could not load before: `lfm2`, `qwen3moe`,
  `mistral3`, `qwen35moe`, `qwen3next`, `gemma3n`, `deepseek2`, and `gpt-oss` from a GGUF.**

  Six of them are qualified on real published weights, measured here on a 16-vCPU m6a.4xlarge. Every
  model below scored `correctAnswerRate`, `retrievalRecall`, `citationRecall`, `citationPrecision`,
  `factCoverage` and `abstentionAccuracy` of **1.000 over 27/27 attempts with zero failures**:

  | model | arch | size | ttft p50 | tpot p50 |
  |---|---|---|---|---|
  | `LFM2-1.2B-Instruct` Q4_K_M | `lfm2` | 0.7 GB | 11.1 s | 65.6 ms |
  | `Qwen3-30B-A3B` Q4_K_M (3 variants) | `qwen3moe` | 18.6 GB | 3.9 s | ~106 ms |
  | `Ornith-1.0-35B` Q4_K_M | `qwen35moe` | 21.2 GB | 13.0 s | 87.2 ms |
  | `KAT-Coder-V2.5-Dev` Q4_K_M | `qwen35moe` | 21.4 GB | 20.1 s | 130.8 ms |
  | `Qwen3.5-35B-A3B` Q4_K_M | `qwen35moe` | 22.0 GB | 15.4 s | 101.0 ms |
  | `Qwen3-Coder-Next` Q4_K_M | `qwen3next` | 48.5 GB | 38.6 s | 241.0 ms |
  | `DeepSeek-Coder-V2-Lite-Instruct` Q4_K_M | `deepseek2` | 10.4 GB | 13.6 s | 190.7 ms |
  | `gemma-3n-E2B-it` Q8_0 | `gemma3n` | 4.8 GB | 25.8 s | 150.8 ms |
  | `gemma-4-E2B-it` Q4_K_M | `gemma4` (already supported, see Fixed) | 3.4 GB | 3.0 s | 58.0 ms |

  `Qwen3-Coder-Next` is an 80B-class model answering correctly on all 27 attempts inside a 64 GB box at
  39.2 GB peak RSS — the largest artifact this backend has qualified.

  `mistral3` and `gpt-oss`-from-GGUF are **implemented and verified against unit-level scalar
  references only; no published-weights qualification yet**, so nothing in the catalogue claims them.

  `deepseek2`'s run exercises multi-head latent attention, 64-expert unnormalised routing and the YaRN
  variant that moves the mscale into the softmax; it is also the first real model to use the
  `deepseek-llm` pre-tokenizer split added here. `gemma3n`'s exercises AltUp's four parallel residual
  streams, LAuReL, shared key-value layers, and a softmax scale of 1.0 rather than 1/sqrt(head_dim).

  What each needed, since "supports the architecture" hides the work: `lfm2` interleaves attention
  with gated short convolutions over a shift register; `qwen3moe` and `qwen35moe` add routed experts
  read as strides over one stacked tensor; `qwen3next` adds a gated delta network whose beta and alpha
  arrive interleaved in a single `ssm_ba` tensor, grouped per key head; `gemma3n` carries four parallel
  residual streams with AltUp predict/correct plus LAuReL low-rank residuals; `deepseek2` uses
  multi-head latent attention with a YaRN variant that moves the mscale into the softmax; `gpt-oss`
  reads MXFP4 experts as one interleaved blob where its safetensors form splits blocks from scales.

  Expert routing is shared between architectures and comes in two forms that are not interchangeable:
  renormalised (softmax over the selected experts, weights sum to one) for `qwen3moe` and `qwen35moe`,
  unnormalised (softmax over all experts, weights used as they are) for `deepseek2`, which publishes
  no `expert_weights_norm` at all. Routed models run the expert branch one token at a time, since
  expert selection differs per token and a batched matmul cannot serve rows that need different
  weights.

- **Pre-tokenizer splits for tekken (Mistral 3), the o200k family, single-digit GPT-2, deepseek-coder,
  deepseek-llm and hunyuan.** A GGUF carries a pre-tokenizer *name*, not its regex, so an unrecognised
  name silently falls back to a split that is wrong for the model: it loads, generates fluent text, and
  scores worse for no visible reason. Several are sequences of expressions applied in order rather than
  one alternation, because the reference applies them that way and joining them changes the split —
  measured, not assumed: joined against sequential over 32 strings differed on 4 for deepseek-coder,
  every one CJK, Hangul or leading whitespace. Every pattern is transcribed from the model's own
  `tokenizer.json` or the reference's corresponding case, with the source named in the javadoc.
  o200k matters beyond gpt-oss: `phi3` declares `gpt-4o` on a byte-level BPE vocabulary, so it would
  have been scheduled and tokenized with no word-boundary split at all.

- Sharded HuggingFace models can be qualified. The CLI required a single `model.safetensors`, which was
  the only thing standing between a sharded bundle and a run — `gpt-oss-20b` ships three shards and an
  index. The artifact digest hashes **the shards**, in index order, rather than the index: an index holds
  a tensor-name-to-shard map and no weight content, so hashing it would let two different models with
  the same layout write the same provenance digest.

- `greedyOracleSweepSmall` and `greedyOracleSweepLarge` run the pinned llama.cpp greedy oracles as
  Gradle tasks, split because one 7B-class fixture can run past 45 minutes and a single task reports
  nothing for most of an hour. The `fusedGroupedAttention` property is forwarded explicitly: its
  default is off, so without forwarding the sweep would exercise the very baseline the oracles were
  recorded on and pass while proving nothing.

- Routing execution controls: per-request and shared spending budgets, deadlines, token bounds and
  cancellation, via `RoutingExecution`, `RoutingExecutionOptions`, `RoutingBudget`,
  `RoutingTokenBounds`, `RoutingUsage`, `RoutingCancellationToken` and
  `RoutingBudgetExceededException`, surfaced on the Spring AI and LangChain4j routed adapters.
  `RoutingBudget` is a thread-safe account in one application-selected currency, reusable across a
  session, tenant or process; candidate prices and the account must agree on currency, and **usage
  the provider does not report is charged at the reserved bound**. It is an in-process ledger and not
  a payment API. Existing constructors keep their unbudgeted behaviour.

- `TaskClassifier.local()` declares whether classification, including every embedding call, stays in
  this process. It defaults to `false`, so **unknown is treated as remote**. Wrapping a
  `PretrainedTaskClassifier` with `TaskClassifier.local` is only correct when the supplied embedder
  is also local: the index establishes nothing about the embedding client's data boundary.

- A task index records the embedding prefixes its exemplars were built with, and
  `PretrainedTaskClassifier` applies the query side itself, so callers pass bare queries. Supplying
  one side without the other is refused rather than half-applied. `--document-prefix` and
  `--query-prefix` select them on `task-index build`, and `task-index evaluate` gains `--per-item`
  for paired comparisons between two indexes.

- `ToolSpecRetriever` applies a document prefix when indexing tool descriptions and a query prefix
  when selecting. It embeds both sides, so it is the easiest place here to get an instruction-tuned
  embedder wrong, and getting it wrong is silent: selection still returns tools, just less accurately
  than the model allows. One prefix without the other is refused.

- Bench gates for the GPU campaign and router evaluation: `CudaParityRun`, `CudaDecodeRun`,
  `GreedyDecode`, `CudaRoutingRecorder` and `RouterSelectionEvaluationCli`.

### Changed

- The bundled router task index is rebuilt and now embeds both exemplars and queries with
  EmbeddingGemma's classification prefix, `task: classification | query: `, **worth +4.2 points of
  held-out accuracy**: 0.9459 (455/481) against 0.9044 (435/481) unprefixed, McNemar exact
  p = 0.0012 on 36 discordant pairs. Gains concentrate where intent rather than surface form
  separates the classes — extraction +0.185, reasoning +0.119, summarization to 58/58 — while code,
  math, translation and near-ceiling sql move by at most one prompt.

  The arm that lost is the more useful result. Embedding exemplars as titled documents and queries as
  search queries, which is the obvious pairing to reach for, measured **0.8399 — 6.4 points below
  using no prefix at all** — and collapsed chat from 41/50 to 17/50. The two sides land in regions
  the model keeps apart. Two plausible prefixes, one worth +4.2 and one worth −6.4, and the GGUF
  carries no template metadata to arbitrate: `BundledTaskIndexTest` now pins both prefixes, including
  the trailing space that `java.util.Properties` would otherwise eat. Pre-registration, decision rule
  and per-task results in `docs/findings/embeddinggemma-task-prefix-ab.md`.

  Not separated: which half of the prefix earns the gain, since the arm moved `task: classification`
  and the `query: ` scaffold together. Not measured: whether any of this transfers to Nomic or E5, or
  the per-query cost of the seven extra tokens.

- Consume Vectors 0.1.24. Its manifest grew from 164 to 200 bytes to carry an embedding recipe
  reference, and the version is an exact match rather than a floor, so **0.1.24 cannot open an index
  written by 0.1.23**. The bundled index is migrated accordingly. `models-router/CORPUS.md` gains the
  archive packaging step, which was previously undocumented — the documented rebuild writes a
  collection directory, but what ships is a zip of its contents with the generation directories at the
  archive root.

- Align `jackson-datatype-jdk8` to 2.21.7, the last Jackson module still on 2.21.4.

- Corrected a `backend-cuda` contract table that asserted the score dot product matches the CPU. It
  was never tested and it does not: the Java path uses a lane-striped `FloatVector` fold.

## [0.3.47] - 2026-09-25

### Fixed

- Preserve standard cancellation/interruption signals through SDK cause chains. Stop Spring AI
  and LangChain4j streaming fallback on these signals without penalizing model health; keep the
  original streaming error and restore interruption only for blocking callers.

- Reject non-finite/negative routing price and latency ceilings, and quality floors outside
  `[0, 1]`, at policy construction. `NaN` previously bypassed hard eligibility comparisons.
- Stop blocking fleet execution on cancellation or interruption without invoking fallback clients
  or recording a model-health failure. Preserve the interrupt flag and propagate interruption
  as `CancellationException` with the original cause.

### Added

- Per-request capability and data-boundary factories on the Spring AI and LangChain4j routed
  adapters. Factories receive the complete original request; requirements constrain both selection
  and fallback, including streams. Existing constructors retain their unconstrained defaults.


- `GgufHugePages` loads weights into an anonymous `MADV_HUGEPAGE` mapping instead of mapping the
  GGUF, so a 2.6 GiB model costs roughly 1,300 page-table entries rather than 650,000. **Off by
  default**, via `models.purejava.hugePages`, because that is what the measurement says. On a
  dedicated CCX33 it is worth +3.7% on a bare 18-token forward pass (0.4999 s to 0.4813 s, seven of
  eight alternating rounds won) and nothing at all on the workload the decision path actually runs:
  ten rounds of a fifteen-decision video-shape cohort give 1.3837 s against 1.3689 s per decision,
  +1.1% on five rounds won out of ten, with per-round deltas that alternate sign. `AnonHugePages`
  was sampled at 2,674,688 kB during those runs, so that is no effect rather than no data. Kept,
  off, because it is bit-exact and cheap to switch on for a workload that prefills narrowly instead
  of in wide batches. Mapping the region directly also removed a 2.7 GB zeroing memset that
  `Arena.allocate` was doing: `GgufParser.parse` went from 1.858 s to 0.653 s when the path is on.

### Changed

- Consume Vectors 0.1.23, including the Q6_K scalar bit-equality correction, and update Jackson
  dependencies to 2.21.7. The latter removes the three OSV advisory matches on the adapter's
  previously declared Databind 2.21.4 dependency.


- Q4_K output-row tiling was implemented, measured and **reverted**. It is worth +9.1% on an
  isolated matmul and +0.4% on a real forward pass, because the isolated arm re-times one ~13 MB
  tensor until it is L3 resident and a forward pass never sees a weight twice. Two earlier
  conclusions are retracted in the benchmark notes as a result: the Q4_K inner loop is **not** at its
  algorithmic ceiling -- the same kernel reaches 25.6 MAC/cycle/core with weights resident against
  17.1 cold, and neither arm uses more than a fifth of the host's measured 38.0 GB/s -- and the
  "43% of a forward pass is not matmul" figure was an artifact of timing resident tensors, where
  cold it is 21%.

## [0.3.46] - 2026-09-24

### Added

- An answer space can say what its labels mean. `AnswerSpace.criteria()` carries an optional
  per-label rubric, `Noul`, `Choice` and `Score` accept one, and `LetterLogitScorer.renderCriteria`
  renders it. The published score for a decision model had been measured on a prompt this API could
  not build: JevBench hands every system a per-label rubric out of `question.criteria`, and
  `AnswerSpace` had only `question()` and `labels()`. Measured over 120 JevBench items on a
  dedicated CCX33, same model and kernel, the only variable being whether a rubric reaches the
  prompt: accuracy 0.7500 and Intelligence 72.2 without, 0.9000 and **88.9** with. Ordinal questions
  went from 0.2500 to 0.9167, which is what `A: 3` means to a reader never told what 3 is. For scale,
  swapping the recurrence kernel across all 24 layers moves one item in that cohort.
- Where the rubric goes was measured rather than chosen, and it is also where the latency is. Rubric
  and letters both after the criterion scored 86.1, both before it 79.6, the rubric before with the
  letters after 88.9. The last is both the most accurate and the cheapest, because everything before
  the criterion is shared across every question about one piece of evidence: a 53-token rubric costs
  0.486 s per question against 0.490 s for no rubric at all. Placed after the criterion it would
  have cost 0.98 s per question.
- `GroupedDecisionBackend.groupedDecisionsMatchSingleDecisions()` and
  `groupedDecisionBreakEven()`, so a backend states whether grouping changes its answers and at what
  size it starts to pay, instead of the caller assuming both.

### Fixed

- Grouping could change an answer. A lone question reads its answer out of a batch of one row and a
  group reads its out of a batch of many, and the native kernel keeps a separate single-row path.
  Measured: three of four grouped answers differed from the same questions asked alone, by up to
  0.28 of a logit; the pure Java decoder is bit-identical. Grouping is now refused unless the
  backend states it preserves answers, however profitable it would be.
- A grouped decision re-read its whole evidence on every call and discarded the resumption point, so
  the next one-at-a-time decision paid for it again. Twenty questions went from 10.79 s to 8.22 s.
- The Gated DeltaNet kernel stays on the calling thread for one token of one sequence, which is right
  for decode; a group called it once per branch and ran every branch on one core. It now takes a
  whole group in one launch, at native kernel ABI 6. Twenty questions: 12.39 s to 10.79 s.
- A decision ended by prefilling all but its last token and then stepping that token alone, reading
  all 2.55 GiB of weights for one token at 67 ms. The answer is now read off the final position of
  one prefill, where it is one more row of an already compute-bound batch. 0.520 s to 0.485 s per
  question, with no answer changed: 0 of 120 winners moved.
- Artifacts written before the rubric existed stay readable. Version 3 appends a per-label rubric;
  version 2 files are read as having none rather than rejected.

### Changed

- Documentation that claimed a decision is bandwidth bound, and that answering N questions one at a
  time pays for the weights N times, said the opposite of what this hardware does. Batched prefill
  measures 18.5 ms per token, linear with no fixed cost, halving from one thread to two and again to
  four before saturating on four physical cores. It is compute bound and already at the ceiling, so
  N questions cost N questions' arithmetic however they are arranged. The claims are corrected in
  place and the measurements are in `benchmark-results/2026-09-24-decision-latency`.

## [0.3.45] - 2026-09-24

### Fixed

- A session was sized for the model's own maximum context. For Qwen3.5 that is 262,144 tokens,
  whose key and value cache alone is about 17 GB, so opening a decision runtime on a default heap
  died before the first answer. `models.purejava.maxContextLength` is now a plan-configuration
  setting like every other, so a ModelJar that knows its prompts are short can recommend a bound.
  A deployment setting still wins, and absent both the session is sized for the model's maximum.
  Measured with Harriet on an 8-vCPU EPYC-Milan box: `OutOfMemoryError` before, `open 2.79 s` on a
  7.7 GiB default heap after.

## [0.3.44] - 2026-09-23

### Added

- `ResumableInferenceBackend`: a backend can capture a position it has read and return to it, so
  repeated decisions over one piece of evidence stop paying for that evidence every time. Measured
  with Harriet on an 8-vCPU EPYC-Milan box over a 112-word contract, per-decision cost fell from
  3.74 s to 0.69 s. Resumed answers are bit-identical to a cold prefill of the same prompt.

  It is deliberately narrower than `SharedPrefixInferenceBackend`: a shared prefix hands out
  branches that live at the same time, a resumption is strictly sequential. Sequential reuse is the
  only shape a recurrent state can offer cheaply, because a Gated DeltaNet state is a running
  summary that can be copied back but not sliced.

### Fixed

- Qwen3.5 `reset()` and `rewind()` reallocated the whole session state on every call. They now
  clear the recurrent and convolution state in place; a key or value at or beyond the new position
  is never read.

## [0.3.43] - 2026-09-23

### Added

- `models-decisions`, a System One decision tier: state in, typed probabilistic decisions out, in
  one pass, with no decode loop anywhere. Three primitives, `Noul` for binary, `Choice` for 2 to
  255 unordered options and `Score` for 2 to 10 ordered levels, with the answer space declared
  before the decision is made so it is a contract rather than something parsed out of prose.
  Candidates are scored against a state read once, forking from a frozen KV prefix, so many
  questions against one document pay for that document once. `DecisionArtifact` is the release
  unit and carries the base weight digest: loading it against weights whose SHA-256 disagrees
  throws rather than scores.

- `backend-cuda`, Models-owned Rust kernels compiled to PTX and driven through Panama FFM: fused
  dequantise-and-multiply projections for Q4_K and Q6_K, and grouped-query attention for the decode
  step. The arithmetic compiles unchanged for the host and for `nvptx64`, so it is tested against a
  CPU reference on any machine with no GPU and no CUDA toolkit. `CudaRoutingCounters` records what
  executed on the device and why anything did not, so a run that accelerates nothing distinguishes
  an unsupported format from an ineligible shape from an absent device. **No kernel here has been
  shown to beat the CPU path**; measured warm on an A40 against 8-core CPU SIMD it is slower. This
  is the substrate the measurements run on, not a performance claim.

- `RuntimeReport`, which prints the kernel, device and settings a run actually used and can fail
  closed on a demanded kernel. An unreported fallback has invalidated measurements here before,
  once landing within 1% of the previous number while hiding a large difference.

### Changed

- The hidden-state prefill now reaches the batched path. `prefill` has consulted the batched
  execution plan since it was written and `prefillHiddenState` never did, so every hidden-state
  consumer ran the whole prompt one token at a time: embeddings through `GgufEmbeddingBackend`, and
  any consumer reading a state off a prompt. Measured on Granite 4.1 3B Q4_K_M on an 8-vCPU
  EPYC-Milan host, over 128 tokens, the hidden-state prefill went from 66,538 ms (1.9 tok/s) to
  4,283 ms (29.9 tok/s), a 15.5x improvement. It now runs slightly faster than the logits prefill
  at that length and reaches parity by 512 tokens, which is the expected shape: it skips the
  vocabulary projection on every row but the last.

  The `vectors.gguf.pollMillis` budget was the leading hypothesis and was ablated in both
  directions, 1.86 tok/s at zero against 1.87 at the default; it explains none of the gap. The
  cause is memory bandwidth, since a single-token forward streams the whole weight set to produce
  one row.

  Results are unchanged within the repository's existing SIMD reduction tolerance: the hidden
  state, the key/value cache contents, and the next generated token all match running the prompt
  one token at a time.

### Added
- `backend-tornado` K-quant projection kernels: Java TornadoVM kernels for GGUF Q4_K and Q6_K by Q8_K, alongside the existing Q4_0 by Q8_0 kernels. This is what a `Q4_K_M` catalog model is made of, so the accelerator previously admitted none of its projections. Admitted grouped shapes are `Q4_K/Q4_K`, `Q4_K/Q4_K/Q4_K` and `Q4_K/Q4_K/Q6_K` — the last being the query/key/value group `Q4_K_M` presents, since llama.cpp promotes the value projection to Q6_K. Q4_0 (Q8_0 activations) and the K-quants (Q8_K activations) are never combined in one grouped dispatch, and K-quant projections additionally require a column count that is a multiple of 256. Q5_K, Q2_K, Q3_K and every unquantized format still fall back to the Vector API per projection.
- `TornadoBackendRuntime.routedProjectionsByFormat()` and `projectionPlanCount()`: how many projections actually reached the device, per GGUF weight format, counted once per matrix in a grouped dispatch. A mixed-format model can otherwise look accelerated while one of its formats silently falls back.
- `models-bench accelerator-profile --model <model.gguf>`: runs prefill and greedy decode on the Tornado backend and writes device, fallback reason, eager-readiness time, prefill/decode tokens per second and the per-format routing counts as JSON. `--require true` makes an unavailable accelerator a hard failure instead of a silent Vector API fallback.
- Numeric contract for the K-quant kernels, asserted off-device against the production vectors-core CPU kernels: bit-for-bit equality on super-blocks whose scales are powers of two and whose reductions stay exactly representable, and agreement to within two float roundings per super-block otherwise (the CPU kernels fuse the per-super-block scale application with `Math.fma`; the device kernels use a plain multiply and add). Worst measured differences on pseudo-random super-blocks, 2026-09-18, Apple M-series, 256-bit species: Q4_K 7.6e-6 / 6.1e-5 / 1.2e-4 and Q6_K 0.0 / 9.2e-5 / 1.8e-4 at 256 / 1024 / 4096 columns, against reference magnitudes of 3.3e2 / 5.3e2 / 1.4e3. Device-side parity is a hardware gate and has not been run.
- Device causal grouped-query attention for the single-token decode step in `backend-tornado`, opt-in through `TornadoBackendOptions.withAcceleratedAttention()` and off by default. `TornadoAttentionKernel` is Java source TornadoVM compiles; it uses only `@Parallel` loops and no `KernelContext`, so the same methods execute as ordinary Java in CI. `LlamaForwardPass` now offers the decode step to the attention SPI, which it previously never did — the SPI was consulted on the batched prefill path only.
- Prefill chunks are refused, with the reason recorded. A layer holds one KV mirror, and a prefill chunk is a different execution shape, so it would compile a second plan owning a second, empty mirror for that layer; sizing the one plan to the prefill batch instead makes every decode dispatch a mostly-idle grid. Sharing one mirror across two task graphs is the way to have both and needs a device to verify.
- Device-resident KV mirror with a host-owned cache. The host `KvCache` stays the source of truth: forking, `freezePrefix`, `rewind`, `discardFrom` and `clear` are unchanged, and `sharesPrefixStorage` still reports physical host sharing between branches. The device holds a position-major mirror per layer, brought up to date from the cache's existing zero-copy spans through the new `BatchedCausalAttentionKernel.mirroredPosition`/`mirrorSpan` methods, so a decode that follows a Java-path prefill does not recompute the history. A mirror belongs to one sequence: two branches of one frozen prefix each rebuild their own, counted rather than hidden.
- `BatchedCausalAttentionKernel.AttentionScope` and `selectScope`: a retained kernel is told which sequence a call belongs to (`KvCache.sequenceId()`), how much of it is shared prefix, the attention scale the Java path would apply, and whether that path uses the fused grouped arithmetic. The scale is now carried rather than assumed to be `1/sqrt(keyLength)`, which it is not on Granite.
- `TornadoAttentionRouting`, reachable from `TornadoBackendRuntime.attentionRouting()` and logged at open: device layer-steps split by decode and prefill, refusals with the gate that closed, mirrored positions, and mirror rebuilds. A kernel that never accepted a step reports itself instead of resembling one that ran and did not help.
- `TornadoAttentionShape`: the device-byte and per-step transfer arithmetic behind the KV-placement decision, checked in `TornadoAttentionShapeTest` rather than asserted in prose. `AcceleratorEligibility.select` takes an optional retained-sequence reservation; passing zero reproduces the projection-only budget exactly.
- `TornadoAttentionDecodeExperiment` in `models-accelerator-bench`: the device gate for this path. Off-device CI proves the arithmetic and routing; PTX generation, mirror residency across executions, and device latency need a GPU host.

### Fixed
- `backend-tornado` no longer admits a weight tensor too large for TornadoVM's 32-bit array index (about 2 GiB on the device). Such a projection now stays on the Vector API instead of reaching `ByteArray.fromSegment` with an unrepresentable size.

### Notes
- Device attention is not in the qualified scope and has had no real-hardware gate. Off-device it is bit-exact against a scalar reference and `2.97e-7` relative L2 against the production Java attention path (contract `2e-5`). The standing real-model measurement for device attention on this codebase is negative — A16, 2026-08-29: a tiled prefill kernel at relative L2 `3.47e-7` and an isolated `1.31x` regressed warm prefill from `4.781 s` to `10.266 s` — and it covered prefill, not the single-token decode step this path targets.

## [0.3.42] - 2026-09-17

### Added
- GGUF chat-template end-of-turn resolution: `GgufTokenizer` reads `tokenizer.chat_template` and adds the token that closes an assistant turn to the end-of-generation set. That token is the last CONTROL token or `eos_token` reference before the generation prompt, and it must also close message content somewhere in the template. Otherwise the result is reported unresolved and the set is unchanged. Of the 55 GGUF vocabularies on the development host, 26 templates resolve, 5 are unresolved (no generation prompt), and no stop set changes, because every resolved marker was already present.
- End-of-generation provenance: `GgufTokenizer.endOfGenerationSources()` lists the rules behind each terminator (metadata key, `vocabulary-text`, `chat-template-end-of-turn`, ...). `chatTemplateEndOfTurnResolution()` returns `resolved:<id>`, `unresolved:<reason>` or `absent`. `PureJavaBackend` and `RustFfmBackend` diagnostics carry `end-of-generation-token-ids`, `end-of-generation.<id>` and `end-of-generation.chat-template`.
- `ChatTemplate.endOfTurnMarker()`, `endOfTurnTokenId(Tokenizer)` and `endOfTurnStopsGeneration(Tokenizer)`: each native template's assistant terminator, pinned by test to what the renderer writes. The one exception is GPT-OSS, where the marker is `<|return|>`. A fixture test checks that eight pinned GGUF and template pairings stop on it. Runtime injection of a native template's marker into the stop set is not wired.
- Performance cliffs `gguf-parallel-disabled` (`-Dvectors.gguf.parallel=false` with more than one processor) and `native-grouped-attention-span-limit` (Granite native grouped attention over a KV cache view with more than two spans falls back to Java).

### Fixed
- Stop sets no longer include ordinary `</s>` vocabulary entries. A vocabulary-text match now requires a token that the GGUF does not type NORMAL, and `</s>` must be typed CONTROL. Metadata-declared ids are unaffected. On `tokenizer.json` vocabularies with added tokens, only added tokens can match by text. This removes id 128247 from every Qwen2-vocabulary GGUF (NORMAL, reachable only through the BPE merge `</s` + `>`) and id 212 from Gemma 3 (the HTML strikethrough close tag). Neither is in the upstream `generation_config.json`. Both ids are now decoded as text. Pinned integration oracles are unchanged.
- `GgufParser` rejects tensor data offsets that are not a multiple of `general.alignment` (default 32) with `MalformedGgufException`. Vocabulary-only GGUFs with no tensors, such as llama.cpp's `models/ggml-vocab-*.gguf`, now parse even when their metadata ends off the alignment boundary. Previously all 38 such files on the development host were rejected. All 74 GGUF files there now parse. Evidence: `benchmark-results/2026-09-16-wave2-followups/README.md`.
- LangChain4j: cancelling through `StreamingHandle` now stops the activated-tool streaming branch. No fragment, completion or error follows a cancel, and the shared turn is still closed. LangChain4j 1.0.0 streams as before.
- Flaky `LlamaForwardPassTest` injected-attention test: vectors-core's F32 matrix-vector kernel sums with `reduceLanes(ADD)`, whose float order changes once C2 compiles it, so identical prefills could differ in the last bit within one JVM. `TensorOps.ggufMatmul` now scores F32 rows with the order-defined `VectorUtil.dotProduct`. The speed of that path was not measured, and among local models only the MS MARCO rerankers use it. Analysis: `benchmark-results/2026-09-16-flaky-injected-attention/README.md`.

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
