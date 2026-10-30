# Rock Memory Anchors: the mechanism, and where a port will go wrong

Derived by reading the reference implementation, `modeling_edgerock.py` in
`rockmatrix-ai/EdgeRock` (1093 lines), class `IndependentGQAMemory` at `:122`. The paper
(arXiv:2610.10118, "EdgeRock: Strong Reasoning at Constant Memory") names the mechanism **Rock
Memory Anchors** and states the intent -- "learns what is worth remembering and stores it as KV
records in a fixed-size bank" -- but not the arithmetic. The arithmetic below is from the code.

Described, not copied. That file ships under the EdgeRock Model License rather than Apache, so
this is a specification in our own words; no lines are transcribed into this repository.

Cross-check that the reading is right: every shape below matches the tensors in the GGUF.
`admission_up` is `[256, 4]` = (rank, groups); `read_gate_weight` is `[256, 16]` = (dim, heads);
`read_scale` is `[16]` = heads; `write_*_proj` sit at `writer_intermediate = 2560`.

The backbone question is settled in the same file: `Qwen3_5SegmentAwareGatedDeltaNet` at `:355`
**subclasses `Qwen3_5GatedDeltaNet`**. EdgeRock's recurrence is Qwen3.5's, segment-aware.

## Per-layer constants

`capacity = 1024`, `max_commit_chunk = 256`, `admission_rank = 256`,
`writer_intermediate = 2560`, `groups = 4` (KV heads), `heads = 16`, `dim = 256`.
Eight layers carry this, on the full-attention layers.

## Write: build a KV record

1. **Residual SwiGLU writer.** `z = x + write_down(silu(write_gate(x)) * write_up(x))`.
   Note the residual -- the writer refines the hidden state rather than replacing it, and
   `write_down` is zero-initialised, so an untrained writer is the identity.
2. `key = k_norm(k_proj(z))`, `value = v_proj(z)`, both reshaped to `(groups, dim)`.
3. **RoPE on the memory key**, partial: rotation applies to the leading `rope.dimension_count`
   lanes and the remaining lanes pass through unchanged.

## Score and query

1. `query = q_norm(q_proj(u))` reshaped to `(heads, dim)`, then the same partial RoPE.
2. **Read gate, per head:** `tanh(read_scale) * 2 * sigmoid(dot(query, read_gate_weight)/sqrt(dim)
   + read_gate_bias)`. Range is `(-2, 2)`; it can suppress or invert a head's read.
3. **Admission score, per group:** `admission_up(silu(admission_down(u)))`, with the input
   detached. `admission_up` is zero-initialised, so admission starts neutral.

## Admission: a stable top-k monoid

`select_topk(scores, positions, references, capacity)` is two sorts, in this order:

1. stable sort by **position**, with invalid references forced to the maximum so they land last;
2. stable sort by **score, descending**, truncated to `capacity`.

Short results are padded with `(-inf, -1, -1)`. `merge_topk` is concatenate-then-select, which
makes admission associative, so prefill uses a balanced reduction tree (`exclusive_prefix_topk`,
`:65`) and decode uses a single merge.

## Decode, per token

**Read first, from committed state only** -- the token being written does not see itself:

1. `logits = q @ keys^T / sqrt(dim)`, with `-inf` at slots whose position is negative.
2. **A zero sink is appended before the softmax and dropped after.** The weights therefore need
   not sum to one: the layer can attend to nothing.
3. `read = weights @ values`, then multiplied by the token's validity.

**Write second, staged:**

4. key, value and score land in a pending buffer at `slot = count`.
5. `count` advances; when it reaches the chunk size, **commit**: `merge_topk` over
   (committed bank) ++ (pending), keeping `capacity`, then gather keys and values by the chosen
   references and reset `count`.
6. `total` advances every valid token; pending positions are derived as `total - count + index`.

So the bank is immutable between commits and changes wholesale every 256 tokens.

## The five ways a port goes wrong silently

Each of these loads, runs, and produces plausible output.

1. **Read is float32.** The reference casts query, keys and values to float before the matmul,
   whatever the model dtype. Reading in bf16 shifts the weights.
2. **Admission scores are float32 with autocast explicitly disabled.** Compute them in the
   model dtype and the top-k selects a *different set of records* -- not a small numeric drift
   but different memory contents, diverging further every commit.
3. **The zero softmax sink is not optional.** Drop it and every read weight changes, because
   the normaliser loses its `exp(0)` term.
4. **Both sorts are stable, and position-ordering comes first.** An unstable sort, or scoring
   before ordering by position, selects a different set among equal scores. Our own
   determinism work on 2026-10-29 is the precedent: identical inputs, different results,
   discovered only by byte-comparing output.
5. **Commit is all-or-nothing at the chunk boundary.** Committing per token would keep a
   strictly better bank and still be wrong -- it would not be this model.

## What to build, in order

1. `EdgeRockMemory` holding the bank as `(keys, values, scores, positions, count, total)` plus
   the pending buffer, sized from `edgerock.memory.*`.
2. The write and score paths. `causalDepthwiseConv1dSilu` and `sigmoidAndScaledSoftplus` in
   vectors already cover the GDN side; the writer is a plain SwiGLU, already available.
3. A stable top-k over `(score, position, reference)` triples. This is the piece with no
   existing equivalent in vectors and the one worth a dedicated test, including ties.
4. Sliding-window attention at `edgerock.attention.window = 1024` on the 8 attention layers
   (`reference_swa` at `:412` is the reference).
5. Parity against the shipped reference runtime on a fixed prompt, **before** any quality claim:
   logits first, then the selected reference set after a commit, which is where divergence will
   show first.

## Not yet established

- Whether mRoPE `dimension_sections [11, 11, 10, 0]` can be ignored for text-only input.
- How `IndependentMemoryState` interacts with our prompt cache. A longest-common-prefix hit
  must restore the bank *and* `count`/`total`, or the first commit after a cache hit lands on a
  bank the prefix never built.
- Nothing here is measured by us. The 82.9% AIME and 99.1% KV-reduction figures are read in an
  abstract and are not ours to repeat.
