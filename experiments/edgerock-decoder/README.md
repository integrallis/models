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

## Order of work

1. **Prove the config parses** the real header with `edgerock` admitted, and prove the loader's
   tensor expectations against the real tensor list. Answers the `ssm_ba` question.
2. **Refuse rather than mis-serve.** Until the memory module exists, `edgerock` must fail loudly.
   Routing it to the qwen35 forward pass would ignore 8 layers of memory contribution and emit
   plausible, wrong logits -- the worst available outcome.
3. **Implement the memory module** against the paper, with a parity test versus the reference
   runtime on a fixed prompt before any quality claim.
4. **Sliding-window attention**, then mRoPE if the multimodal path is wanted.
5. **Qualify** under the generalised publisher-reference policy.

## What this document does not claim

No number here is measured by us. The paper's figures -- and the 10-31x and 99.1% KV-reduction
claims seen in its abstract -- are **read, not reproduced**, and none may be repeated as ours. The
decision to build is justified by the architecture and the licence, not by a self-reported score.
