# Qualification payload provenance

Each payload name is immutable once launched: a deployed `0.3.47` tarball must keep meaning exactly
one build, or a shard's results stop being attributable. Never overwrite one — add the next number.

| payload                               | shards | what it contains beyond the previous |
|---------------------------------------|--------|--------------------------------------|
| models-rag-bench-0.3.47.tar           | 0-99   | original epoch-1 run (17/24 qualified) |
| models-rag-bench-0.3.47-rerun.tar     | 20-23  | qwen3moe routed FFN, mistral3 + attention temperature scaling, tekken/gpt-4o/deepseek pre-tokenizers, partial rotary, MXFP4 matmul |
| models-rag-bench-0.3.47-rerun2.tar    | 24     | gemma4 E-series (per-layer embeddings, shared KV) |
| models-rag-bench-0.3.47-rerun3.tar    | 25     | LFM2 decoder (short convolution) |
| models-rag-bench-0.3.47-rerun4.tar    | 26     | gemma4 BF16 matrix type (`per_layer_model_proj` is BF16) |
| models-rag-bench-0.3.47-rerun5.tar    | 27     | the four defects the shard-20..26 results exposed (see below), plus qwen35moe |
| models-rag-bench-0.3.47-rerun6.tar    | 28     | the six header-derived fixes below, plus qwen3next |
| models-rag-bench-0.3.47-rerun7.tar    | —      | **superseded, never launched.** gemma3n + deepseek2; folded into rerun8 before shard-29 launched, so one box runs all three |
| models-rag-bench-0.3.47-rerun8.tar    | —      | **superseded, never launched.** gemma3n, deepseek2 and the gpt-oss GGUF loader; shard-30 was re-armed on rerun10 before it fired. sha `c7455c159944` |
| models-rag-bench-0.3.47-rerun9.tar    | —      | **superseded, never launched.** Same contents as rerun10; its recorded commit was rewritten before launch, so the name was retired rather than repointed |
| models-rag-bench-0.3.47-rerun10.tar   | 30     | **supersedes rerun8 before launch.** gemma3n, deepseek2 and the gpt-oss GGUF loader, plus the Gemma 4 shared-KV weights-loader fix, so E4B re-runs on the same box. sha `004e2fd48651` |

Payload sha256[0:12] = `cf5f64c49bfe`, reported by the worker as
`--backend-version models@0.3.47+rerun5-cf5f64c49bfe`, so every result JSON names an artifact that
can be re-hashed. Launched on i-047ddd2337daa04ba (m6a.4xlarge, 320 GB, 18000s deadline).

## rerun5 contents (2026-09-29)

Four failures from shards 22/25/26, each in code whose own tests were green, each found only by a
real model:

1. **qwen3moe** (3 models, `rc1`): `ModelTopology.threadShareable` NPE on a routed layer's null
   dense feed-forward segments. The tensor *types* were routing-aware; the segment probe one method
   below was not.
2. **gemma4 E2B** (`rc1`): `gemma4.feed_forward_length` is a per-layer array on a MatFormer model and
   was read as `.get(0)`. Refused at `blk.15.ffn_gate.weight shape must be [1536, 6144], found
   [1536, 12288]`.
3. **gemma4 E4B** (`rc1`): `gemma4Topology` read `blk.24.attn_k.weight`, which a shared-KV layer does
   not carry. The forward pass knew about shared KV; the planner in front of it did not.
4. **lfm2** (`rc1`): the decoder adapter refused `rewind`, and the RAG harness rewinds to reuse a
   shared prompt prefix before the first question. Now rewinds by replaying retained tokens, as
   qwen35 does for its Gated DeltaNet recurrence.

Plus **qwen35moe**: routed + shared-expert feed-forward on the Qwen3.5 hybrid graph, confined to the
single-token path (the graph clamps its own prefill batch size to 1 when routed), and added to
`SUPPORTED_ARCH`.

## rerun6 contents

Payload sha256[0:12] = `525915c3c0de`, reported as
`--backend-version models@0.3.47+rerun6-525915c3c0de`. (A tar embeds mtimes, so repacking the
same sources changes the hash -- always re-render the user-data from the template after packing,
which is why the template now carries a `__PROVENANCE__` placeholder rather than a baked string.)

Found by reading the **published GGUF headers** of Qwen3.5-2B, Qwen3.5-35B-A3B and
Qwen3-Coder-Next over HTTP range requests, before spending anything on them. Five of the six would
have made the qwen35moe support useless or silently wrong:

1. **`output.weight` is untied on qwen35moe and qwen3next** but absent (tied) on dense Qwen3.5. The
   loader tied unconditionally. The token embedding has exactly the head's shape, so this is not a
   load failure -- it computes every logit from the wrong matrix, i.e. fluent, confident, wrong text.
2. **`ffn_gate_inp_shexp.weight` is 1-D** `[n_embd]`: GGUF drops a trailing dimension of 1. The
   loader required `[n_embd, 1]` and would have refused all four published qwen35moe models. The
   synthetic fixtures could not see it, because a builder handed `{columns, rows}` writes both dims.
3. **MXFP4 is a matrix type on qwen3next** (`ffn_gate_shexp`, `ffn_up_shexp`), while every other
   matrix in that file is a K-quant.
4. **qwen3next fuses beta and alpha into `ssm_ba`**, packed *interleaved by key-head group* -- for 32
   value heads over 16 key heads the 64 rows read `[b,b,a,a]` sixteen times. Reading the first 32 as
   beta pairs every value head with the wrong parameter and still produces plausible output. Taken
   from `llama.cpp/src/models/qwen3next.cpp`, not guessed.
5. **`Qwen35Config` discarded the architecture name**, so `topology()` inferred it from the
   feed-forward shape and would have reported every qwen3next as a qwen35moe.
6. **Gemma 4 E4B carries `per_layer_model_proj` as F16** where E2B carries BF16 -- and `ggufMatmul`
   implemented F16 only in its batched form, while the per-layer projection runs a token at a time.
   (Found from the shard-27 log, i.e. a real model again.)

## shard-28 outcome so far (rerun6) — 3 of 6, all qualified

All at `summary.correctAnswerRate = 1.000`, 27/27 attempts, zero failures, provenance
`models@0.3.47+rerun6-525915c3c0de`.

| model | arch | artifact | verdict | what it confirms |
|---|---|---|---|---|
| deepreinforce_ai_ornith_1_0_35b_gguf_q4_k_m | qwen35moe | 21.2 GB | car 1.000 | the five header-derived qwen35moe fixes |
| bartowski_kwaipilot_kat_coder_v2_5_dev_gguf_q4_k_m | qwen35moe | 21.4 GB | car 1.000 | same |
| unsloth_qwen3_coder_next_gguf_q4_k_m | qwen3next | **48.5 GB** | car 1.000 | the fused `ssm_ba`, interleaved by key-head group |

Still to run: gemma_4_e4b_it_q4_0 (the F16 fix), unsloth_qwen3_5_35b_a3b, unsloth_qwen3_6_35b_a3b.

### A measured performance data point worth keeping

Qwen3-Coder-Next is an **80B** and its Q4_K_M is 48.5 GB -- the shard manifest's `gb: 1.00` is repo
metadata and is not the artifact size. On an m6a.4xlarge (16 vCPU, 64 GB) the weights therefore do not
fit in page cache alongside the heap, and it still answered **every case correctly**, at 38.6 s to first
token and 241 ms/token against roughly 4 s and 80 ms for the 21 GB models on the same box.

So an oversized artifact **degrades rather than fails**. That is why `MAX_ARTIFACT_BYTES` should be set
from the shard's deadline and model count, not from the instance's RAM -- and why the cap being dead code
(it read a `bytes` field the manifests never carried) cost throughput rather than correctness. The worker
now HEADs the pinned URL instead.

## Models to re-qualify on rerun6

- unsloth_qwen3_5_35b_a3b_gguf_q4_k_m          (qwen35moe)
- bartowski_kwaipilot_kat_coder_v2_5_dev_gguf_q4_k_m (qwen35moe)
- unsloth_qwen3_6_35b_a3b_gguf_ud_q4_k_m       (qwen35moe)
- deepreinforce_ai_ornith_1_0_35b_gguf_q4_k_m  (qwen35moe)
- unsloth_qwen3_coder_next_gguf_q4_k_m         (qwen3next)
- gemma_4_e4b_it_q4_0                          (gemma4 E-series, F16 projection)

## shard-27 outcome (rerun5) — 5 of 6 qualified

Measured, not projected. Every one at `summary.correctAnswerRate = 1.000`, 27/27 attempts, zero
failures, provenance `models@0.3.47+rerun5-cf5f64c49bfe`.

| model | verdict | what it confirms |
|---|---|---|
| liquidai_lfm2_5_1_2b_instruct_gguf_q4_k_m | car 1.000 | LFM2 rewind-by-replay |
| lmstudio_community_gemma_4_e2b_it_gguf_q4_k_m | car 1.000 | per-layer feed-forward width (MatFormer) |
| unsloth_qwen3_30b_a3b_gguf_q4_k_m | car 1.000 | routed-layer segment probe in the planner |
| unsloth_qwen3_30b_a3b_instruct_2507_gguf_q4_k_m | car 1.000 | same |
| unsloth_qwen3_coder_30b_a3b_instruct_gguf_q4_k_m | car 1.000 | same |
| gemma_4_e4b_it_q4_0 | rc1 | `per_layer_model_proj` is **F16** here where E2B is BF16, and `ggufMatmul` had F16 only in its batched form. Fixed in rerun6 |

## Models to re-qualify on rerun5

- unsloth_qwen3_30b_a3b_gguf_q4_k_m
- unsloth_qwen3_30b_a3b_instruct_2507_gguf_q4_k_m
- unsloth_qwen3_coder_30b_a3b_instruct_gguf_q4_k_m
- lmstudio_community_gemma_4_e2b_it_gguf_q4_k_m
- gemma_4_e4b_it_q4_0
- liquidai_lfm2_5_1_2b_instruct_gguf_q4_k_m

## rerun10 contents

Payload sha256[0:12] = `004e2fd48651`, reported as
`--backend-version models@0.3.47+rerun10-004e2fd48651`.

It was built and uploaded once as rerun9 against commit `c19c4b41bf4a`. That commit was then rewritten
-- a formatter fix belonged in it rather than in the commit after it -- which changed its sha, so the
payload named a commit that no longer existed. Nothing had launched against rerun9 and no result
referenced it, so the name was retired and the tar rebuilt as rerun10 rather than the deployed name
being quietly repointed at a different build. **The lesson is the cheaper one: do not rewrite a commit
whose sha has already been baked into a deployed payload.**

rerun8 was armed for shard-30 and had not launched, so it was replaced rather than followed. It adds
one fix to the same three architectures, and adds a fourth job to the shard:

**gemma4 E4B, found on shard-28.** The weights loader demanded `attn_k`, `attn_k_norm` and `attn_v`
on every layer:

    Tensor not found: blk.24.attn_v.weight for sliding attention

E4B declares `shared_kv_layers=18` over 42 layers, so layers 24-41 attend to an earlier layer's cache
and the published file carries none of those three for them -- confirmed by reading its tensor list,
where all three are present on every layer up to 23 and absent from 24 on. The reference marks exactly
those three not-required for exactly these layers (`models/gemma4.cpp`, `kv_flags`).

This is the **second** defect of this shape in the same model. rerun5 fixed `gemma4Topology` reading
`blk.24.attn_k.weight`; the planner learned about shared KV and the weights loader did not. The forward
pass had been right all along -- it already skipped both projections for a layer that does not own its
cache, so nothing downstream of the loader changed.

Shard-30 therefore runs four jobs: deepseek2, gemma3n, gpt-oss GGUF, and gemma4 E4B.

