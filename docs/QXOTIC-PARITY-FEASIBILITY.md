# Feasibility: the 12 architectures qxotic loads and we do not

2026-10-22. This is Stage 0 of `docs/QXOTIC-PARITY.md`, **executed rather than planned**. Every
architectural fact below was read out of a real published GGUF header by range-fetch today, and
every claim about our own side carries a `path:line`. Raw artifacts:
`benchmark-results/2026-10-22-qxotic-header-intake/`.

The artifacts were not chosen by us. They are the exact repository/file pairs in qxotic's own test
manifest (`jinfer/scripts/models.txt` at `6f08d93ec`), so no arm here was picked to make a port look
easy. The probe reuses `projects/model-jars-central-gate/tools/gguf-inspector.mjs` — the repo's
existing inspector — rather than a new parser.

**This evidence contradicts three things in the plan I wrote earlier today. Those corrections are in
§5 and the plan must be amended, not read alongside this.**

**Amended 2026-10-31 (#288).** `bailingmoe3` had no artifact when this was written and was recorded
as unassessable. One has since been located and its header read the same way as the other ten
(`headers.json` entry 11), so it moves out of §1 and into a tier. That amendment adds a fourth
correction, §5.4, and changes the count in §1, §2, §6 and §7. Nothing else in this document is
restated from memory — the new rows carry the same `headers.json` and `path:line` provenance as the
originals.

## 1. Two of the twelve are not feasible because there is nothing to load

Not "hard". **No artifact declares the string**, which an architecture-string comparison alone could
never have shown:

| string | what the header actually says | consequence |
| --- | --- | --- |
| `minicpm` | `openbmb/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q8_0.gguf` reports **`general.architecture = llama`**, 219 tensors, a plain Llama tensor set (`attn_{q,k,v,output,norm}`, `ffn_{gate,up,down,norm}`) | **Already supported.** We load it today through the Llama graph, and `minicpm5_1b_q4_k_m` is in our catalog. Their `minicpm` branch is a chat-template selector (`LlamaProvider.java:44`), not a graph. Nothing to port. |
| `nemotron_h` | the only Nemotron artifact in their manifest reports **`nemotron_h_moe`** | the dense variant has no fixture even on their side; it comes free with the MoE port or not at all |

`bailingmoe3` was a third row here on 2026-10-22, for the same reason: absent from their manifest.
It no longer is. `inclusionAI/Ling-3.0-tiny-GGUF` at revision `01b850e29ed5` — first-party, MIT,
**4.82 GB** — reports `general.architecture = bailingmoe3`, and its header is `headers.json` entry
11. It is assessed in §2, §5.4 and §6 like any other artifact. The absence was an absence in
qxotic's manifest, not in the world, and the two rows above are absences of the second kind.

So **the real target is 10 strings, not 12**, and one of those ten (`minicpm`) resolves to work
already done. That is the single most useful result of this intake and it cost no downloads.

## 2. What the ten remaining artifacts actually are

Read from `headers.json` and `arch-keys-and-dtypes.txt`:

| arch | artifact | size | tensors | dtypes present | shape of the thing |
| --- | --- | --- | --- | --- | --- |
| `maple` | `deepgrove/maple-preview-GGUF/maple-preview-TQ1_0-head-Q4_K.gguf` | **4.98 GB** | 291 | TQ1_0, Q4_K, F32, F16 | 24 blocks, 2048 dim, **256 experts / 8 used**, sliding window 512 |
| `mellum` | `JetBrains/Mellum2-...-Thinking-GGUF-Q4_K_M` | **8.07 GB** | 339 | Q8_0/Q4_K, F32 | 28 blocks, 2304 dim, 64 experts / 8 used, **YaRN** ×16, SWA 1024 |
| `lfm2moe` | `LiquidAI/LFM2.5-8B-A1B-GGUF/...-Q4_0.gguf` | **4.84 GB** | 256 | Q4_0/Q8_0, F32 | 24 blocks, short-conv + 32 experts / 4 used, 2 leading dense blocks |
| `laguna` | `poolside/Laguna-XS-2.1-GGUF/...-Q4_K_M.gguf` | **20.27 GB** | 678 | Q4_K, Q6_K, F32 | 40 blocks, 256 experts / 8 used, shared experts, **two rope dims** (64 global / 128 SWA) |
| `nemotron_h_moe` | `bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-GGUF` | **18.09 GB** smallest | 401 | Q8_0, F32 | 52 blocks, **Mamba-2** (`ssm.state_size=128`, `conv_kernel=4`, `group_count=8`, `inner_size=4096`, `time_step_rank=64`) + 128 experts / 6 used + 1 shared |
| `bailingmoe3` | `inclusionAI/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-Q4_K_M.gguf` | **4.82 GB** | 526 | Q4_K, Q5_0, Q6_K, F32 | 24 blocks, 1536 dim, **hybrid on a 4-layer interval**: 6 layers MLA with `q_lora_rank` 256 / `kv_lora_rank` 512, 18 layers **KDA** (`kda.head_dim` 128, `kda.safe_gate` 1, `kda.gate_lower_bound` -5.0); 128 experts / 8 used in **8 groups / 4 used**, 1 shared, 1 leading dense block, `expert_gating_func` 2 with `exp_probs_b` |
| `parakeet` | `mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf` | **0.94 GB** | 701 | Q8_0, F32 | `arch=hybrid_tdt_ctc`; FastConformer encoder (24 layers, d=1024, conv kernel 9, batch-norm, ×8 subsampling), mel front end (n_fft 512 / hop 160 / 128 mels), **2-layer LSTM** predictor, TDT joint |
| `kokoro` | `simonfxr/kokoro.cpp-GGUF/kokoro-82m-f16.gguf` | **0.16 GB** | 459 | F16, F32 | PL-BERT (12 layers) + text encoder + prosody predictor + ISTFT decoder; **178-token phoneme vocabulary** |
| `kokoro-voice` | `simonfxr/kokoro.cpp-GGUF/voices/kokoro-voice-af_heart.gguf` | **<0.01 GB** | **1** | — | a one-tensor companion artifact; 25+ voices in the repo |
| `inflect-v2` | `remixerdec/Inflect-Nano-v2-GGUF/inflect_nano_v2_q8_0.gguf` | **<0.01 GB** | 302 | Q8_0, F16 | VITS-shaped (`enc_p`, `dp`, `flow`, `dec`) and **no `inflect-v2.*` metadata keys at all** |

### The one structural split that matters

**Seven of the ten are transformer decoders with `blk.*` tensors; three are not.** `parakeet`,
`kokoro` and `inflect-v2` have **zero** `blk.` tensors — their roots are
`preprocessor/encoder/decoder/joint`, `bert/text_enc/pred/dec` and `enc_p/dp/flow/dec`. None of
`DecoderArchitecture`, `LlamaConfig` or the execution planner applies to them at all. They are not
harder ports of the same thing; they are different model classes.

## 3. The prerequisites, now measured instead of inferred

| prerequisite | needed by | do we have it | evidence |
| --- | --- | --- | --- |
| **TQ1_0 ternary blocks** | `maple` | **yes** | `GgufTensorType.java:44` maps GGML id 34; `quant/TernaryDequantizer.java` implements it. The artifact's dtype histogram is 168×TQ1_0 + 1×Q4_K head, exactly what the filename claims. |
| YaRN rope scaling | `mellum`, `laguna` | **yes** | `ops/RotaryTable.java`, `gptoss/GptOssConfig.java`, `deepseek2/Deepseek2ForwardPass.java` |
| two rope dims (global + SWA) | `laguna` | **yes** | `rope.dimension_count_swa` and `freq_base_swa` in `gemma4/Gemma4Config.java` |
| `expert_weights_scale`, `expert_weights_norm`, `expert_shared_count`, `leading_dense_block_count` | `laguna`, `lfm2moe`, `nemotron_h_moe` | **yes** | `deepseek2/Deepseek2Config.java`, `ops/ExpertRouting.java`, `qwen35/Qwen35ForwardPass.java` |
| short-conv blocks | `lfm2moe` | **yes** | `lfm2/Lfm2ShortConv.java` (artifact confirms `shortconv.{conv,in_proj,out_proj}`, `l_cache=3`) |
| **`expert_gating_func`** | **all four MoE ports** | **NO — read nowhere** | zero occurrences across `backend-java`. The artifacts split on it: `maple`=1 (softmax), `laguna`/`lfm2moe`=2 (sigmoid). |
| **sigmoid gating + `exp_probs_b` bias-corrected top-k** | `laguna`, `lfm2moe`, `nemotron_h_moe` | **NO** | `ops/ExpertRouting.java` is 124 lines of **softmax only**, in two renormalisation variants (`:53`, `:101`). No sigmoid path, no routing bias. |
| **Mamba-2 selective scan** | `nemotron_h_moe` | **NO** | ours is GatedDeltaNet (`ssm_a/alpha/ba/beta/conv`), the artifact wants `ssm_in`, `ssm_conv1d`, `ssm_d`. We already *parse* `ssm.state_size`, `ssm.group_count`, `ssm.time_step_rank` (`qwen35/Qwen35Config.java`), so the config surface exists and the recurrence does not. |
| I-quants (IQ2/IQ3/IQ4) | nothing, but see §4 | **NO** | `GgufTensorType.java:23-62` stops at Q8_K/BF16/MXFP4/TQ/PTQ. 9 of Nemotron's 14 small quants are I-quants and unreadable to us. |
| transcription model class | `parakeet` | **NO** | zero matches for a transcription/speech-recognition interface in any `*/src/main`, where TTS has `models-api/.../TextToSpeechModel.java:21` |
| conformer, LSTM | `parakeet` | **NO** | neither appears in any source of ours |
| grapheme-to-phoneme | `kokoro`, `inflect-v2` | **NO**, and it is **external on their side too** | `references/qxotic/NOTICE:6-9` names an external program for `inflect2` and an external phonemizer for `kokoro` |

**So there is exactly one shared software prerequisite worth building as its own unit:**
sigmoid expert gating with `exp_probs_b` bias correction, driven by a newly-read
`expert_gating_func`. It unblocks three of the six decoder ports.

## 4. Size is the hard constraint, and it is not uniform

The ModelJars founding principle is small-and-medium-first. Measured against it:

| arch | smallest artifact that exists | readable by us | verdict against the thesis |
| --- | --- | --- | --- |
| `inflect-v2` | 0.005 GB | yes | **comfortably in scope** |
| `kokoro` | 0.16 GB (+ <0.01 GB per voice) | yes | **comfortably in scope** |
| `parakeet` | 0.94 GB (and the same repo ships `tdt_ctc-110m-q4_k.gguf` at **0.13 GB**) | yes | **comfortably in scope** |
| `lfm2moe` | 4.84 GB (Q4_0) | yes | in scope |
| `maple` | 4.98 GB (TQ1_0) | yes | in scope |
| `mellum` | 8.07 GB (Thinking Q4_K_M) | yes | upper edge |
| `laguna` | **20.27 GB — only Q4_K_M and a 66.93 GB BF16 exist** | yes | **out of scope at any published quantization.** Not a port decision; an artifact fact. |
| `nemotron_h_moe` | **18.09 GB**; the five non-I-quant options are Q2_K 18.16, Q3_K_S 18.19, Q4_0 18.27, Q3_K_M 19.07, Q8_0 33.59 | Q2_K/Q3_K/Q4_0 yes, I-quants no | **out of scope**, and additionally exceeds this dev box |

Two of the six decoder ports therefore cannot be qualified on small-model terms **no matter how the
code goes**, because no small artifact exists to qualify. That is a finding about the ecosystem, not
about our runtime, and it is the sort of thing that has to be said before effort is spent.

## 5. Corrections to `docs/QXOTIC-PARITY.md`

The plan I wrote this morning was built on their codecs' expectations. The real headers refute three
parts of it. Recording the refutations rather than quietly editing:

1. **`lfm2moe` was placed as an independent stage. It is not.** The artifact carries
   `blk.exp_probs_b.bias` and `expert_gating_func=2`, so it depends on the sigmoid-routing
   prerequisite exactly as `laguna` does. Four MoE ports hang off that one unit, not three, and the
   plan's claim that it needed "the MoE branch inside the existing LFM2 block dispatch" and nothing
   more was wrong.
2. **"Stage 1 — the free one: `minicpm`" has no subject.** No artifact declares `minicpm`; the one
   in their manifest reports `llama` and we already load it. The stage is deleted, not deferred.
3. **The ordering was backwards on size.** I put speech last and chat first. The speech artifacts are
   0.005–0.94 GB and the chat MoEs are 4.8–20.3 GB with two of them unqualifiable at any published
   quant. Smallest-first says speech is *first* by size — while costing a new model class, which is
   the genuine reason to weigh it carefully rather than the "deliberately last" I asserted.

4. **§4's `bailingmoe3` reading was taken from their codec, and the header shows it described a
   third of the model.** "MLA plus Q-LoRA" is confirmed as *present* — `attn_q_a` with
   `attn_q_a_norm` and `attn_q_b` at `q_lora_rank` 256, `attn_kv_a_mqa` with `attn_k_b`/`attn_v_b`
   at `kv_lora_rank` 512 — but it is present on **6 of 24 layers**. The other 18 carry
   `ssm_conv1d_{q,k,v}`, `ssm_a`, `ssm_beta`, `ssm_dt.bias`, `ssm_f_a`, `ssm_g_a` and `ssm_norm`
   with `kda.*` metadata. It is a hybrid on a 4-layer interval, and the stage estimate the plan's
   Stage 4 carried was made against the attention third of it.
   Two further details the codec reading would have got backwards, both of which change the work:
   the file ships the **absorbed** MLA form (`attn_k_b`/`attn_v_b`, no unsplit `attn_kv_b`), which
   is precisely the form our DeepSeek-V2 decoder does *not* implement
   (`Deepseek2Weights.java:32-34`, `Deepseek2Config.java:31-32`); and the hybrid interval is encoded
   as a **per-layer `attention.head_count_kv` array** (`[0,0,0,1]` repeated six times), where
   `Qwen35Config.java:177` reads that key as a single `requiredInt`.

What survives unchanged: `maple` and `mellum` really are the config-only ports (their tensor sets
are **byte-identical to each other** and to `qwen3moe` — `attn_{q,k,v,output,norm}`, `attn_{q,k}_norm`,
`ffn_norm`, `ffn_gate_inp`, `ffn_{gate,up,down}_exps`, and **neither carries `exp_probs_b`**), and
Mamba-2 really is the one large piece of new math.

## 6. Feasibility verdicts

Four tiers, by what the evidence says — not by appeal.

### Tier A — feasible now, no new primitive (2 ports)

**`maple`**, then **`mellum`**. Config parse plus chat template, over the existing Qwen3-MoE path.
`maple` first: it is smaller (4.98 vs 8.07 GB), its quantization is already implemented, and its
gating is the softmax we have. `mellum` adds only YaRN and SWA, both present.

- Gate: logits match an external oracle at a tolerance fixed before the comparison, then the two-arm
  qualification record. **`expert_gating_func` must be read and asserted even for `maple`, where the
  value happens to match our default** — an assumed default is how a routing change becomes
  invisible.
- Risk: these two are also the test of the "config-only" reading. If either needs kernel work, that
  is the finding, and Tier B's estimate is revised before it starts.

### Tier B — feasible after one shared prerequisite (2 ports)

Build **sigmoid expert gating with `exp_probs_b` bias-corrected top-k**, switchable observably, in
`ops/ExpertRouting.java`. Then:

**`lfm2moe`** — 4.84 GB, short-conv already implemented, 2 leading dense blocks, 32/4 experts. The
best-value port on the whole list after Tier A, and it was mis-ordered in the plan.

**`laguna`** — code-feasible (every other primitive is present), but **not qualifiable**: 20.27 GB is
the smallest artifact that exists. Implement it only as a deliberate choice to support a model we
cannot qualify on our own terms, or leave it and say why. **Recommend: leave it, and record the
artifact fact.**

- Gate for the prerequisite: expert selection matches the oracle's per token on a real
  `exp_probs_b` artifact, and turning the bias correction off visibly changes the selection.

### Tier C — feasible, but each needs a recurrence we have not written (2 ports)

Both are hybrid stacks: a transformer attention path on some layers and a linear-recurrent path on
the rest. The attention paths are within reach; the recurrences are the work. Ordered by what each
buys and **not** by what it costs — the one ranked first needs four pieces to the other's one, and
is still the better port, because its artifact is the only one here that can be qualified.

**`bailingmoe3`** — 4.82 GB, MIT, first-party (`inclusionAI/Ling-3.0-tiny-GGUF` at `01b850e29ed5`).
**The only port in this tier whose artifact can be qualified on our own terms**, and the last
decoder port in §7's ordering that can — after it, every remaining decoder is blocked on a published
artifact rather than on us. Four pieces, none of them large:

1. **KDA on 18 of 24 layers** — the recurrence, and the only genuinely new math. It is a *variation
   on the gated delta rule we already ship*, not a new family: qwen35's GDN path already loads
   `ssm_beta`, `ssm_dt.bias`, `ssm_a`, `ssm_norm` and `attn_gate` (`Qwen35Weights.java:204-240`),
   and KDA carries all five. What differs is
   **three per-projection convolutions** (`ssm_conv1d_{q,k,v}`, `ssm.conv_kernel` 4) where GDN has
   one fused `ssm_conv1d`, **two separate gate projections** (`ssm_f_a`, `ssm_g_a`) where GDN has
   `ssm_alpha`/`ssm_ba`, and two clamping knobs with no analogue here at all — `kda.safe_gate` 1 and
   `kda.gate_lower_bound` -5.0.
2. **The absorbed MLA form on the other 6 layers.** The file ships `attn_k_b`/`attn_v_b` and **no
   unsplit `attn_kv_b`** — the exact inverse of what our DeepSeek-V2 decoder reads, which documents
   that it implements the *decompressing* form precisely because DeepSeek-Coder-V2-Lite ships only
   that (`Deepseek2Weights.java:32-34`, `Deepseek2Config.java:31-32`). The latent compression and
   its norm carry over; the decompression does not.
3. **Query-side Q-LoRA.** `attn_q_a` [1536, 256], `attn_q_a_norm`, `attn_q_b` [256, 3072] at
   `q_lora_rank` 256. Small, and absent from the tree: no file reads `attn_q_a` or any
   `q_lora_rank`, where `kv_lora_rank` is read at `Deepseek2Config.java:137`.
4. **Grouped expert top-k.** 128 experts / 8 used, restricted to 4 of 8 expert groups
   (`expert_group_count` 8, `expert_group_used_count` 4). Neither key is read anywhere in the tree.
   Everything *around* it is already built and needs only to be reached from a second architecture:
   `FlintRouting.java:22-32` is sigmoid gating with `exp_probs_b` bias-corrected top-k,
   `expert_weights_scale` and `expert_weights_norm` (`FlintConfig.java:92,130-133`,
   `FlintWeights.java:158-159`), and `leading_dense_block_count` (`FlintConfig.java:126`) — which is
   Tier B's prerequisite, implemented. It sits in the `flint` package rather than in
   `ops/ExpertRouting.java`, whose two entry points are both softmax
   (`ExpertRouting.java:53,101`), so Tier B's "build it in `ops`" still stands as the generalisation
   step; bailingmoe3 is its second caller and the reason to do it.

One intake detail that is cheap to get wrong: the hybrid interval is **not** a
`full_attention_interval` key as in qwen35. It is encoded as a per-layer `attention.head_count_kv`
**array**, `[0,0,0,1]` repeated six times — so layers 3, 7, 11, 15, 19 and 23 are the MLA layers and
the KDA layers report 0 KV heads. `Qwen35Config.java:177` reads that key as a single `requiredInt`,
so this is a loader change and not only a graph change, and a `head_count_kv` of 0 is a value no
config in the tree accepts today.

- Gate: the KDA recurrence **alone** against a reference's state trajectory step by step before any
  full model runs, exactly as for Mamba-2 below. Then, separately, grouped selection matching the
  oracle's per token with the group restriction *observably* switchable — a grouped top-k that
  silently degrades to ungrouped picks the same experts often enough to look correct.
- Kill-criterion: if the KDA recurrence does not reach the oracle's trajectory tolerance, stop at
  that gate and publish the null. Do not proceed to the attention path to produce a model-level
  number, which would hide a recurrence error behind a plausible first token.

**`nemotron_h_moe`** — Mamba-2 selective scan, then the hybrid stack, on top of Tier B's routing.
The config surface is already parsed; the recurrence is not written. Gate the **scan alone** against
a reference's state trajectory step by step before any full model runs — a model-level comparison
would hide a recurrence error behind a plausible first token.

Same artifact caveat as `laguna`, worse: 18.09 GB smallest, and 9 of its 14 small quants are
I-quants we cannot read. **Feasible to implement, not feasible to qualify small** — which is the
whole reason `bailingmoe3` is ranked ahead of it despite needing four pieces to its one.

### Tier D — feasible, smallest artifacts, but each needs a new model class (3 ports)

Ordered by what they cost us, which is the inverse of their size:

**`kokoro`** (0.16 GB) — reuses `TextToSpeechModel` and the Soprano DSP (`SopranoIstft.java`,
`audio/Radix2Fft.java`). Blocked on G2P: its vocabulary is 178 phonemes, so text cannot reach it
without a phonemizer, and qxotic's own is an external component (`NOTICE:6-9`). **The licensing and
packaging question must be answered before the porting question** (`CLAUDE.md` §5). Also needs a
companion-artifact concept for the per-voice GGUFs — a 1-tensor file per voice, which our catalog has
no shape for today.

**`inflect-v2`** (0.005 GB) — same TTS class, same G2P blocker, plus a worse intake problem: the
artifact publishes **no `inflect-v2.*` metadata keys at all**, so its configuration is implicit and
must come from the reference implementation rather than the header. Lowest priority of the three.

**`parakeet`** (0.94 GB, or 0.13 GB for the 110M sibling) — needs the most and is worth the most: a
**transcription model class we do not have**, plus a conformer encoder and an LSTM predictor, plus
the TDT decoding loop. The mel front end is within reach of the existing FFT. Its header is
unusually self-describing (every preprocessor and encoder parameter is a metadata key), which makes
it the *easiest to get right* of the three once the SPI exists.

Decide ASR against `docs/MODALITY-EXPANSION.md`'s Whisper work and pick the first implementation on
**measured WER against a published test set**, not on which one reads better.

## 7. Recommended order, and what it buys

| order | port | new primitive | artifact | qualifiable small |
| --- | --- | --- | --- | --- |
| 1 | `maple` | none (+ read `expert_gating_func`) | 4.98 GB | yes |
| 2 | `mellum` | none | 8.07 GB | edge |
| 3 | sigmoid + `exp_probs_b` routing | the one shared unit | — | — |
| 4 | `lfm2moe` | none beyond 3 | 4.84 GB | yes |
| 5 | `bailingmoe3` | KDA recurrence, absorbed MLA, Q-LoRA, grouped top-k | 4.82 GB | yes |
| 6 | transcription model class | new SPI | — | — |
| 7 | `parakeet` | conformer, LSTM, TDT | 0.13–0.94 GB | yes |
| 8 | `kokoro` | G2P licensing decision first, companion artifacts | 0.16 GB | yes |
| — | `laguna`, `nemotron_h_moe` | Mamba-2 for the latter | 20.27 / 18.09 GB | **no** |
| — | `minicpm` | **already supported** | — | already in the catalog |

Steps 1-4 take us from 10 shared architectures to 13, at no new math. **Step 5 makes 14 and closes
the decoder sweep**: `bailingmoe3` is the last decoder port on the list with an artifact small enough
to qualify, so after it, every remaining decoder is blocked on a published artifact rather than on us.
It is placed after step 4 and not earlier because it is the first port here that needs a recurrence
written, and it is placed before the modality work because it extends a decoder family we already
have rather than requiring a new SPI. Steps 6-8 then open a modality we have zero coverage in, at
the smallest artifact sizes on the list. `laguna` and `nemotron_h_moe` are where "support all qxotic
models" stops being reachable on our own terms, and the reason is the published artifacts, not the
runtime.

## 8. What this does not establish

- **No performance claim.** Nothing here was run. Whether our graph would decode any of these
  faster or slower than jinfer is unmeasured, and no tier above may assert it.
- **`bailingmoe3` is unassessed, not assessed as hard.** Its artifact was not found in their
  manifest; locating a Ling 3 GGUF is the first step, and the MLA-plus-Q-LoRA reading in
  `QXOTIC-PARITY.md` §4 came from their codec, not from a header, so it stays unconfirmed.
- **Oracle availability: RESOLVED 2026-10-24, and read from upstream rather than from the clone.**
  All nine architectures exist in `ggml-org/llama.cpp` at `3f03b17724ad8b69040825f45c43c59b092a0244`
  as real `LLM_ARCH_*` registrations in `src/llama-arch.cpp` — including `maple` (line 70),
  `bailingmoe3` (119), `laguna` (124) and `mellum` (165).

  **The local clone would have said otherwise.** It was pinned at `a58222229` (2026-07-15), **1839
  commits behind**, and at that revision `maple`, `mellum`, `laguna` and `bailingmoe3` are all
  absent — `mellum` appears only as a pretokenizer in `llama-vocab.cpp`. Checking the checkout on
  disk would have recorded four false negatives and plausibly cancelled four ports.

  So the default for any cross-implementation claim is: bring the oracle current, then record the
  revision beside the result. `scripts/oracle-revision.sh` reports it and **fails on a stale or
  dirty clone**; `--update` fast-forwards first. An oracle with local edits is not the revision its
  sha names, which is why the script refuses that too.
- **The 110M parakeet sibling was sized but not probed.** Only `tdt-0.6b-v3-q8_0.gguf` had its
  header read; `tdt_ctc-110m-q4_k.gguf` is assumed to share the architecture on the strength of its
  filename, which is exactly the kind of assumption this document exists to avoid. Probe it before
  it is relied on.
