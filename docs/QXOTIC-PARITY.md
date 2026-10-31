# qxotic architecture parity: what they load, what we load, and what it takes to close the gap

Written 2026-10-22 against `references/qxotic` at `6f08d93ec` (tag `jinfer-v0.3.1`), fast-forwarded
329 commits from `c14552a3` on this date. **Every architecture string, tensor name and capability
below was read out of one of the two codebases at the cited `path:line`.** Nothing here is recalled
from a model card, a README summary or memory; where the two disagree, the code wins and the
disagreement is written down.

qxotic is Apache-2.0 (`references/qxotic/LICENSE:1`, `NOTICE:4`), so its sources may be studied and,
if a translation is made, it must retain the upstream copyright and NOTICE. That satisfies the
runtime licensing rule in `CLAUDE.md` §5. **Studying is not porting**: every number this plan
produces has to be measured on our own harness regardless of what their benchmarks say.

> **Superseded in part.** `docs/QXOTIC-PARITY-FEASIBILITY.md` executed this document's Stage 0 on
> 2026-10-22 by reading the real GGUF headers, and it refutes three things below: `lfm2moe` is not
> independent of the `exp_probs_b` stage, the `minicpm` stage has no artifact and is deleted, and the
> size evidence inverts the speech-last ordering. **Read the feasibility document for anything about
> ordering, effort or artifacts.** What stands here is the coverage comparison in §1-§3 and the
> per-architecture reading in §4 — with one exception, added 2026-10-31 (#288): §4's `bailingmoe3`
> row was read from their codec and not from an artifact, because no artifact existed then. One does
> now, and it shows the row described about a third of the model. The corrected reading is §5.4 of
> the feasibility document; the row below is marked and Stage 4 is rewritten accordingly.

## 1. What qxotic loads

Their dispatch key is `general.architecture`, declared per port by
`ModelProvider.architectures()` (`jinfer/jinfer-chat/src/main/java/com/qxotic/jinfer/chat/ModelProvider.java:39`).
Read from each provider:

| `general.architecture` | module | model class implemented | provider |
| --- | --- | --- | --- |
| `bailingmoe3` | `jinfer-bailingmoe3` | language | `BailingMoe3Provider.java:21` |
| `gemma4` | `jinfer-gemma4` | language | `Gemma4Provider.java:22` |
| `gpt-oss` | `jinfer-gptoss` | language | `GptOssProvider.java:22` |
| `inflect-v2` | `jinfer-inflect2` | speech synthesis | `Inflect2Provider.java:26` |
| `kokoro`, `kokoro-voice` | `jinfer-kokoro` | speech synthesis | `KokoroProvider.java:18`, `Kokoro.java:31-32` |
| `laguna` | `jinfer-laguna` | language | `LagunaProvider.java:21` |
| `lfm2`, `lfm2moe` | `jinfer-lfm2` | language, embedder, reranker | `Lfm2Provider.java:26` |
| `granite` | `jinfer-llama` | language | `GraniteProvider.java:22` |
| `llama`, `minicpm`, `mistral3`, `smollm3` | `jinfer-llama` | language | `LlamaProvider.java:23` |
| `maple` | `jinfer-maple` | language | `MapleProvider.java:21` |
| `mellum` | `jinfer-mellum` | language | `MellumProvider.java:25`, `Mellum.java:48` |
| `nemotron_h`, `nemotron_h_moe` | `jinfer-nemotronh` | language | `NemotronHProvider.java:22` |
| `parakeet` | `jinfer-parakeet` | transcription | `ParakeetProvider.java:18`, `Parakeet.java:32` |
| `qwen3` | `jinfer-qwen3` | language, embedder, reranker | `Qwen3Provider.java:35` |
| `qwen35`, `qwen35moe` | `jinfer-qwen35` | language | `Qwen35Provider.java:24` |

**22 architecture strings across 15 provider services in 14 modules** — `jinfer-llama` registers
two, `LlamaProvider` and `GraniteProvider`. Two corrections to their own README while reading it:

- `maple` is implemented (`jinfer-maple/.../Maple.java:37`, 751 LOC) but **absent from the family
  table** at `jinfer/README.md:29-45`. The code is the authority; the table undercounts by one.
- The README table reads as the model list, but their actual dispatch is on architecture strings,
  and `minicpm` is one of them while `MiniCPM5` checkpoints report `llama`
  (`LlamaProvider.java:48-52`). An architecture-string comparison is therefore the only comparison
  that means anything; a family-name comparison would double-count.

## 2. What we load

Our dispatch is `PureJavaBackend.java:477-531`, which branches on `general.architecture` and falls
through to the Llama graph, whose admissible set is the `DecoderArchitecture` enum
(`backend-java/.../llama/DecoderArchitecture.java:31-46`). `parse` refuses anything not listed, on
purpose — "Resolves only architectures whose complete semantics are implemented"
(`DecoderArchitecture.java:59`).

| route | `general.architecture` | source |
| --- | --- | --- |
| Llama graph | `llama`, `gemma`, `gemma2`, `gemma3`, `hunyuan-dense`, `phi3`, `mistral3`, `qwen2`, `qwen3`, `qwen3moe`, `granite`, `smollm3`, `gemma-embedding` | `DecoderArchitecture.java:31-46` |
| dedicated | `gemma-embedding2` | `PureJavaBackend.java:483` |
| dedicated | `gemma4` | `PureJavaBackend.java:487` |
| dedicated | `qwen35`, `qwen35moe`, `qwen3next` | `PureJavaBackend.java:498-500` |
| dedicated | `gpt-oss` | `PureJavaBackend.java:505` |
| dedicated | `deepseek2` | `PureJavaBackend.java:508` |
| dedicated | `gemma3n` | `PureJavaBackend.java:511` |
| dedicated | `lfm2` | `PureJavaBackend.java:514` |
| dedicated | `bert`, `nomic-bert` | `PureJavaBackend.java:517` |
| TTS backend | `audiocpp` (Soprano) | `soprano/SopranoBackend.java:97` |
| non-GGUF | CACT / Needle2 container | `PureJavaBackend.java:465-473` |
| non-GGUF | HuggingFace safetensors (gpt-oss, qwen2) | `PureJavaBackend.java:420-463` |
| **in flight** | `modern-bert` (Laya) | `bert/ModernBertConfig.java:255`; **not wired into the dispatch yet** |

**24 strings through the `PureJavaBackend` dispatch, plus `audiocpp` through the Soprano TTS
backend, plus two container formats they do not read at all.**

## 3. The gap, in both directions

Overlap is 10 strings: `llama`, `granite`, `mistral3`, `smollm3`, `qwen3`, `qwen35`, `qwen35moe`,
`gemma4`, `gpt-oss`, `lfm2`.

**Theirs that we do not load — 12 strings; 8 of their modules are absent from our tree entirely.**
This is the parity target, and it is finite:

`bailingmoe3`, `laguna`, `maple`, `mellum`, `nemotron_h`, `nemotron_h_moe`, `lfm2moe`, `minicpm`,
`parakeet`, `kokoro` (+ `kokoro-voice`), `inflect-v2`.

Two of the twelve sit inside modules we already cover in part: `lfm2moe` beside our `lfm2`, and
`minicpm` beside our `llama`.

**Ours that they do not load — 15 strings.** Worth stating because "support all qxotic models" is a
bounded catch-up and not a general deficit: `gemma`, `gemma2`, `gemma3`, `gemma3n`,
`gemma-embedding`, `gemma-embedding2`, `phi3`, `qwen2`, `qwen3moe`, `hunyuan-dense`, `deepseek2`,
`bert`, `nomic-bert`, `qwen3next`, `audiocpp`, plus the CACT/Needle2 and safetensors intake paths.

## 4. Why each one is missing — the specific absent primitive

The distinguishing tensor set of each missing architecture was extracted from its own codec, and
checked against what our `backend-java` already names. **Every gap below is "not ported yet", and
for all but one the missing piece is named. None is "cannot be done".**

| missing | distinguishing tensors (read from their codec) | what we already have | what is actually absent |
| --- | --- | --- | --- |
| `minicpm` | none — same graph as `llama`, differs only in chat template (`LlamaProvider.java:43-44`) | `ChatTemplate.MINICPM5_NO_THINK` (`models-runtime/.../chat/ChatTemplate.java:53`), `ToolSyntax.MINICPM5`, `BpePreTokenizer.java:187` | **one enum entry.** `DecoderArchitecture.parse` refuses the string (`DecoderArchitecture.java:60-74`); the graph and the template both already exist |
| `maple` | `attn_q_norm`, `attn_k_norm`, `ffn_{gate,up,down}_exps`, `ffn_gate_inp` | all of these (`qwen35/Qwen35Weights.java`, `qwen3moe` via the Llama graph) | config parse + template. Structurally a `qwen3moe` with q/k norm — the smallest real port on this list |
| `mellum` | as `maple` plus `attention.sliding_window`, `attention.{key,value}_length` | sliding window (`gptoss/GptOssConfig.java`, `gemma4/Gemma4Config.java`) | config parse + template only |
| `lfm2moe` | LFM2 short-conv blocks plus expert tensors | `lfm2/Lfm2ShortConv.java`, `Lfm2ForwardPass.java`; MoE from `qwen35` | the MoE branch inside the LFM2 block dispatch |
| `laguna` | `attn_gate`, `ffn_{gate,up,down}_shexp`, `exp_probs_b.bias`, mixed dense+MoE blocks | `attn_gate` (`qwen35/Qwen35Weights.java`), `ffn_gate_shexp` (`qwen35`, `deepseek2/Deepseek2Weights.java`) | **`exp_probs_b` — bias-corrected sigmoid expert routing. We implement it nowhere: 0 occurrences across every `*/src/main`.** |
| `nemotron_h`, `nemotron_h_moe` | `ssm_in`, `ssm_out`, `ssm_conv1d`, `ssm_dt.bias`, `ssm_norm`; shared experts; `exp_probs_b.bias` | a **different** SSM: ours is GatedDeltaNet for Qwen3-Next (`ssm_a`, `ssm_alpha`, `ssm_ba`, `ssm_beta`, `ssm_conv`, `ssm_dt`, `ssm_norm`, `ssm_out` in `qwen35/Qwen35Weights.java`), plus the hybrid per-layer dispatch and recurrent-state cache it needs | **Mamba-2 selective scan.** `ssm_in` and `ssm_conv1d` appear in no source of ours. The *scaffolding* for a hybrid recurrent/attention stack is already proven by Qwen3-Next and LFM2; the recurrence itself is new |
| `bailingmoe3` ⚠ | MLA: `attn_kv_a_mqa`, `attn_kv_a_norm`, `attn_q_a`, `attn_q_b`, `attn_k_b`, `attn_v_b`; `attn_gate`; shared experts; `exp_probs_b.bias` | MLA (`deepseek2/Deepseek2Config.java`, `Deepseek2ForwardPass.java`, `Deepseek2Weights.java`), `attn_gate`, shared experts | **Q-LoRA (`attn_q_a`/`attn_q_b`)** — our MLA reads a full `attn_q.weight` (`Deepseek2Weights.java:131`), i.e. the DeepSeek-V2-Lite shape — **plus `exp_probs_b`** |

⚠ **This row is incomplete and was read from their codec, not an artifact.** It describes 6 of the
model's 24 layers. The real header (`headers.json` entry 11) shows the other 18 are **KDA** linear
attention, the MLA form shipped is the **absorbed** one our decoder does not read, grouped expert
top-k is also required, and the hybrid interval is a per-layer `head_count_kv` array. Full reading:
`docs/QXOTIC-PARITY-FEASIBILITY.md` §5.4 and §6 Tier C.
| `kokoro` (+ `kokoro-voice`), `inflect-v2` | TTS stacks: PL-BERT text encoder, prosody predictor, vocoder decoder, G2P (`jinfer-kokoro`, 3029 LOC; `jinfer-inflect2`, 2802 LOC incl. `TextNormalizer`, `Lexicon`, `LetterToSound`) | **the model class exists**: `models-api/.../TextToSpeechModel.java:21`, `PcmAudio`, `AudioStream`, `WavEncoder`, and a shipping TTS backend in `soprano/` with its own ISTFT (`SopranoIstft.java`) and FFT (`audio/Radix2Fft.java`) | the two stacks themselves, and for both a grapheme-to-phoneme front end we have no equivalent of |
| `parakeet` | FastConformer encoder + TDT decoder (`ParakeetEncoder.java`, `ParakeetTdt.java`, 1533 LOC) | FFT (`audio/Radix2Fft.java`) for the mel front end | **the whole model class.** No transcription SPI exists: 0 matches for a transcription/speech-recognition interface across every `*/src/main`, where TTS has `TextToSpeechModel`. Conformer appears in no source of ours |

### The two shared prerequisites

Three of the eight absent modules are blocked on the same two primitives, which is what makes the
ordering below non-arbitrary rather than a wish list:

1. **`exp_probs_b` bias-corrected expert routing** — needed by `laguna`, `bailingmoe3`,
   `nemotron_h_moe`. Absent everywhere today.
2. **Mamba-2 selective scan** — needed by `nemotron_h` and `nemotron_h_moe`, and by nothing else
   on the list.

`parakeet` is blocked on a third thing that is not an architecture at all: a transcription model
class. `docs/MODALITY-EXPANSION.md` already works that problem from the Whisper side with headers
read off real artifacts; Parakeet should be decided **after** that SPI exists, not as the reason to
create it.

## 5. Plan

Dependency order and gates only — no dates, per `CLAUDE.md`. Each stage's gate is what has to be
true before the next one starts, and **a stage that fails its gate is reported as a null, not
reframed**.

Every stage shares one evidence rule, which is the point of doing this on our own harness rather
than reading their benchmark tables: **a new architecture is not "supported" until a real
checkpoint of it decodes correctly against an external oracle.** The oracle discipline is the one
that worked for Laya (`projects/models-final-integration/experiments/laya-modern-bert/`): write the
reference's own output to a fixture, compare per stage, and read the GGUF header by range-fetch
before writing any Java — the rule from `docs/MODALITY-EXPANSION.md` and the
`read-gguf-headers-before-implementing` finding, which caught five blockers before any EC2 spend.

### Stage 0 — intake, before any Java

For each of the 12 target strings, range-fetch the header of a real published GGUF and record
tensor roots, `general.*`, the per-architecture keys, and the quantizations actually published.

- **Gate:** a header record per architecture, committed, naming the exact repo and file. An
  architecture with no reachable artifact drops out of the plan here and is written down as
  "no artifact", not as "unsupported".
- This stage is also where the `maple`/`mellum`/`laguna` structural claims in §4 get confirmed
  against a real file rather than against their codec's expectations.

### Stage 1 — the free one

`minicpm`: add the enum entry, select the existing MiniCPM5 template, qualify.

- **Gate:** a MiniCPM (not MiniCPM5) checkpoint decodes and passes the two-arm qualification. If
  Stage 0 finds no such artifact worth loading, this stage is deleted rather than carried.
- Ordered first because it is one line and it proves the Stage-0 intake format end to end on
  something with no new math in it.

### Stage 2 — the config-only ports

`maple`, then `mellum`. Both are the Llama/Qwen3-MoE graph with tensors we already read.

- **Gate per port:** logits match an external oracle within a tolerance fixed *before* the
  comparison, on a real checkpoint; then the two-arm qualification record.
- These two are the test of whether §4's "config parse + template only" reading is right. **If
  either needs new kernel work, that is the finding**, and the estimates for every later stage get
  revised on it before continuing.

### Stage 3 — `exp_probs_b` routing

Implement bias-corrected sigmoid expert routing once, where the MoE gate already lives, and make it
**observably switchable** so an ablation of it cannot silently do nothing — the toggle-that-does-
nothing failure is already written down in `CLAUDE.md` §1.

- **Gate:** routing decisions match the oracle's chosen experts per token on a real
  `exp_probs_b`-carrying checkpoint, and flipping the correction off visibly changes the selection.

### Stage 4 — `laguna`, then `bailingmoe3`

**Rewritten 2026-10-31 (#288).** The original staged `bailingmoe3` as "Stage 3 plus Q-LoRA", a
one-branch job. That estimate was made against the attention third of the model.

- `laguna` first: it needs Stage 3 and `attn_gate`, both then in hand. The feasibility document
  recommends **not** porting it at all — 20.27 GB is its smallest published artifact, so it cannot
  be qualified on our own terms.
- `bailingmoe3` second, and it is **four pieces, not one**: the KDA recurrence on 18 of 24 layers,
  the absorbed MLA form (`attn_k_b`/`attn_v_b`, which `Deepseek2Weights.java:32-34` documents we
  deliberately do not read), query-side Q-LoRA, and grouped expert top-k. It is **Tier C** in the
  feasibility document, alongside `nemotron_h_moe`, and ranked ahead of it because its artifact is
  4.82 GB and therefore qualifiable. Read §6 Tier C there for the per-piece breakdown and the gate;
  it supersedes this bullet.
- **Gate per port:** the recurrence alone against an oracle state trajectory *first*, then oracle
  parity on the full model, then qualification. Q-LoRA lands as a shape branch in the existing MLA
  forward pass, not as a second MLA implementation; the absorbed MLA form does not — that is a
  second decompression path and must be written.

### Stage 5 — `lfm2moe`

The MoE branch inside the existing LFM2 block dispatch.

- **Gate:** oracle parity on a real `lfm2moe` checkpoint, and the dense `lfm2` path unchanged —
  asserted, by re-running the existing `lfm2` qualification and requiring the same verdict.

### Stage 6 — Mamba-2 and `nemotron_h`

Selective scan, then the hybrid stack, then `nemotron_h_moe` on top of Stage 3.

- **Gate:** the scan alone matches a reference implementation's state trajectory step by step
  before any full model runs. A model-level-only comparison here would hide an error in the
  recurrence behind a correct-looking first token, which is exactly the class of bug the Laya
  per-stage oracle was built to catch.
- Largest single piece of new math on this list. If it slips, Stages 1-5 still close 6 of the 12
  strings and the plan says so rather than reporting partial work as parity.

### Stage 7 — speech, deliberately last and deliberately not first

- **`kokoro` and `inflect-v2`** reuse `TextToSpeechModel` and the Soprano DSP, and each needs a
  G2P front end we have nothing like. Their own G2P is an *external* dependency on both ports
  (`NOTICE:6-9` names an external program for `inflect2` and an external component for `kokoro`),
  so **the licensing and packaging question has to be answered before the porting question**, per
  `CLAUDE.md` §5.
- **`parakeet`** waits on a transcription SPI. Decide that from `docs/MODALITY-EXPANSION.md`'s
  Whisper work, then choose Parakeet or Whisper as the first implementation **on measured WER
  against a published test set, not on which one a README praises**.
- **Gate:** for TTS, a licensing determination in writing before any code; for ASR, the SPI merged
  and one architecture measured against an external WER oracle.

## 6. What this plan cannot tell us yet

Stated plainly rather than left implied:

- **Nothing here is a performance claim.** This is a coverage comparison read out of two source
  trees. Whether our graph decodes any of these architectures *faster or slower* than theirs is
  unmeasured, and no stage above is allowed to assert it without a run on our own harness.
- **Artifact availability is unverified for all 12.** Stage 0 exists because the architecture
  strings in their code prove only that a loader was written, not that a published checkpoint of it
  is reachable. Some of these may have no artifact we can qualify, and that is a finding, not a
  failure.
- **Their README and their code disagree by one port** (`maple`). Where any later doc cites qxotic
  coverage, it must cite the provider, not the table.
- **The LOC figures in §4 size the port, not the difficulty.** `jinfer-kokoro` at 3029 LOC and
  `jinfer-nemotronh` at 1357 LOC are not in that ratio of effort for us, because we already have
  the DSP for one and none of the recurrence for the other.
