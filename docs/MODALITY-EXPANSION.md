# Expanding past text: what it takes, from the real headers

Written 2026-10-09. **Every architectural fact below was read out of the artifact's own GGUF header
by range-fetch, not from a model card and not from memory.** That is the rule this repo learned the
expensive way, and it is what makes the effort estimates below worth anything.

## Where the catalog actually stands on modality

101 qualified models. **One** of them is not text: `walkingcat_soprano_1_1_80m` TTS, qualified at
two quantizations. Across all **454** candidates there are only five non-text entries, and three of
those are unqualified.

| modality | qualified | candidates | runtime support |
| --- | --- | --- | --- |
| text generation / chat | 66 | 384 | 20 architectures |
| text embedding | 30 | 60 | bert, nomic-bert, qwen3, gemma-embedding, lfm2 |
| reranking | 2 | 3 | deberta-v2 |
| text-to-speech | 2 | 4 | soprano only |
| **speech-to-text** | **0** | **0** | **none** |
| **image understanding** | **0** | 1 (claim is wrong, see below) | **none** |
| **OCR** | **0** | **0** | **none** |
| **video understanding** | **0** | 1 (same entry) | **none** |

### A correctness bug to fix before anything else

`gemma_3n_e2b_it_q8_0` declares `image-understanding`, `audio-understanding` and
`video-understanding`. Its artifact cannot do any of them. Read from the file:
`ggml-org/gemma-3n-E2B-it-GGUF/gemma-3n-E2B-it-Q8_0.gguf` has **727 tensors whose roots are only**
`blk`, `altup_proj`, `altup_unembd_proj`, `per_layer_model_proj`, `per_layer_proj_norm`,
`per_layer_token_embd`, `token_embd`, `output_norm` — no vision tower, no audio tower, no
projector. The repo publishes exactly two GGUFs (`Q8_0`, `f16`) and **no `mmproj`**, so the
companion weights do not exist there to point at.

Multimodal weights in the ggml convention ship as a *separate* projector GGUF. The entry should
either drop the three claims or gain a real projector file. Leaving it is the same class of error as
claiming a `backendVersion` a user can check and find false.

## Path 1 — speech-to-text, via Whisper

`handy-computer/whisper-tiny-gguf`, Apache-2.0, 38M, `general.architecture = whisper`, 169 tensors.
**The header specifies the entire signal frontend**, which is unusual and very welcome:

| key | value |
| --- | --- |
| `stt.frontend.type` / `normalize` | `mel` / `whisper_logmel` |
| `num_mels` / `n_fft` / `win_length` / `hop_length` | 80 / 400 / 400 / 160 |
| `window` / `pad_mode` / `center` | `hann_periodic` / `reflect` / true |
| `sample_rate` / `chunk_length` / `n_samples` / `nb_max_frames` | 16000 / 30 s / 480000 / 3000 |
| `mel_norm` / `f_min` / `f_max` | `slaney` / 0.0 / 8000.0 |
| `stt.whisper.decoder.{d_model, ffn_dim, n_heads, max_target_positions}` | 384 / 1536 / 6 / 448 |
| `stt.whisper.decoder.activation` | `gelu` |
| `stt.capability.{timestamps, translate, lang_detect}` | all true |

**What we already have:** quantized matmul for every type in the file, the GGUF reader, a
transformer block, KV cache, tokenizer plumbing.

**What is genuinely new, and must not be understated:**

1. **An STFT + log-mel frontend.** Pure Java, no model weights, fully specified above. Self-contained
   and testable against a reference mel for a fixed WAV.
2. **Cross-attention.** Every decoder in this repo is self-attention only. Whisper's decoder attends
   to encoder output, which is a new block shape and a second KV cache with different lifetime
   (computed once per 30 s window, not per token).
3. **A convolutional encoder stem** — two conv1d layers before the transformer.
4. **A new gate.** `correctAnswerRate` does not describe transcription. This needs WER against
   pinned audio with a committed reference transcript, which is a new harness and a new policy
   version, in the shape of the existing `speech-oracle-streaming-latency-v1`.

Smallest-first order: `tiny` (38M) → `base` → `small`. The tiny model is the one to prove the
frontend and cross-attention on, and it is 38M, so iteration is cheap.

## Path 2 — vision and OCR, via Qwen3-VL

`unsloth/Qwen3-VL-2B-Instruct-GGUF`, Apache-2.0, ships `mmproj-{BF16,F16,F32}.gguf` beside the
weights. **Two files, two architectures.**

Text tower — `general.architecture = qwen3vl`, **not `qwen3`**, so our existing Qwen3 decoder does
not load it:

| key | value |
| --- | --- |
| `block_count` / `embedding_length` / `feed_forward_length` | 28 / 2048 / 6144 |
| `attention.head_count` / `head_count_kv` | 16 / 8 |
| `attention.key_length` / `value_length` | 128 / 128 |
| `n_deepstack_layers` | **3** |
| `rope.dimension_sections` | **4-element array** |
| `rope.freq_base` / `context_length` | 5000000.0 / 262144 |

Projector — `general.architecture = clip`, 316 tensors:

| key | value |
| --- | --- |
| `clip.projector_type` | **`qwen3vl_merger`** |
| `clip.vision.block_count` / `embedding_length` / `feed_forward_length` | 24 / 1024 / 4096 |
| `clip.vision.attention.head_count` | 16 |
| `clip.vision.patch_size` / `image_size` | 16 / 768 |
| `clip.vision.projection_dim` / `spatial_merge_size` | 2048 / 2 |
| `clip.vision.is_deepstack_layers` | 24-element array |
| `clip.use_gelu` | true |

**What is new:**

1. **`rope.dimension_sections` is mRoPE** — rope split into sections across a 4-element layout, not
   the single-axis rope every current decoder uses. Getting this wrong produces plausible text that
   degrades with image position, which is the worst failure mode: silent.
2. **A ViT tower** — patch embed at 16px, 24 blocks, GELU MLP, layer norm. Structurally familiar,
   entirely new code.
3. **The `qwen3vl_merger` projector with `spatial_merge_size = 2`** — 2×2 patch merge before
   projection to 2048.
4. **Deepstack** — `n_deepstack_layers = 3` on the text side and a 24-element
   `is_deepstack_layers` on the vision side: several vision layers feed the text tower, not just
   the last one.
5. **Image preprocessing** — resize/pad to the model's expectations, `image_mean`/`image_std`
   normalization from the header.
6. **A second gate again.** Image understanding needs its own metric; OCR needs page accuracy, and
   `correctAnswerRate` describes neither.

OCR rides the same runtime. `ggml-org/GLM-OCR-GGUF` ships `mmproj-GLM-OCR-Q8_0.gguf`, and
`datalab-to/surya-ocr-2-gguf` exists — so **once a vision tower and a projector work, OCR is a
catalog entry and a metric rather than another runtime project.** That is the argument for doing
vision before OCR, and for doing it on the smallest Apache-2.0 VL available.

## Honest ordering

1. **Fix the `gemma_3n_e2b` capability claims.** Minutes, and it is currently wrong in public.
2. **Whisper tiny.** One new modality, a self-contained frontend, cross-attention that nothing else
   needs yet, 38M to iterate on, Apache-2.0.
3. **Qwen3-VL 2B.** Two new modalities at once (image, and video is the same tower plus frame
   sampling), but a new text architecture *and* a new vision tower *and* mRoPE.
4. **OCR on top of 3.** Mostly a metric and catalog work once 3 lands.

Each of 2, 3 and 4 needs a qualification policy that does not exist yet. That is the part most
likely to be underestimated: the runtime is the visible half, and the gate is what makes the entry
mean anything.

## What has not been done

No code has been written for any of this. No throughput, accuracy or WER number appears above
because none has been measured here. The upstream olmOCR-Bench scores in
`model-jars/docs/modeljars-operations-and-model-candidates.md` are upstream self-reported and are
explicitly not ours.
