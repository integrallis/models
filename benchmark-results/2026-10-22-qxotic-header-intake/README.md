# qxotic header intake — raw artifacts

2026-10-22. Stage 0 of `docs/QXOTIC-PARITY.md`, executed. Every architectural number in
`docs/QXOTIC-PARITY-FEASIBILITY.md` comes from these files; nothing in that document is estimated.

| file | what it is |
| --- | --- |
| `qxotic-models.txt` | copied verbatim from `references/qxotic` at `6f08d93ec`, `jinfer/scripts/models.txt` — **their** test manifest, which is where every artifact below came from. Their own fixtures, so no artifact here was chosen by us to flatter a result. |
| `probe.mjs` | the probe. Calls `projects/model-jars-central-gate/tools/gguf-inspector.mjs`, the repo's existing range-fetch inspector; no new GGUF parser was written. |
| `headers.json` | its output: `general.architecture`, `general.name`, tensor count, tensor roots, byte size and every non-tokenizer scalar metadata key, per artifact. |
| `quants.mjs`, `arch-keys-and-dtypes.txt` | per-artifact GGML dtype histogram and the `<arch>.*` metadata keys. The dtype numbers are GGML type ids; `GgufTensorType.java:23-62` maps them. |
| `repo-file-sizes.txt` | every `.gguf` in each repository with its size, from the HuggingFace API. This is what decides whether a quantization small enough to qualify exists at all. |

Reproduce: `node probe.mjs` then `node quants.mjs`. Both are read-only HTTP range requests against
public repositories and download no weights — the largest artifact probed is 33.59 GB and the probe
reads a few hundred KB of it.

Three results are absences, which is why the manifest is committed next to the probe: no artifact in
qxotic's own manifest declares `minicpm` (`openbmb/MiniCPM5-1B-GGUF` reports `llama`), none declares
plain `nemotron_h` (the Nemotron artifact reports `nemotron_h_moe`), and `bailingmoe3` had no
artifact in the manifest at all.

**`bailingmoe3` is no longer an absence (2026-10-31, #288).** A first-party MIT artifact exists
outside qxotic's manifest — `inclusionAI/Ling-3.0-tiny-GGUF`, revision `01b850e29ed5` — and its
header is now the eleventh entry here, read the same way as the other ten. Its
`general.architecture` is `bailingmoe3`, so this is the architecture and not a near neighbour.

The header also settles a reading that `docs/QXOTIC-PARITY.md` §4 took from their codec rather than
from an artifact. MLA plus Q-LoRA is confirmed — `attn_q_a` [1536, 256] with `attn_q_a_norm` and
`attn_q_b` [256, 3072] at `q_lora_rank` 256, and `attn_kv_a_mqa` [1536, 576] with `attn_k_b` and
`attn_v_b` at `kv_lora_rank` 512 — but it is **only half the model**. Those tensors appear on 6 of
24 layers. The other 18 carry `ssm_conv1d_q`, `ssm_conv1d_k`, `ssm_conv1d_v`, `ssm_a`, `ssm_beta`,
`ssm_f_a`, `ssm_g_a` and `ssm_norm`, with `kda.head_dim`, `kda.safe_gate` and
`kda.gate_lower_bound` in the metadata. So it is a hybrid on a 4-layer interval, and a feasibility
estimate taken from the MLA reading alone would have described a third of it.
