# Headers the earlier intake left unread — 2026-10-31

Five artifacts, range-fetched with the repo's own inspector
(`projects/model-jars-central-gate/tools/gguf-inspector.mjs`). No weights downloaded; the probes are
read-only HTTP range requests.

Each one was probed because an open issue plans against an artifact whose header had **never been
read** — either sized only, or probed in a different quantisation. Three of the five produced
findings that change the issue they belong to.

| file | what it probes | for |
| --- | --- | --- |
| `probe.mjs` / `headers.txt` | `parakeet` 110M and 0.6b, `inflect-v2` | #284, #286 |
| `probe-kokoro.mjs` / `kokoro-keys.txt` | every `kokoro` metadata key | #285, #286 |
| `probe-decoders.mjs` / `decoder-headers.txt` | the `mellum` and `lfm2moe` quants the port issues name | #281, #283 |

Reproduce: `node probe.mjs`, `node probe-kokoro.mjs`, `node probe-decoders.mjs`.

## 1. `inflect-v2` does publish its configuration — the prefix is not the architecture string

**Correction.** `docs/QXOTIC-PARITY-FEASIBILITY.md` §2 and §6, and #286, all state that the
`inflect-v2` artifact publishes "**no `inflect-v2.*` metadata keys at all**", so its configuration is
implicit and would have to come from the reference implementation. That is wrong.

It publishes **27 non-tokenizer keys**. They are prefixed **`inflect.`** and **`inflect.v2.`**, while
`general.architecture` is **`inflect-v2`**. The earlier intake searched for `<arch>.*`, found no
`inflect-v2.*`, and concluded there was nothing. The keys were there under a prefix that does not
match the architecture string.

What they contain is not marginal:

```
inflect.v2.symbol_count                       = 178
inflect.v2.symbol_hash                        = 3e81afee...391ea92f7
inflect.v2.sample_rate                        = 24000
inflect.v2.add_blank                          = true
inflect.v2.inter_channels / hidden_channels   = 128 / 72
inflect.v2.filter_channels / n_heads / n_layers = 384 / 2 / 3
inflect.v2.folded_weight_norm_pairs           = 108
inflect.v2.parameter_count                    = 3958801
inflect.v2.reference_deployed_parameter_count = 3966721
inflect.source.revision / checkpoint_sha256 / config_sha256
```

So the premise of #286's "no keys at all" claim is wrong, and that correction stands because it is a
**presence** finding (§0).

The ordering argument it implies does **not** stand. #286 ranks `inflect-v2` last of the three Tier
D ports partly on this intake defect and calls it a "worse intake problem" than `kokoro`'s, and the
first pass of this document tried to invert that. Both readings are unsound: **`kokoro` carries its
symbol table too**, enumerated in `tokenizer.ggml.tokens` rather than pinned by a hash, which is the
stronger form — and neither artifact reconciles against `kv_count`, so neither can be the subject of
an absence claim in either direction. The ordering should be decided on the decoder work and the
companion-artifact shape, not on intake quality.

Two further things worth having: `folded_weight_norm_pairs = 108` says weight norm is already folded
into the published tensors, so a port must not fold it again; and the 7,920-parameter gap between
`parameter_count` and `reference_deployed_parameter_count` is declared rather than left to be
discovered.

## 2. Both TTS artifacts declare 178 symbols

`kokoro.n_token = 178` and `inflect.v2.symbol_count = 178`. That is a strong indication the two
share a phoneme inventory, which bears directly on #285: **one G2P would serve both ports**, so the
cost of building one is amortised over two models rather than charged to each.

It is an indication and not a proof — the counts match and `inflect-v2` pins its inventory by hash
while `kokoro` does not enumerate its at all, so the two cannot be compared symbol by symbol from the
headers alone.

## 3. `parakeet`'s 110M sibling is a real `parakeet`, and is *not* a scaled-down 0.6b

#284 says the 110M file was "sized but never probed. Its architecture is assumed from its filename,
which is exactly the assumption the header intake exists to avoid." Probed:
`general.architecture = parakeet`, `general.name = nvidia/parakeet-tdt_ctc-110m`,
`parakeet.arch = hybrid_tdt_ctc`, 690 tensors, 131,387,520 bytes. **The filename assumption holds.**

But it is not the same model smaller, and four of the differences change the port:

| | 110M | 0.6b-v3 |
| --- | --- | --- |
| encoder `d_model` / `n_layers` | **512 / 17** | 1024 / 24 |
| `preprocessor.n_mels` / `encoder.feat_in` | **80** | 128 |
| `vocab_size` / `blank_id` | **1024 / 1024** | 8192 / 8192 |
| `decoder.pred_rnn_layers` | **1** | 2 |
| tensor roots | `ctc_decoder` decoder encoder joint preprocessor | decoder encoder joint preprocessor |

The last row is the one to notice. **Only the 110M ships a `ctc_decoder` tensor root**, although both
declare `arch = hybrid_tdt_ctc`. So the declared architecture string does not tell you which heads
are present, and a loader that trusts it would look for a CTC branch the 0.6b does not have, or
silently ignore one the 110M does.

It also means #284's "**2-layer LSTM** predictor" is a property of the 0.6b only; the 110M has one
layer. A port written to the 0.6b's shape and tested on the 110M would be wrong in the predictor.

**This makes the 110M the better first target** — 7x smaller, 17 encoder layers against 24, half the
model width, a 1024-entry vocabulary and a single-layer predictor — provided the extra CTC head is
loaded or deliberately skipped rather than ignored by accident.

## What this does not establish

- **No port, no model run, nothing measured.** These are headers. Every number above is a declared
  metadata value, not an observed one.
- **Nothing about `parakeet`'s quality or ours.** #284 requires the ASR direction to be decided on
  **measured WER against a published test set**, against `docs/MODALITY-EXPANSION.md`'s Whisper work.
  Nothing here speaks to that, and a smaller artifact being cheaper to port is not an argument that
  it is the right model.
- **The 178-symbol match is not a shared-inventory proof**, as stated in §2.
- **Neither TTS artifact reconciles** (§0), so nothing here supports a claim that either one lacks a
  given key. `kokoro`'s symbol table is present; what else either file carries is unknown.
