# Gated-fusion study — results log (append-only)

One row per executed phase. The rules are those of `benchmark-results/2026-09-17-logit-fusion-study/RESULTS-LOG.md`: append only, every row carries a provenance label (`measured` / `read` / `believed`), evidence must exist in the host evidence directory beside this file and be listed in its `SHA256SUMS`, a gate failure is recorded and is never a reason to adjust a result.

## S1 (2026-11-25, box `fusion-s1`, Hetzner cpx62, Models `05cbefc6`, pure-Java backend, context 4096, T = 0)

| date | commit | host | phase | gate / arm | evidence (`evidence/s1-cpx62-2026-11-25/`) | outcome | provenance |
|------|--------|------|-------|------------|---------------------------------------------|---------|------------|
| 2026-11-25T20:44Z | `05cbefc6` | fusion-s1 (cpx62) | data | — | `data-prepare.log` | done (gsm8k 1319, arc 1172, math500 500 from the September pins) | measured |
| 2026-11-25T20:45Z | `05cbefc6` | fusion-s1 (cpx62) | models | — | `gguf-sha256.txt`, `convert-math.log` | pinned: L, I official Q4_K_M; M converted Q8_0 + Q4_K_M; sha256 in `models.json` | measured |
| 2026-11-25T20:46:29 | `05cbefc6` | fusion-s1 (cpx62) | gate | g0 (attempt 1) | `gate-g0.attempt1.json` | FAIL — M vs L differ in tokenizer.ggml.add_eos_token only (absent in the official files, false in the conversion); member re-pinned without the key, gate unchanged | measured |
| 2026-11-25T21:46:44 | `05cbefc6` | fusion-s1 (cpx62) | gate | g0 | `gate-g0.json` | PASS G0 members=3 mismatches=0 report=/root/evidence/gate-g0.json (wall=4s) | measured |
| 2026-11-25T21:46:44 | `05cbefc6` | fusion-s1 (cpx62) | gate | g4-gsm8k | `gate-g4-gsm8k.json` | PASS G4 gsm8k 1319/1319 report=/root/evidence/gate-g4-gsm8k.json (wall=0s) | measured |
| 2026-11-25T21:46:45 | `05cbefc6` | fusion-s1 (cpx62) | gate | g4-math500 | `gate-g4-math500.json` | PASS G4 math500 500/500 report=/root/evidence/gate-g4-math500.json (wall=1s) | measured |
| 2026-11-25T22:11:12 | `05cbefc6` | fusion-s1 (cpx62) | gate | g8 | `gate-g8.json` | PASS G8 comparisons=240 worst=0.0 report=/root/evidence/gate-g8.json (wall=1467s) | measured |
| 2026-11-25T22:56:01 | `05cbefc6` | fusion-s1 (cpx62) | gate | g1-gated | `gate-g1-gated.json` | PASS G1-gated comparisons=40 report=/root/evidence/gate-g1-gated.json (wall=2689s) | measured |

What S1 establishes, and nothing more: the three Q4_K_M members share one tokenizer byte for byte (G0, after the conversion artifact was removed); the scorers are exact on gold (G4); on real weights an auxiliary's catch-up by one prefill is bit-identical to stepping (G8: 240 comparisons over 20 items x 3 members x d in {1,2,7,31}, worst max |Δlogit| 0.0); and the gated backend is the lead alone when the gate never fires and is the every-token reference when it always fires, token for token on 20 GSM8K items at up to 256 tokens (G1-gated, 40 comparisons). No accuracy, no speed and no survival claim follows from S1. S2 is the pilot.
