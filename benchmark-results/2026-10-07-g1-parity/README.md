# G1 passes: 1280 token ids identical, device against CPU

**Measured 2026-10-07 on an RTX 4090 (compute capability 8.9, driver 13020), Runpod.** Models
revision `15661fb2836967e200322e5b0ceaa91416790994`. Kernel sha256 `4c534e622452df95...`, PTX target
`sm_80`. Model: Granite 4.1 3B Q4_K_M, sha256 recomputed by the gate from the file it opened.

```
PASS cuda-kernel-gate mode=parity device=NVIDIA GeForce RTX 4090 prompts=20 tokens=1280 identical
```

| gate | result |
| --- | --- |
| `tokenParity` (G1) | **passed** -- 1280 token ids identical across 20 prompts |
| `routingObservability` (G2) | passed -- 2,405,576 accelerated operations |
| `startupHonesty` (G5) | passed -- 344 ms readiness against a 120,000 ms ceiling |
| `qualified` | **true** |

`parity.firstDivergence` is **null**: every token id sequence matched. `parity.selfTest` is
**false**, which is the field that makes this G1 evidence at all -- off-device both arms are the
Vector API and the command says `SELF-TEST` instead of `PASS`. One sequence hit end-of-generation,
which is recorded and deliberately not obeyed, so the compared length is not shortened where the
arms agree.

`routing.totalDeclinedProjections` is **0**, from the counter added the same day. Nothing fell back
to the Java path, so the 1280 identical tokens were produced with the projections actually on the
device rather than by a quiet fallback that would have made parity trivial.

## What changed since G1 last failed

A note from a 2026-09-26 A40 run recorded G1 failing on an argmax flip at token 7, attributed to
`models_gqa_decode_attention`'s `expf`. Between then and now `expf` was rewritten on a different
principle, stated in `backend-cuda/src/main/rust/models-cuda-kernels/src/attention.rs`:

> *"The requirement is not that it be accurate -- it is that it be **the same function the CPU
> runs**."*

It transcribes the CPU's clamp, magic-constant rounding, Cody-Waite split, Taylor coefficients and
single-step exponent construction, in that order, with `fma` exactly where the CPU has one. That is
what exact token parity needed, and accuracy alone would not have given it.

Two differences from the failing run are recorded rather than waved at: this is an RTX 4090 at
compute capability 8.9 where that was an A40 at 8.6, and this revision routes the FFN gate and up
projections to the device where that one left them on the CPU. The same `sm_80` module serves both
capabilities, but a single host does not establish hardware independence. **Re-run on an A40 or
L40S before claiming G1 holds across the qualifying profiles.**

## Both gates now pass

- **G1**, here: 1280/1280 token ids identical.
- **G4**, `benchmark-results/2026-10-07-g4-dualpath`: 6.184x decode against a 3.00x gate, 31.26
  against 5.06 tok/s, with a control arm that moved 0.2%.

Numerically exact and 6.18x, on one host, on one model.

## What this does not establish

One model and one device. Granite 4.1 3B is 40 blocks of `granite` architecture with Q4_K and Q6_K
projections; it exercises neither a mixture-of-experts routed FFN nor a per-layer feed-forward width
nor an architecture whose attention scale differs. G1 is a no-tolerance gate, so each qualifying
hardware profile and each architecture family earns its own run.
