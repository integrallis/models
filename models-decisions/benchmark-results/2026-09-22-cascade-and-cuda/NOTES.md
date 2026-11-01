# Cascade (cheap proposer + gated verifier) — REJECTED, with the condition that would revive it

Measured 2026-09-22. No new inference: this joins per-item outputs already produced on the same
231 JevBench items (48 easy + 72 original + 111 hard, `datasets/public/*.jsonl`, gold from
`expected`) on the same Hetzner ccx33 host. Reproduce with the scripts in this directory.

Threshold is always fitted on a deterministic stratified half (md5 parity over split+type+id,
fit n=125) and reported on the other half (eval n=106). Binomial SE at these accuracies is
+/- 0.046, and no result below survives it.

## Arms (whole 231, measured)

| arm | acc | p50 | p95 |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B Q4_K_M, letter logits | 0.4069 | 0.621 s | 20.746 s |
| Qwen3-0.6B Q4_K_M, candidate scoring | 0.4935 | 0.665 s | 26.744 s |
| Granite 4.1 3B Q4_K_M, candidate scoring | 0.6580 | 4.274 s | 77.182 s |

## Result 1 — two-model cascade (0.6B proposes, Granite verifies): rejected

Three gates tried: proposer `p_max`, top1-top2 `margin`, and a two-scoring agreement gate.
Every one of them, at its fit-half-selected threshold, is **worse than the verifier alone on
both axes** on the held-out half:

| gate | tau | held-out acc | vs verifier | defer | mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| p_max | 0.970 | 0.6038 | −0.0660 | 0.660 | 1.27x slower |
| margin | 0.950 | 0.6226 | −0.0472 | 0.698 | 1.27x slower |
| agree | 0.980 | 0.6321 | −0.0377 | 0.783 | 1.29x slower |

The latency column is arithmetic, not tuning: the proposer costs 6.081 s mean and is paid on
100% of items; the verifier costs 18.394 s mean. Deferring everything gives 24.475 s
= 18.394 + 6.081 exactly. Break-even needs deferral < 67%, and at that deferral accuracy is
already 0.58–0.60.

## Result 2 — the free-proposer upper bound: still rejected

Setting proposer cost to zero (the best any engineering could achieve) and sweeping:

| tau | acc | vs verifier | defer | mean speedup |
| ---: | ---: | ---: | ---: | ---: |
| 0.300 | 0.5660 | −0.1038 | 0.349 | 1.68x |
| 0.500 | 0.5849 | −0.0849 | 0.425 | 1.38x |
| 0.900 | 0.6038 | −0.0660 | 0.594 | 1.08x |
| 0.990 | 0.6604 | −0.0094 | 0.906 | 1.01x |

There is no knee. Accuracy loss and speedup move together, which is the signature of a gate
whose score carries little information about correctness. The risk-coverage curve confirms it:
the 0.6B scoring arm is 0.87 @ 10% coverage, 0.78 @ 20%, 0.70 @ 30%, 0.63 @ 40%, 0.60 @ 50%.
It only exceeds Granite's *overall* 0.658 out to about 30% coverage. **For a cascade to pay,
the proposer's selective accuracy must exceed the verifier's overall accuracy out to
60–70% coverage.** Ours reaches 30%.

Separately: p95 never improves even with a free proposer, because the gate defers exactly the
long-prompt items. A confidence cascade cannot move the JevBench p95 term by construction.

## Result 3 — same-model shared-prefix cascade: rejected, and this one is informative

Letter-logits (one forward pass) proposes, candidate scoring (k passes over the frozen prefix)
verifies, same 0.6B, so the proposer is free by construction. Best held-out point is 1.25x mean
latency at −0.066 accuracy; the no-loss threshold defers 95% and gives 1.01x.

The reason is worth keeping: **candidate scoring costs only 1.25x the single pass in mean
latency** (6.081 s vs 4.872 s) because prefill dominates and the k candidate forwards reuse the
frozen prefix. The shared-KV-prefix architecture has already removed the cost a cascade was
going to save. There is nothing left to cascade away.

## What this does not test

A stronger proposer. Every number here uses a 0.6B at 70% of the verifier's accuracy. The
condition stated in Result 2 is a real, measurable gate on any future proposer, and it is the
thing to check first rather than building the routing machinery again.

## Provenance of the idea

Read, not measured: MiMo-V2-Flash (arXiv 2601.02780) reports MTP-as-draft giving 3.6 acceptance
length and 2.6x decoding speedup. That is a decode-step technique; System One emits no tokens,
so only the draft-and-gate *principle* was portable, and it does not survive our latency ratio.
Xiaomi's "MiMo-V2.6-Pro-UltraSpeed, up to 20x faster output" claim has no published mechanism on
the product page, the release notes, or the AINews writeup, and third-party listings say 10x
rather than 20x. Treated as unverified.
