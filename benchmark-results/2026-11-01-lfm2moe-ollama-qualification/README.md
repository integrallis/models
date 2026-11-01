# lfm2moe two-arm qualification against an authoritative comparator

**Verdict: `FAILED_RELATIVE_GATE`.** The first admitted comparison for any of the qxotic MoE ports.

Measured 2026-11-01 on one RunPod `cpu3c` pod, 16 vCPU of an AMD EPYC 7713 (64-core, **no
AVX-512**), both arms sequential in one session on that box.

| | candidate (rust-ffm) | comparator (ollama 0.40.13) | ratio | gate |
|---|---:|---:|---:|---|
| p50 decode tok/s | 23.88 | 60.08 | **0.3974** | ≥ 0.80 FAIL |
| p50 prefill tok/s | 25.54 | 139.31 | **0.1833** | not gated |
| end-to-end latency | | | **2.8670** | ≤ 1.50 FAIL |
| load ms | 690 | 9713 | | |

Contribution gate **passes**: `modelAnswerRate` 0.4444 against a 1/3 minimum,
`modelAnswerCorrectRate` 1.0000 against 0.90, over the same nine cases and the same
`corpusSha256` as the comparator. `absoluteTier` is `PERFORMANCE_REDUCED`, not a failure tier. Both
arms carry `artifactSha256 = 48ed1465d761311b2fd57b7fb46cf969a20b3a8281945b04e10f52bd1609e715`, and
the verdict records no exclusions, so the policy compared them rather than discarding one.

## Why this run exists, and what it replaces

Three earlier two-arm runs (maple, mellum, lfm2moe, 2026-10-31) reported throughput ratios that
**no gate had admitted**. The runner passed `--model` but not `--artifact` to the comparator arm, so
its `artifactSha256` was null, and `RagProductionQualificationPolicy.exclusion` drops any comparator
whose sha differs from the candidate's. Running the policy on those artifacts returns
`NO_COMPARABLE_BASELINE` with `exclusions: {"llama.cpp": "artifact SHA-256 differs"}`. The ratios had
been computed by dividing the two reports by hand. They are withdrawn.

Those runs also used `--warmups 0 --iterations 1`, where all 116 comparator arms in the certified
catalog used 1 and 3.

**The off-protocol numbers were flattering, not penalising.** lfm2moe's hand-computed figures were
0.4809 decode and 1.4915 end-to-end; measured properly they are 0.3974 and 2.8670. Warming the
comparator sped the comparator up, so our ratio got worse. The direction was not predictable from
the fault, which is the argument for re-running rather than correcting arithmetic.

## Comparator choice

`AUTHORITATIVE_ENGINES` is `{ollama, transformers}`. A llama.cpp arm is admitted at 0.45/2.0 but is
never a backend that can qualify a model, so it reaches `FAILED_RELATIVE_GATE` at best; the
`--publisher-reference` route that does let it qualify one at 0.80/1.5 is reviewed per artifact and
exists for publishers shipping their own llama.cpp (Bonsai a fork, EdgeRock a patch). Upstream serves
lfm2moe without either, so ollama is the comparator.

Checked before spending the run: ollama serves this architecture, at **49.74 tok/s** decode warm
against the llama.cpp arm's **49.50** on the same box. 0.5% apart, which cross-validates the two
comparators instead of trusting either. The model is created from this exact GGUF, never a registry
copy of the same name, which is why the layer digest equals the candidate's `artifactSha256`.

## What the shortfall is, and what it is not

Prefill is the worse half, 5.5x short against decode's 2.5x, and it drives the end-to-end failure:
per-case ratios run 4.25-16.04. The grouped expert dispatch committed for this issue batches the
experts selected for **one token**; prefill walks tokens one at a time, so it does not benefit. The
standard shape -- group the batch's tokens by expert, one matmul per expert over its own token
subset, which is what ggml's `mul_mat_id` does -- is not implemented here.

Not a silent fallback: all four quantisations report grouped-eligible at batch 1 for these shapes,
lfm2moe's tensors are all Q4_0, and the profile shows the grouped kernel being entered.

`/proc/loadavg` on this pod reports the **256-core host**, not our 16-vCPU cpuset -- `cpu.max`
carries no quota and `/proc/cpuinfo` lists every core while `nproc` reports the 16 we are pinned to.
It read 41 while our own processes were idle. Recorded here, not used as a gate; pairing both arms in
one session on one box is what controls for the neighbour.

## Files

`candidate.json`, `comparator.json` -- the two arms. `verdict.json` -- the policy's output, which is
the only source for a ratio in this directory. `run.log` -- host, protocol and the printed verdict.
`candidate.log`, `comparator.log` -- per-case timings. `create.log`, `warm.json` -- comparator setup.

Reproduce with `scripts/fleet/two-arm-pod.sh`, which defaults to the ollama comparator and prints
the policy's verdict rather than leaving two reports to be divided.
