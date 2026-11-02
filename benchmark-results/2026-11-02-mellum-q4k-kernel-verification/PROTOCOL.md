# mellum, Q4_K per-block unpack: end-to-end verification protocol

Frozen before the numbers were read. Results land in README.md beside the raw arms.

## What is being tested

Whether the Q4_K per-block scale/min unpack fix (`perf/q4k-decode-gap-rebased`, commit 760f7776,
measured at 1.68x on the row dot against current main) moves mellum's `decodeThroughputRatio` past
the 0.80 floor. Contribution and end-to-end already pass; decode is the only failing gate.

## Design

Two complete arm sets on **one box, strictly sequential**, differing only in which
`libjmodels_kernels.so` is loaded via `-Dmodels.native.kernels.library`:

- **arm set 1** `libkernels-main.so`, built from `origin/main` @ `33700300`, label `main-33700300`
- **arm set 2** `libkernels-fixed.so`, built from the rebased branch, label `q4kfix-760f7776`

The two libraries were confirmed distinct by sha256 (`0c6127f1...` fixed, `d9fa85fe...` main) before
launch, because `kernel-implementation` is derived from capability bits and reads identically for
both -- the `backendVersion` label and these digests are what separate the arms in the record.

Each arm set runs the repo's own `scripts/fleet/two-arm-pod.sh`, so the candidate and the comparator
are paired in one session and the verdict comes from `RagQualificationCli` under
`RagProductionQualificationPolicy` -- never from dividing the two reports by hand. Running the whole
script twice means the ollama arm is measured twice, which is deliberate: the earlier record warns
its decode spread on one box reached +/-8%.

## Configuration -- recorded, not guessed

Taken from `benchmark-results/2026-11-01-mellum-final-qualification`, which ran this model:
workload `coding`, template `chatml-no-think`, top-k 1, `--warmups 1 --iterations 3`,
`--threads 16`, context 2048, max-tokens 256, `COMPARATOR=ollama` (the policy's
`AUTHORITATIVE_ENGINES` is `{ollama, transformers}`, so a llama.cpp arm cannot yield QUALIFIED).
`--artifact` is passed, without which `artifactSha256` is null and the policy excludes the
comparison.

Artifact: `Mellum2-12B-A2.5B-Thinking-Q4_K_M.gguf`, sha256
`489cf0d7ca86ef4683e34e2efe8a46a9b52573bfe994a6ad9c86bf57c7173ccb` -- byte-identical to the
artifact the earlier mellum runs used, checked rather than assumed.

## Decision rule

`decodeThroughputRatio >= 0.80` and `endToEndLatencyRatio <= 1.50`, plus the contribution gate
(`modelAnswerRate >= 1/3`, `modelAnswerCorrectRate >= 0.90`). The verdict is whatever the policy
emits for arm set 2. If it fails, that is the result and gets published as one.

## Known limitation of this box, stated up front

The pod is 16 vCPU of an **AMD EPYC 9654 with AVX-512**; the 2026-11-01 mellum ladder was measured
on an **EPYC 7713 without** it. Our Rust kernels gate on `avx2 && fma && f16c` and carry no AVX-512
tier, while llama.cpp/ollama do, so the comparator has an ISA advantage we cannot use here. US-KS-2
had no capacity for matching hardware. This is why arm set 1 re-measures the baseline on this same
box rather than comparing against the 0.6422 taken elsewhere: the **delta** between the arm sets is
controlled, while the absolute ratio is conservative and is not comparable to the earlier ladder.

Host `/proc/loadavg` reports the 192-core host, not our 16-vCPU cpuset, so it is recorded and never
gated on. Pairing both arms in one session on one box is the actual control.
