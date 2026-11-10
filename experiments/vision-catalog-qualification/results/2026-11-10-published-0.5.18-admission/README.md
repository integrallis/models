# Eight-gate admission attempt at the 0.5.18 release commit

The baseline the F16 KV store measurement in `../2026-11-10-f16-kv-store/` is compared against,
banked here because it existed only on an ephemeral instance.

Provenance clarification: the worker built the JARs and native library from the release commit.
This directory's historical name does not establish that the loaded bytes came from Maven Central.
The subsequent 0.5.19 admission must separately record downloaded Central runtime JARs and the
native library extracted from the published JAR.

Runtime `168b7df4bd8eb5bc09e80833569c24ffae4f194d` on `m6a.4xlarge` (AMD EPYC 7R13, 8 cores / 16
threads, AVX2, no AVX-512), 8 threads, context 1024, 32 output tokens, external oracle
`2cbca389f515df455412dafa07e2b1359cc72a6d`.

## Verdict: not admissible, on one quantity

| gate | Q4_K_S | Q4_K_M |
| --- | --- | --- |
| accuracyLoss | pass | pass |
| vqaFloor | pass | pass |
| textVqaFloor | pass | pass |
| imageContribution | pass | pass |
| tokenAgreement | pass | pass |
| performance | pass | pass |
| strongestReferencePerformance | pass | **fail** |
| integrationConfirmation | **fail** | **fail** |

Every failing number is our decode measured against the reference's **F16 KV** arm, and every one
of them sits within a few percent of the 1.10 ceiling:

- `integrationConfirmation`, both variants: **1.10080**, which is 0.0727% over. The initial
  schedule measured 1.09546 on the same box, a 0.49% spread across warmup schedules — 6.7x the
  margin by which it failed.
- `strongestReferencePerformance`, Q4_K_M: `decodeMs` of **1.10279** on `vqav2-132415014`, with its
  other three registered cases at 1.09170, 1.06958 and 1.06584.
- Q4_K_S passed the same gate at 1.09183, 1.09848, 1.07329 and 1.06421 — margins of 0.2% and less.

Nothing here is an accuracy or agreement failure. Both variants reproduce the reference's answers
exactly (`tokenAgreement = 1`) and score identically to it on both datasets.

## Reproduce

    python3 ../../finalize.py . <initial-integration> <confirmation-integration>

`admission-attempt.json` is that output. `receipt.json` records the host, revision, corpus and
artifact digests and every payload jar's SHA-256. The `.f32` embedding blobs and `.png` probe
images are excluded, matching the convention of the other committed result directories.
