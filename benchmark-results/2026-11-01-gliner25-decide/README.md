# GLiNER2.5-Decide CPU qualification experiment

Status: **FAILED_ABSOLUTE_QUALITY** for the pinned base checkpoint. No catalog admission.
The first performance screening pair also fails.

The tested development runtime is Models `6add46ee740f8acd80295741fc8e22bae34f845f`,
Vectors 0.1.32, Java 25.0.3, Rust tokenizer 0.22.2, F32 encoder/head. The runtime jars
carry the branch's 0.5.9 development version; they are **not released-runtime evidence**.
`deployment.json` records their individual digests and the source/payload identity.
The publisher is GLiNER2 commit `ca4f68235b50fe4c3ecf3876851d748fe4b15c6c`, with the
hashed dependency lock under `experiments/gliner25-decide/`.

Both arms ran serially on the same m6a.4xlarge Linux x86_64 CPU worker with eight
inference threads. Model weights and real inference stayed on the remote worker.
The original worker was terminated after an S3 upload permission failure, before inference.
Its replacement uses direct object copies without requiring ListBucket permission.
The original two-hour aggregate rented-compute ceiling was retained.

## Correctness and accuracy

| Check | Result |
| --- | --- |
| Initial diagnostic parity, 9 cases | Pass; maximum absolute logit difference 0.0000038316 |
| Shape parity, 13 cases through 512 tokens | Pass; maximum absolute logit difference 0.0000181011 |
| Publisher extract versus classify_text, 46 diagnostic cases | Exact token, logit and selected-label agreement |
| All 529 held-out cases | No errors; identical selected labels in both runtimes |
| JevBench overall | 80.8333%, passes 70% floor |
| JevBench choice | 87.5%, passes 60% floor |
| JevBench yes/no | **61.1111%, fails 70% floor** |
| CLINC banking macro-F1 | 86.6708%, passes 70% floor |
| GoEmotions micro-F1 | **43.6860%, fails 70% floor** |
| GoEmotions macro-F1 | **43.9511%, fails 60% floor** |
| Existing F16 DeBERTa reranker regression | Pass; maximum absolute logit difference 0.0000030994 |

The paired report has zero oracle-relative quality loss on every cohort. Numerical parity
does not turn the shared accuracy failures into passes. The API diagnostic uses all 36
JevBench yes/no cases and the first 10 frozen GoEmotions cases, preserving their original
task mapping and threshold. It is a harness check, not a new qualification cohort.
All 529 inputs fit the declared context limit (maximum 174 encoded tokens); none was dropped
or truncated. Single-label ECE is reported without a calibrated-confidence claim.

Raw outcomes and stderr are in `results-restart-1/`. Reproduce the paired accuracy report:

```sh
python3 experiments/gliner25-decide/assess_quality.py \
  --cases experiments/gliner25-decide/quality-cases.jsonl \
  --results benchmark-results/2026-11-01-gliner25-decide/results-restart-1/quality-1-models.jsonl \
  --reference benchmark-results/2026-11-01-gliner25-decide/results-restart-1/quality-1-reference.jsonl \
  --output benchmark-results/2026-11-01-gliner25-decide/paired-quality-report.json
```

Exit code 1 is the expected failing qualification result. Policy floors and cohort selection
were frozen before either runtime produced quality results. No threshold fitting, test-set
tuning, model replacement, or scope reduction has been used to change this verdict.

## Performance screen and cleanup

All thirteen registered buckets completed 3 warmups and 30 timed requests per runtime.
Only the 32-token/eight-label bucket passes both performance thresholds. The 32-token/two-label
bucket reaches 0.778x reference throughput, below 0.80x. At 128–512 tokens, throughput is
0.313–0.365x and p95 latency is 2.717–3.532x the reference. At 512 tokens, median Models latency
is approximately 4.03 seconds versus 1.34 seconds for the publisher. This is one screening pair,
not a final replication sequence. `performance-1-report.json` retains every bucket and both
arms' latency, throughput and peak RSS. The shared encoder currently projects each input row
separately; batched dense matrix kernels are the next performance experiment, not a measured fix.

Both instances are verified terminated and no volumes from their launch window remain.
`deployment.json` records a conservative aggregate rented-time upper bound of 2,602 seconds
(43 minutes 22 seconds), leaving at least 4,598 seconds within the original two-hour cap.

## Remaining work

The user authorized a separately named accuracy-improved variant for qualification and general
use. Training/development/fresh qualification splits and bounded LoRA tooling are prepared;
no training has run and no accuracy improvement is claimed. The AWS account reports zero G- and
P-family GPU quota; locating the existing RunPod configuration is pending. ModelJars integration, a released
runtime repeat, native packaging checks across advertised platforms, and catalog publication
remain outstanding. A different or fine-tuned checkpoint would need a distinct identity and
fresh held-out qualification; this base checkpoint remains a candidate.
