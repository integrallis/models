# LLaVA 1.5 performance closure — measured 2026-11-07

All three encoder probes and both paired decoder comparisons pass the frozen 1.10 maximum median-latency ratio and unchanged correctness gates. This closes the measured integration performance gap; it is not catalog admission or a VQA/OCR task score.

Candidate runtime commit: `0f2bb7d6a4b30c6b5297d86456b9320388d86707`. Host: `ip-172-31-11-163`; 8 threads in each arm; 3 warmups and 7 measured runs. Models uses the original pinned GGUF weights, F32 KV, context 1024, and its production image-prefill default. Reference settings and binary/library hashes are embedded in the decoder reports.

| Probe | Models median ms | PyTorch median ms | Ratio | Cosine | Normalized RMSE | Pixel max error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cats | 885.911 | 862.667 | 1.0269 | 0.999999987775 | 0.00015645 | 0.00000000 |
| wide-gradient | 854.124 | 887.706 | 0.9622 | 0.999999998656 | 0.00005184 | 0.00000000 |
| tall-bands | 842.448 | 912.710 | 0.9230 | 0.999999999828 | 0.00001859 | 0.00000000 |

Reference prefill below includes its separately measured PyTorch encoding/preprocessing plus llama.cpp decoder prefill. Models prefill measures the complete image/text call. Decode emits 32 greedy tokens (31 forward calls); every token matches the reference in every repetition.

| Reference KV | Models prefill ms | Reference image + prefill ms | Prefill ratio | Models decode ms | Reference decode ms | Decode ratio | Total ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f32 | 12319.254 | 11591.270 | 1.0628 | 3958.658 | 4430.320 | 0.8935 | 1.0166 |
| f16 | 12319.254 | 11522.470 | 1.0692 | 3958.658 | 3635.850 | 1.0888 | 1.0743 |

The F32 comparison matches cache precision; F16 also tests the faster stock reference default. Encoder vectors are checked independently against Transformers/PyTorch on the same converted weights. llama.cpp is an external comparator, never linked or bundled with Models.

The historical pre-optimization run and its distinct worker/protocol remain in `../2026-11-07/`; no historical timing is used as the denominator of the paired closure gate.

Qualification still requires frozen published VQA/OCR workloads, accuracy scoring, full-workload memory/cold-load measurements, and two-artifact catalog packaging. The three probes and one bounded caption establish integration fidelity and performance only.

## Evidence and verification

The full raw archive is retained at `java-ai/audit-artifacts/pr380-vision-performance/pr380-performance-evidence.tar.gz` in the shared workspace. `archive-sha256.txt` verifies the archive; `raw-vector-sha256.json` identifies the exact vectors. The archive contains final and rejected-trial sources, payloads, vectors, logs, profiles and reference binaries, without model weights or credentials. All verdicts were recomputed successfully from the downloaded archive before terminating the worker.

[Rejected experiments](EXPERIMENTS.md) · [Worker lifecycle](../../PERFORMANCE-CONTROL.md) · [Reproduction guide](../../../../docs/VISION-QUALIFICATION.md)

Local `models-api:check`, `backend-java:check` and `backend-native:check` passed, including formatting, static analysis, coverage and dependency policy. Cached real-model fixtures were excluded on the development Mac. The retained test reports count:

| Module | Passed | Skipped | Failures/errors |
| --- | ---: | ---: | ---: |
| models-api | 94 | 0 | 0 |
| backend-java | 1272 | 11 | 0 |
| backend-native | 120 | 0 | 0 |

Regenerate both tables/notes from committed reports:

```sh
python3 experiments/vision-qualification/summarize_performance.py experiments/vision-qualification/results/2026-11-07-performance
python3 experiments/vision-qualification/summarize_exploration.py experiments/vision-qualification/results/2026-11-07-performance
```
