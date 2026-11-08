# Pre-merge admission findings

Neither LLaVA 1.5 7B quantization qualifies. Do not merge PR380 on the assumption that
the earlier integration caption establishes catalog readiness. The current measured
runtime is `ff627e32a3910488d9d4a9c1d15d361fe3242e25`; thresholds and measured outputs
are unchanged. [Generated verdict](README.md) contains every gate and ratio.

Accuracy, absolute task floors, image contribution and matched-F32 performance pass
for both candidates. Q4_K_S has 63/64 identical token sequences; Q4_K_M has 62/64,
below the frozen 99% requirement. Q4_K_S's task scores equal the reference. Q4_K_M
has higher VQAv2 score on this bounded subset and equal TextVQA score. These are
fidelity failures, not evidence of lower aggregate answer accuracy.

Against the faster F16 reference, short-answer decode ratios reach 1.167027 for
Q4_K_S and 1.137717 for Q4_K_M; the ceiling is 1.10. Prefill and total latency pass.
The long-caption integration repeat also misses the decode ceiling at 1.102956.
Its exact tokens and numerical comparisons pass. Both the initial one/three attempt
and the declared three/seven confirmation remain recorded; neither is relabeled.

## Disagreement localization

The crossed diagnostic uses the same model bytes, native JARs, saved prompt tokens,
F32 KV, eight threads and greedy cap. All original diagonal answers reproduce.
Leading-space differences in reference decoding are omitted in this display only;
the admission gate compares token IDs.

| Quantization / case | Models decoder + Models encoder | Models decoder + reference encoder | Reference decoder + Models encoder | Reference decoder + reference encoder |
| --- | --- | --- | --- | --- |
| Q4_K_S / textvqa-39587 | Aegre | Aegerter | Aegerter | Aegerter |
| Q4_K_M / vqav2-359592004 | No | Yes | Yes | Yes |
| Q4_K_M / textvqa-38337 | Movies | Movies | Electronics | Electronics |

Pixels are exactly equal on all three disputed images. Projected-vector normalized
RMSE is respectively 0.00001696, 0.00005928 and 0.00002657. Two cases depend on the
interaction between encoder perturbations and the Models decoder. The last follows
the decoder with either encoder. This localizes the work; it does not establish a
specific defective operator or justify patching an answer. The human labels favor
"No" in the green-beans case; both runtimes are wrong in the two TextVQA cases.

Raw evidence: [crossed outputs](disagreement-diagnostics/report.json),
[pixel/vector differences](disagreement-diagnostics/input-differences.json), and
per-step Models top-two margins in the adjacent per-case JSON files.

The first diagnostic compilation failed because a macOS archive included AppleDouble
`._*.jar` metadata. That failed log and exit status are preserved. The corrected
runner hashes and selects only executable JARs from the original receipt; the
qualification payload and timings were not changed or rerun to hide this harness
failure. Diagnostic execution then completed successfully.

## Required engineering before another admission attempt

1. Trace the first divergent decoder logits on `textvqa-38337` with identical image
   embeddings. Compare layer outputs, activation quantization and Q4_K/Q6_K
   accumulation with the pinned reference. Preserve a small operator-level regression
   fixture when the first discrepancy is isolated. Do not special-case the question.
2. For the two interaction cases, capture CLIP/projector intermediates and replay the
   same decoder inputs. Determine which numerical change improves agreement across
   the corpus; the existing pixel checks rule out these images' preprocessing.
3. Profile short-answer native decode separately from prefill. Close the measured
   F16-reference gap with a retained production improvement, then repeat both short
   and long decode workloads. A passing matched-F32 ratio alone is insufficient.
4. Freeze a new runtime commit, retain this rejected attempt, and rerun the complete
   two-arm workload. Keep the current dataset, gates and failed examples. Do not
   choose another warmup schedule or enlarge the denominator merely to pass.
5. After backend admission passes, run the ModelJars public API with the actual
   two-file marker, empty cache, default execution settings, both datasets and reset
   repeat, plus the CLI-generated demo. Only after the ordered release train may a
   released-runtime repeat permit public catalog publication.

## Reproduce without inference

From the Models repository, use the exact scorer revision/digest in the manifest:

```sh
python3 experiments/vision-catalog-qualification/reproduce.py \
  experiments/vision-catalog-qualification/results/2026-11-08 /path/to/m4c_evaluator.py
python3 experiments/vision-catalog-qualification/render.py \
  experiments/vision-catalog-qualification/results/2026-11-08/campaign/results/admission-attempt.json
```

The first command reproduced all task scores, paired timing ratios and final admission
decisions from the downloaded observations. It does not rerun numerical tensors.
Full vectors, sources, binaries and diagnostic arrays are retained in the checksummed
workspace archive; model weights and credentials are excluded.
