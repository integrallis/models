# Models 0.5.19 released-runtime LLaVA admission

Both Q4_K_S and Q4_K_M passed all eight frozen backend admission gates. This run
started on 2026-11-10 at 23:24:35 UTC and finished on 2026-11-11 at 00:59:40 UTC.
The directory uses its start date. No threshold, case, prompt, or gate was changed.

| Variant | Gates passed | Worst task latency ratio | Token agreement | Backend qualified |
| --- | --- | --- | --- | --- |
| Q4_K_S | 8/8 | 0.99962 | 1.000 | true |
| Q4_K_M | 8/8 | 1.00162 | 1.000 | true |

Ratios are candidate/reference latency (lower is better), taking the worse F32/F16
reference for each of the four task performance cases and each timed component.
The ceiling is 1.10. Q4_K_M's previously failing task decode case,
`vqav2-132415014`, now measures 1.0016165147164997.

The shared Q4_K_M/cats integration confirmation also passes: F32-reference decode
ratio 0.8103865995178908 and F16-reference decode ratio 0.9821158370008769.
Both have exact greedy-token agreement. All three encoder/projector probes pass
numerical and performance checks. This shared integration probe is not a separate
Q4_K_S task measurement.

## Scope and provenance

- Registered host class: EC2 m6a.4xlarge, AMD EPYC 7R13, 8 physical cores / 16
  logical CPUs, AVX2, Java 25.0.5. Both reference arms and candidates ran sequentially
  on `ip-172-31-9-66` in the same session, using 8 threads and context 1024.
- Frozen corpus: 32 VQAv2 questions, 32 TextVQA questions, and 16 paired blank-image
  controls per quantization; maximum output 32 tokens. Four performance cases use
  one warmup and three repetitions. This bounded subset is not a publisher benchmark
  reproduction, and does not establish performance on other host classes.
- Runtime release/tag commit: `214d53f928923d21514cf2d8bca6e02712d4dd51`.
  Every Integrallis runtime JAR came from Maven Central, Models 0.5.19 and Vectors
  0.1.34. The unpublished `models-bench` harness alone was built from that exact
  source commit; third-party dependencies came from its resolved distribution.
  `runtime-receipt.json` records the origin and SHA-256 of every JAR.
- The explicit native path selects the Linux x86_64 library extracted from the
  published backend-native JAR, SHA-256
  `9e21989209d90d3d5caa21d4bfe1bcec40b47a858129fb0013914b91d0d08bb1`.
  This backend measurement is not a ModelJars public consumer or bundled-native
  loading proof; those separate catalog publication flags remain false here.
- External reference source: `2cbca389f515df455412dafa07e2b1359cc72a6d`;
  binary/library hashes, complete outputs, controls, and resource usage are retained.

## Reproduce the verdict

Run `python3 derive.py` in this directory. It reruns the unchanged finalizer against
the retained evidence in a temporary directory, requires byte-identical output,
then prints the table above. `admission-attempt.json` is an exact copy of
`campaign/results/admission-attempt.json` for an immutable catalog report URL.

The committed files retain JSON, JSONL, logs, TSVs and provenance. Binary numerical
tensors are preserved in the separately checksummed release asset described in
`raw-evidence.json`. Model weights and source dataset images are not bundled.
The orchestration scripts record runtime staging, sequential collection, supervision,
and archive selection; the benchmark and scorer remain the frozen repository tools.

Backend qualification is complete. Public catalog admission still requires the
ModelJars API/CLI preview, released SDK consumer, published marker resolution, and
ordinary generated JBang demo checks described in the ModelJars protocol.
