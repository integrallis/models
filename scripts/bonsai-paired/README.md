# Combined BF16 and activation correction: whole-model paired validation

This protocol is frozen before execution in `protocol.json`. It compares the same actual
Models commit and native library with released Vectors core against the exact combined BF16 and all-activation
correction. Only the core JAR differs; every file and runtime origin is recorded. This is
pre-publication evidence, never permission to publish.

The copied `Bf16ModelProof.java` removes only the obsolete hardware-layout description from its runtime-origin metadata and preserves the existing Qwen Transformers logits/tolerance,
Bonsai oracle continuation, tokenizer, native-route, repeated-session, implicit-reset and
whole-vocabulary batch-parity assertions. A candidate fixture failure blocks timing for that model only. Bonsai and Qwen run in separate jobs, so the independently known Qwen BF16 failure cannot prevent Bonsai timing; neither failure is hidden. A
released-baseline failure is retained without blocking comparative measurements. The root
workflow separately runs the unchanged repository JUnit fixtures.

Each model and architecture has ten fresh counterbalanced process pairs. Fixed shards 0–4
and 5–9 keep both arms on the same host. Each process retains its cold request, three warmups,
and three measured requests with immediate owned full-logit snapshots. Thirty distinct
measured prompts alternate the first fifteen frozen JEVBench and SNIPS cases; this selection
is fixed by the existing harness, not based on observed results. Both public datasets' pinned
source revisions and licenses are retained. Their full 148-case corpus is included as source
material; this timing protocol does not claim to execute all 148 cases.

The analysis uses geometric mean paired ratios and host-stratified 20,000-resample percentile
bootstrap with seed 716309. No observation is discarded. Each metric's upper confidence limit
must meet its frozen limit and the interval width must be at most two percentage points.
Latency limits are 1.0; startup, allocation and RSS limits are 1.01. Cold latency is reported
and gated separately. Throughput for the fixed eight decode tokens is the inverse decode
latency ratio. Inconclusive precision is not success. Baseline/candidate numeric differences
are recorded because the baseline has a known determinism defect; candidate repeatability,
oracle fixtures, and unchanged-or-better classification counts on each dataset are required.

GitHub hosted x86 and ARM runners are the supported hosts tested by this workflow. Their CPU,
JDK, native library, artifact hashes, commands, exit status, and elapsed time are retained.
No compiler flags, warmup-dependent routes, or native-selection properties are added.

The launched candidate is broad revision `6893a587`, whose unchanged Bonsai fixtures passed on both architectures. Bonsai 2 is `qwen35`, verified directly from the SHA-pinned GGUF header; it exercises GDN sigmoid/softplus and causal convolution as well as SwiGLU. Earlier candidate-only SwiGLU and BF16 ablations failed and are not represented as qualified. Any narrower final runtime requires new matching qualification.
