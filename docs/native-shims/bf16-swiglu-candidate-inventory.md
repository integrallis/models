# BF16 and SwiGLU candidate evidence inventory

Both operations remain **UNPROVEN**. This inventory accompanies the external composed Models patch `a43b47ca8ba326b456456fd008f416b2d740e39ee25e0da873c03c269ffce918` over `218daaea7b3a52e97311b56e5d950245ed917343`, with actual Gradle Vectors JAR `9fc500f992f5b5ef73086691d53d82a8ea8702e5d80d6145d582365b3ef7c189`. It is not a release or a completed native-necessity certificate.

## Current external composition delta

The paragraphs below describe historical `a43` evidence and retain its original scope. The current ARM SLEEF composition changes capability-27 ARM128 SwiGLU to the pinned permissively licensed SLEEF recipe, including full-domain exceptional handling and consistent scalar activation tails. It also adds sigmoid/scaled-softplus and causal convolution hooks. The ordinary Qwen2/Qwen3 Llama path now consults the SwiGLU provider; Granite retains its separate arithmetic. x86 keeps the earlier finite-domain native algorithm. Caller-owned prepared views remove repeated heap-segment wrappers without caching session arrays in the provider. Each new route remains UNPROVEN until the final artifact passes its own model-specific quality and complete resource/performance gates; historical component results do not transfer automatically.

## BF16 fixed-tree projection

Capability bit 25 and `jmodels_bf16_f32_batched_matmul` implement the measured 48 × 5120 matrix shape, batches 1–32, selected Java 4/8-lane tree and fused/unfused policy. Qwen35's individual and paired projections consult the provider; other shapes remain Java. AVX2+FMA or AArch64 NEON is required. Output overlap is rejected before writes. Native load/FFM cost is part of the actual-model comparison, not removed by pre-initialization.

Java allocation/compiler diagnostics identified interpreted Vector API wrappers and context-dependent escaped vector objects in actual Bonsai BF16 calls. An exact native component control reduced allocation and latency in four matched Linux pairs on each architecture. That establishes a measured mechanism and candidate improvement, not proof that every better Java form is impossible or that all supported hosts meet release gates. The first actual SPI fixture reported zero BF16 calls, exposing an unhandled paired Java route. The corrected graph route has a retained failing counterfactual and passing graph test. The final local Mac/JDK25.0.3 Bonsai fixture then observed 5,760 BF16 calls and 11,312 SwiGLU calls; all seven owned full-logit records matched between instrumented and production binaries. This verifies local dispatch and numerical correspondence, not Linux performance or resources. Supported-host dispatch, task quality and the complete resource/performance protocol remain required.

## Stable F32 SwiGLU

Capability bit 26 and `jmodels_swiglu_f32` implement the fixed Arm-derived float polynomial with explicit fused arithmetic. The checked ABI accepts positive lengths divisible by 4, finite gate magnitudes at most 80, and no forward output overlap. Unsupported inputs leave buffers untouched and retain Java fallback. Equal-offset in-place and backward output overlap are supported. Qwen35 uses the optional hook; other Models consumers of VectorUtil use the same new Java F32 fallback.

Actual compiled SPI numerical proof covers 580,911,106 signed float inputs in the declared nontrivial domain, special/boundary cases and owned cold/hot snapshots. Maximum SiLU absolute error was 1.9073486328125e-6, maximum 4 ULP, with no failures against the original support-test tolerance. This is not an unconditional 2e-6 bound after arbitrary up multiplication or a task-quality guarantee. Up amplification and special values have separate checks. Both native and Java fallback calls were observed. The final Gradle core passed 1,624 tests and the composed backendJava suite passed 1,004 tests, all without skips.

Actual Bonsai profiles attributed substantial additional cold allocation to the exact double-vector Java activation. Several Java and native alternatives failed retained performance/resource checks. The stable F32 Java candidate improves warm latency but still exceeds the published cold-allocation baseline. Native F32's first x86 form also failed latency; replacing unnecessary masked full-block loads/stores produced a four-pair Linux component improvement. Those screens are descriptive and do not satisfy all whole-model, precision or host-profile gates. The new polynomial changes floating results from both published and earlier exact-fdlibm implementations, so quality and deterministic whole-model checks must use the final composition.

## Reachability and retirement

No newly introduced helper or export is unreachable in the reviewed composition. The Rust architecture bodies serve detected hardware, selected lane/FMA policies, and actual block tails. Java's retained fdlibm code is used for supported forward-overlapping public buffers; it is not a competing normal-input route. The separate diagnostic counter exports exist only in an external instrumentation patch and must never enter a timed or production binary.

Retire either native operation only when its best Java replacement passes unchanged numerical, actual-model, latency, allocation, RSS, startup, confidence and supported-host requirements with executed-route evidence. Remove that operation's capability, export, FFM binding and architecture bodies together; retain independent behavior and tests. Worker/context infrastructure is shared and is not justified as an independent shim.

## Durable evidence location

Raw experiments remain outside project repositories under workspace `audit-artifacts/`:

- `combined-spi-final-v2/manifest.json`, `dispatch-instrumentation.json`, `numerical-proof/`, `full-java-fixture-results/` and `mobilemoe-integration/` record final source, binaries, numerical and fixture scope.
- `bonsai-native-spi-harness-v2/local/final-preflight/dispatch-correspondence.json` (SHA256 `21e6b3bfe746e5c4fa7fe934672b6b98a2843137b582cbe704a3fb0be0f9329e`) records the local actual-Bonsai dispatch and exact instrumented/production correspondence.
- `vectors-f32-final/manifest.json` and `VALIDATION.md` record actual Gradle source/JAR correspondence and the retained transition from the unreleased exact-StrictMath oracle.
- `bf16-native-fixed-tree/` and run 38355245649 retain BF16 component evidence; `bf16-native-model-v1/` retains the earlier BF16-only caller comparison.
- `swiglu-arm-float-native/linux-38356381088/` retains the failed original native form; `swiglu-arm-float-native/unmasked-v2/linux-38356974787/` retains the corrected component comparison.

Each result remains bound to its own artifact. Earlier component or prototype successes are not relabeled as proof of the final composed SPI. No completed proof JSON is supplied because the campaign is still incomplete.
