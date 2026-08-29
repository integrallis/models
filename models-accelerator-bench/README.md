# Java accelerator experiments

This private Gradle module retains accelerator experiments, rejected candidates, and release gates.
The qualified Q4 projection provider now lives in the optional published `backend-tornado` module.
Its kernels are Java source compiled for an accelerator by TornadoVM; there is no external
inference server and no handwritten CUDA or Rust shim.

The first experiment targets quantized matrix projection because profiling and the existing Models
execution seams identify it as the dominant reusable operation. It currently contains:

- a Q8_0-by-F32 projection used to establish device execution, persistent weights, and transfer
  costs;
- a production-compatible Q4_0-by-Q8_0 projection with the same activation quantization and FP16
  scale rounding as `vectors-core`;
- dual and triple Q4_0 dispatches that share one prepared activation across gate/up and Q/K/V
  projections;
- one fixed 32-token device shape that lets prompt chunks reuse the same compiled plans; an eager
  readiness pass can compile those plans before the first user-visible request;
- an experimental `GgufBatchedMatrixKernel` that injects the Q4_0 kernel into an otherwise unchanged
  `PureJavaBackend` for prefill only; and
- a full-model Qwen experiment that compares output and Models-owned generation metrics.

The tests execute the kernels as ordinary Java and compare every output with `vectors-core`. A real
accelerator run is a separate qualification gate because a JVM-equivalent method can still expose a
device-compiler lowering defect.

## Run the tests

```shell
./gradlew :models-accelerator-bench:test
```

## Run on TornadoVM

Install a TornadoVM distribution that matches JDK 25 and the desired backend, then build the
application distribution:

```shell
./gradlew :models-accelerator-bench:installDist
```

Use the TornadoVM launcher with the generated runtime classpath. For example:

```shell
lib=models-accelerator-bench/build/install/models-accelerator-bench/lib
classpath=$(printf '%s:' "$lib"/*.jar)

tornado -cp "$classpath" \
  com.integrallis.models.accelerator.Q4ProjectionExperiment \
  32 3072 1024 5 20
```

The grouped-projection correctness gate executes both dispatches on the selected device:

```shell
tornado -cp "$classpath" \
  com.integrallis.models.accelerator.Q4GroupedProjectionExperiment
```

The full-model experiment takes a local Qwen3 0.6B Q4_0 GGUF path:

```shell
tornado -cp "$classpath" \
  com.integrallis.models.accelerator.QwenFullModelExperiment \
  /path/to/Qwen3-0.6B-Q4_0.gguf
```

Add `--eager` to compile the fixed-shape projection plans during an explicit readiness phase before
measuring the first visible request:

```shell
tornado -cp "$classpath" \
  --params="/path/to/Qwen3-0.6B-Q4_0.gguf --eager" \
  com.integrallis.models.accelerator.QwenFullModelExperiment
```

Add `--decode` with `--eager` to create separate single-token plans and run an exact eight-token
CPU/GPU decode-parity gate. The experiment reports median and total decode time and fails if the
greedy token sequence changes:

```shell
tornado -cp "$classpath" \
  --params="/path/to/Qwen3-0.6B-Q4_0.gguf --eager --decode" \
  com.integrallis.models.accelerator.QwenFullModelExperiment
```

`--attention` is retained only as a reproducible negative experiment. Its kernels pass the isolated
numeric gate, but the full-model result on the A16-2Q is slower than the Vector API attention path;
it is therefore not a candidate for automatic dispatch.

Pass application flags through TornadoVM's `--params` option. For example, the 250-token case is:

```shell
tornado -cp "$classpath" \
  --params="/path/to/Qwen3-0.6B-Q4_0.gguf --long-prompt" \
  com.integrallis.models.accelerator.QwenFullModelExperiment
```

Use ordinary `java` with the generated classpath and `--cpu-only` when collecting a Vector API
control on a machine without a TornadoVM runtime.

Results are qualification evidence, not universal performance claims. See
the [A16-2Q](results/vultr-a16-2q-2026-08-29.md) and
[A40-4Q](results/vultr-a40-4q-2026-08-29.md) reports.
