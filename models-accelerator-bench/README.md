# Java accelerator experiments

This private Gradle module tests accelerator ideas without placing an unproven backend in a
published Models artifact. The kernels are Java source compiled for an accelerator by TornadoVM;
there is no external inference server and no handwritten CUDA or Rust shim.

The first experiment targets quantized matrix projection because profiling and the existing Models
execution seams identify it as the dominant reusable operation. It currently contains:

- a Q8_0-by-F32 projection used to establish device execution, persistent weights, and transfer
  costs;
- a production-compatible Q4_0-by-Q8_0 projection with the same activation quantization and FP16
  scale rounding as `vectors-core`;
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

Use the TornadoVM argument file with the generated runtime classpath. For example:

```shell
lib=models-accelerator-bench/build/install/models-accelerator-bench/lib
classpath=$(printf '%s:' "$lib"/*.jar)

java @"$TORNADOVM_HOME/tornado-argfile" \
  --add-modules=jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -cp "$classpath" \
  com.integrallis.models.accelerator.Q4ProjectionExperiment \
  32 3072 1024 5 20
```

The full-model experiment takes a local Qwen3 0.6B Q4_0 GGUF path:

```shell
java @"$TORNADOVM_HOME/tornado-argfile" \
  --add-modules=jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -cp "$classpath" \
  com.integrallis.models.accelerator.QwenFullModelExperiment \
  /path/to/Qwen3-0.6B-Q4_0.gguf
```

Add `--long-prompt` for the 250-token prefill case. Add `--cpu-only` when collecting a Vector API
control on a machine without a TornadoVM runtime.

Results are evidence for the next design, not published performance claims. See
[`results/vultr-a16-2q-2026-08-29.md`](results/vultr-a16-2q-2026-08-29.md).
