# Models Rust FFM backend

[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/models/main/backend-native/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

`backend-native` accelerates selected GGUF compute kernels without embedding
llama.cpp, Ollama, or another inference engine. Model parsing, tokenizer behavior,
the transformer graph, KV-cache ownership, sampling, and generation remain in Java.
The native boundary is a versioned C ABI implemented by the Models-owned
`jmodels-kernels` Rust crate.

ABI 2 supports Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, and Q6_K batched projections and
grouped dispatch. Mixed Q4_K/Q5_K/Q6_K groups share one Q8_K activation
quantization. Its x86-64 path uses format-specialized AVX2/FMA integer dots,
vectorized Q8_0 activation preparation, batched weight reuse, reusable activation
scratch, and an explicitly owned persistent worker context. Scalar kernels
remain available as cross-platform conformance fallbacks.

## Build and test

```shell
./gradlew :backend-native:check
./gradlew :backend-native:nativePlatformJar
./gradlew :backend-native:integrationTest
./gradlew :backend-native:slowTest \
  --tests com.integrallis.models.backend.nativekernel.RustFfmQ5KLargeModelTest
```

Gradle builds the host library with Cargo under
`backend-native/build/rust-target`. Native Java tests run with
`--enable-native-access=ALL-UNNAMED`. The real-model integration test requires
`~/.jvllm/models/Qwen3-0.6B-Q4_0.gguf`,
`~/.jvllm/models/smollm2-360m-instruct-q8_0.gguf`, and
`~/.jvllm/models/MiniCPM5-1B-Q4_K_M.gguf`. The Q5_K slow test requires
`~/.jvllm/models/sqlcoder-7b-q5_k_m.gguf`.

## Load

When the matching platform artifact is present on the runtime classpath,
`RustFfmBackend` verifies its ABI/platform metadata and SHA-256, extracts it to
`~/.models/native-kernels/abi-2/<platform>/<sha256>/`, and opens it through Java
25 FFM:

```java
Path gguf = Path.of("/opt/models/private-model.gguf");

try (var backend = RustFfmBackend.load(gguf)) {
  // Use the same Models backend API as the pure-Java implementation.
}
```

The content-addressed extraction directory and library are owner-only on POSIX
filesystems. Set `models.native.kernels.cache` to override the cache root.

For a local development build, the library path can still be supplied
explicitly:

```shell
java \
  --enable-native-access=ALL-UNNAMED \
  -Dmodels.native.kernels.library=/path/to/libjmodels_kernels.so \
  -jar application.jar
```

`models.native.kernels.threads` controls the worker-context size and defaults to
the JVM-reported processor count.

Q5_0 grouped projection dispatch is not enabled by default. On the controlled
Qwen2.5-0.5B x86-64 profile, fused grouping recovered worker-barrier overhead but
still decoded at 37.33 tokens/second versus 38.94 tokens/second for independent
Q5_0 projections. `-Dmodels.native.q5_0.grouped=true` is therefore retained for
controlled qualification runs and is not selected by release profiles.

## CI artifacts and private packages

`.github/workflows/native-kernels.yml` compiles and opens the packaged library on
Linux x86-64/AArch64, macOS x86-64/AArch64, and Windows x86-64/AArch64. Pull
requests retain the raw library, platform JAR, slim Java JAR, and standalone JAR
as workflow artifacts.

Successful `main` builds publish immutable CI versions to the repository's
GitHub Packages Maven registry:

```text
com.integrallis:backend-native-standalone:<ci-version>
com.integrallis:models-kernels-linux-x86_64:<ci-version>
com.integrallis:models-kernels-linux-aarch64:<ci-version>
com.integrallis:models-kernels-macos-x86_64:<ci-version>
com.integrallis:models-kernels-macos-aarch64:<ci-version>
com.integrallis:models-kernels-windows-x86_64:<ci-version>
com.integrallis:models-kernels-windows-aarch64:<ci-version>
```

The standalone coordinate is the private-test artifact: it contains the Java
backend, its runtime dependencies, and all six native resources with no Maven
dependencies. The platform artifacts allow isolated binary testing. These CI
packages are not release-qualified or a substitute for signed Maven Central
artifacts.

The backend is not release-qualified merely because a kernel exists. A
model/platform profile must preserve its correctness oracle and meet the
project's controlled performance floor before ModelJars selects it.
