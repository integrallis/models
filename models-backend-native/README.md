# Models Rust FFM backend

`models-backend-native` accelerates selected GGUF compute kernels without embedding
llama.cpp, Ollama, or another inference engine. Model parsing, tokenizer behavior,
the transformer graph, KV-cache ownership, sampling, and generation remain in Java.
The native boundary is a versioned C ABI implemented by the Models-owned
`jmodels-kernels` Rust crate.

The first ABI supports exact Q4_0 batched and grouped projections. Its x86-64 path
uses AVX2/FMA packed-byte dots, vectorized Q8_0 activation preparation, batched
weight reuse, and an explicitly owned persistent worker context. Scalar exports
remain available as conformance fallbacks.

## Build and test

```shell
./gradlew :models-backend-native:check
./gradlew :models-backend-native:nativePlatformJar
./gradlew :models-backend-native:integrationTest
```

Gradle builds the host library with Cargo under
`models-backend-native/build/rust-target`. Native Java tests run with
`--enable-native-access=ALL-UNNAMED`. The real-model integration test requires
`~/.jvllm/models/Qwen3-0.6B-Q4_0.gguf`.

## Load

When the matching platform artifact is present on the runtime classpath,
`RustFfmBackend` verifies its ABI/platform metadata and SHA-256, extracts it to
`~/.models/native-kernels/abi-1/<platform>/<sha256>/`, and opens it through Java
25 FFM:

```java
try (var backend = RustFfmBackend.load(modelPath)) {
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

## CI artifacts and private packages

`.github/workflows/native-kernels.yml` compiles and opens the packaged library on
Linux x86-64/AArch64, macOS x86-64/AArch64, and Windows x86-64/AArch64. Pull
requests retain the raw library, platform JAR, slim Java JAR, and standalone JAR
as workflow artifacts.

Successful `main` builds publish immutable CI versions to the repository's
GitHub Packages Maven registry:

```text
com.integrallis:models-backend-native-standalone:<ci-version>
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
