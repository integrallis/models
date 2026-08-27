# backend-apple

[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/models/main/models-backend-apple/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

Java FFM bridge to Apple's on-device Foundation Models runtime.

```kotlin
implementation("com.integrallis:backend-apple:0.3.11")
```

This module is intentionally separate from `backend-java`. The core
models runtime remains pure Java; this module uses Java FFM to load a tiny Swift
dynamic library because Apple's `FoundationModels` API is exposed as an Apple
platform framework, not as Java bytecode.

## Shape

The native boundary is three C ABI symbols:

- `jmodels_afm_available`
- `jmodels_afm_generate`
- `jmodels_afm_result_free`

Availability, generated text, and errors use the same owned result structure.
It carries an exact UTF-8 byte length, so the Java FFM layer never scans an
unbounded native C string. Each call owns its result and error detail; there is
no shared native error buffer.

The Java side loads the dylib with FFM, checks platform support before loading,
and exposes a prompt/response client that also implements Models'
`TextGenerationModel` contract:

```java
try (var client = AppleFoundationModels.create()) {
    if (!client.availability().available()) {
        throw new IllegalStateException(client.availability().reason());
    }

    var response = client.generate(
        AppleFoundationModelsRequest.builder("Summarize this in one sentence.")
            .instructions("Be concise.")
            .maxOutputTokens(64)
            .build());
    System.out.println(response.text());
}
```

Run Java with native access enabled:

```bash
java --enable-native-access=ALL-UNNAMED ...
```

Published `backend-apple` JARs include the macOS AArch64 bridge. Models verifies
its SHA-256, extracts it to a content-addressed local cache, and loads it
automatically. No separate bridge installation is required.

An explicit bridge path is available for source builds and bridge development:

```bash
bridge=/path/to/libjavamodels_apple_foundation.dylib
export MODELS_APPLE_FOUNDATION_LIBRARY="$bridge"
export MODELS_APPLE_FOUNDATION_LIBRARY_SHA256="$(
  shasum -a 256 "$bridge" | cut -d ' ' -f 1
)"
```

or:

```bash
-Dmodels.apple.foundation.library=/path/to/libjavamodels_apple_foundation.dylib \
-Dmodels.apple.foundation.library.sha256=<64-hex-digit-sha256>
```

Models rejects an explicit library whose digest is absent or does not match.
For a locally rebuilt bridge only, verification can be bypassed explicitly
with `-Dmodels.apple.foundation.library.allow-unverified=true` or
`MODELS_APPLE_FOUNDATION_LIBRARY_ALLOW_UNVERIFIED=true`.

## Native Bridge

Build on an Apple Silicon Mac with the macOS SDK that includes
`FoundationModels`:

```bash
models-backend-apple/src/native/apple-foundation-models/build-bridge.sh
```

The script prints the dylib path. Configure both that path and its SHA-256 as
shown above.

The Swift side follows the same foundation used in Apfel:

- check `SystemLanguageModel.default.availability` before generation
- create `LanguageModelSession` with optional instructions
- pass `GenerationOptions(maximumResponseTokens:)`
- call `session.respond(to:options:)`

## Framework Adapters

The client can be passed directly to the existing framework adapters:

```java
try (var client = AppleFoundationModels.create()) {
    var langChain4j = new ModelsChatModel(client);
    var springAi = new ModelsSpringAiChatModel(client);
}
```

See the
[Apple Foundation Models guide](https://integrallis.github.io/models/docs/models/current/apple-foundation-models.html)
for complete dependencies and examples.

## Stub Mode

Until real Apple Intelligence hardware is available, use the deterministic stub
mode to exercise the Java availability and generation paths on any development
machine:

```bash
java -Dmodels.apple.foundation.mode=stub ...
```

or:

```bash
export MODELS_APPLE_FOUNDATION_MODE=stub
```

The stub reports `available=true` and returns deterministic text such as
`hello`, `Stub summary: ...`, or `Stub response: ...`. It is not a model and it
does not claim quality or compatibility with FoundationModels outputs. Its job
is to keep application wiring, error handling, and smoke tests moving until a
real Apple Silicon Mac with Apple Intelligence can run the true integration
test.

There is also a native C ABI stub for the FFM boundary:

```bash
models-backend-apple/src/native/apple-foundation-models/build-stub-bridge.sh
```

That library exports the same ABI as the Swift bridge, so the Java FFM loader,
exact-length payload handling, concurrent error isolation, result ownership,
and native error path can be tested without importing Apple's
`FoundationModels` framework.

## Testing

Intel Macs and ordinary CI can run the unit tests:

```bash
./gradlew :backend-apple:test
```

The FFM/native ABI smoke test builds the C stub and is part of the module's
integration task:

```bash
./gradlew :backend-apple:integrationTest
```

The real Apple Intelligence smoke test is tagged `integration` and skips unless
the current machine is macOS on Apple Silicon, the native bridge path is
configured, and `SystemLanguageModel` reports available:

```bash
bridge=/path/to/libjavamodels_apple_foundation.dylib
export MODELS_APPLE_FOUNDATION_LIBRARY="$bridge"
export MODELS_APPLE_FOUNDATION_LIBRARY_SHA256="$(
  shasum -a 256 "$bridge" | cut -d ' ' -f 1
)"
./gradlew :backend-apple:integrationTest
```

Testing Apple Intelligence requires a real Apple Silicon Mac with a logged-in
user session and Apple Intelligence enabled. A generic VPS or VM is not a good
target. Practical options are:

- local M-series Mac mini, MacBook, Studio, or similar hardware
- a self-hosted CI runner on dedicated Apple Silicon hardware
- a bare-metal hosted Mac provider, after confirming Apple Intelligence can be
  enabled in the hosted user session

GitHub-hosted macOS arm64 runners are virtualized and should only be treated as
compile/unit-test infrastructure for this module.
