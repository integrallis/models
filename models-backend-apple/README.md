# models-backend-apple

Experimental bridge from Java to Apple's on-device Foundation Models runtime.

This module is intentionally separate from `models-backend-purejava`. The core
models runtime remains pure Java; this module uses Java FFM to load a tiny Swift
dynamic library because Apple's `FoundationModels` API is exposed as an Apple
platform framework, not as Java bytecode.

## Shape

The native boundary is four C ABI symbols:

- `jmodels_afm_available`
- `jmodels_afm_generate`
- `jmodels_afm_last_error`
- `jmodels_afm_free`

The Java side loads the dylib with FFM, checks platform support before loading,
and exposes a small prompt/response client:

```java
try (var client = AppleFoundationModels.create()) {
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

Point the Java client at the bridge with either:

```bash
export MODELS_APPLE_FOUNDATION_LIBRARY=/path/to/libjavamodels_apple_foundation.dylib
```

or:

```bash
-Dmodels.apple.foundation.library=/path/to/libjavamodels_apple_foundation.dylib
```

## Native Bridge

Build on an Apple Silicon Mac with the macOS SDK that includes
`FoundationModels`:

```bash
models-backend-apple/src/native/apple-foundation-models/build-bridge.sh
```

The script prints the dylib path. Use that path as
`MODELS_APPLE_FOUNDATION_LIBRARY` or `models.apple.foundation.library`.

The Swift side follows the same foundation used in Apfel:

- check `SystemLanguageModel.default.availability` before generation
- create `LanguageModelSession` with optional instructions
- pass `GenerationOptions(maximumResponseTokens:)`
- call `session.respond(to:options:)`

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

That library exports the same symbols as the Swift bridge, so the Java FFM
loader, pointer ownership, and native error path can be tested without importing
Apple's `FoundationModels` framework.

## Testing

Intel Macs and ordinary CI can run the unit tests:

```bash
./gradlew :models-backend-apple:test
```

The FFM/native ABI smoke test builds the C stub and is part of the module's
integration task:

```bash
./gradlew :models-backend-apple:integrationTest
```

The real Apple Intelligence smoke test is tagged `integration` and skips unless
the current machine is macOS on Apple Silicon, the native bridge path is
configured, and `SystemLanguageModel` reports available:

```bash
export MODELS_APPLE_FOUNDATION_LIBRARY=/path/to/libjavamodels_apple_foundation.dylib
./gradlew :models-backend-apple:integrationTest
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
