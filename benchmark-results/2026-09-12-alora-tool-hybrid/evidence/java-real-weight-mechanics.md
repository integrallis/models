# Java real-weight mechanics gate

Date: 2026-09-12

Status: **passed**, implementation evidence only. The adapter used by this test failed the separate
tool-quality gates and remains rejected.

## Artifacts

- Base GGUF: Qwen3 0.6B Q4_0, 428,970,080 bytes,
  SHA-256 `da2572f16c06133561ce56accaa822216f2391ef4d37fba427801cd6736417d4`.
- Activated-LoRA: `qwen3-06b-tool-alora-r32-v1`, 80,792,456 bytes,
  SHA-256 `dec7257a3aaedc7e20b94f996482821d512d957737d9d64bc713802671f7c310`.
- Adapter provenance pins `Qwen/Qwen3-0.6B` revision
  `c1899de289a04d12100db370d81485cdf75e47ca` and invocation tokens
  `[151644, 77091, 198]`.

## Host

- Intel Core i7-9750H, macOS/Darwin x86_64
- Temurin 25.0.3+9 LTS, mixed mode

## Command

```bash
./gradlew :backend-java:activatedLoraCompatibilityTest \
  -Dmodels.fixtures.activatedLoraDirectory="$ADAPTER_DIRECTORY" \
  -Dmodels.fixtures.activatedLoraOracle="$PROJECTION_ORACLE_DIRECTORY" \
  --no-daemon
```

`ADAPTER_DIRECTORY` is deliberately not committed because this first adapter is a rejected,
experimental oracle rather than a distributable artifact.

## Assertions exercised with real weights

1. The public `PureJavaBackend.loadActivatedAdapter(...)` path validates the exact base and adapter
   hashes, metadata, tensor names, dtypes, and shapes and loads the adapter in-process.
2. The tool prompt is evaluated once up to the pinned activation boundary.
3. The activated tool branch and exact-base response branch report that they reference the same
   physical immutable KV-prefix storage.
4. Both branches report the full shared prefix as cache-read input tokens.
5. The response generated from the shared base branch is exactly equal to ordinary base-model
   generation for the same complete tool-result prompt.
6. All 196 real adapter projections (seven matrices in every one of 28 layers) match independently
   generated NumPy `B @ (A @ input) * alpha/rank` deltas with maximum absolute error no greater
   than `2e-4` and cosine at least `0.999999`.
7. A conversation can extend the same physical prefix across consecutive tool-selection turns.
8. After a completed response, canonical ChatML rerendering may replace only the generation-control
   suffix. The runtime rewinds and recomputes that mutable suffix while retaining the same immutable
   KV blocks; a rerender that crosses the immutable prefix is rejected.

Result: all four `ActivatedLoraModelIntegrationTest` methods passed; Gradle build successful in 8
minutes 22 seconds with the exhaustive oracle enabled.
This does not assert that the rejected adapter chooses the correct tool.
