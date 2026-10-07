# Releasing models

The release path mirrors `integrallis/mfcqi-java`: Gradle stages publications,
JReleaser signs and validates one Maven Central bundle, and the workflow creates
the GitHub release.

The publication allowlist contains `models-api`, `models-runtime`, `models`,
`models-rag`, `models-semantic-order`, `backend-java`, `backend-native`,
`backend-cuda`, `backend-apple`, `models-langchain4j`, `models-spring-ai`,
`models-spring-boot-starter`, `models-embedding`, `models-audio`, `models-router`, and `models-decisions`. Benchmark
applications, documentation tooling, and modules containing only package scaffolding are not
published.

`backend-cuda` publishes an **opt-in** artifact and nothing activates it by accident. There is no
`META-INF/services` entry, so no `ServiceLoader` discovers it: a consumer has to call
`CudaGgufBatchedMatrixKernel.open()` and inject the kernel, and `-Dmodels.cuda.disabled=true` is a
kill switch on top of that. One `sm_80` PTX module serves every device of compute capability 8.0 or
above, carried in the jar under `META-INF/models/cuda/` with a SHA-256 the Java loader recomputes.

**Its numeric state must be stated in the release notes, not assumed from its presence.** G4 (decode
speed) passes at 6.184x against a 3.00x gate, measured in
`benchmark-results/2026-10-07-g4-dualpath`. G1 (exact token parity against the CPU path) is a
separate gate with no tolerance, and a release note must say which way it last ran and on what
hardware. Shipping the jar does not mean the device path is numerically verified; it means a caller
can opt into it and read the gates.

## Cut a release

1. Set a non-snapshot version in `gradle.properties`.
2. Update `CHANGELOG.md` and open a release-preparation pull request.
3. Merge only after the normal CI and integration workflows pass.
4. Run **Actions → Release** with `dry_run` enabled.
5. After validation succeeds, rerun with `dry_run` disabled.

`backend-java` depends on the released `vectors-core` artifact for JDK Vector
API numeric kernels. The release workflow builds and tests the Models-owned
Rust kernels on every supported native platform and compiles the Apple
Foundation Models bridge on macOS before staging the signed Maven artifacts.
Models ships **one** GPU implementation, `backend-cuda`. The TornadoVM arm was removed in 0.3.53:
measured on the same RTX 4090 and the same model, `backend-cuda` reached exact token parity and
31.26 tok/s decode where TornadoVM managed 6.49 tok/s over 36 failed `cuLaunchKernel` calls, and it
required a separately installed runtime that the Maven artifact could not carry. The release
precondition that pointed at `models-accelerator-bench/results/` went with it; `backend-cuda`'s
gates are `cuda-kernel-gate --mode capability|parity|decode` and need no external runtime.

The workflow uses the same Maven Central and GPG secrets as `mfcqi-java`:
`MAVENCENTRAL_USERNAME`, `MAVENCENTRAL_PASSWORD`, `GPG_PUBLIC_KEY`,
`GPG_SECRET_KEY`, and `GPG_PASSPHRASE`.
