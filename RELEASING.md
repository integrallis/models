# Releasing models

The release path mirrors `integrallis/mfcqi-java`: Gradle stages publications,
JReleaser signs and validates one Maven Central bundle, and the workflow creates
the GitHub release.

The publication allowlist contains `models-api`, `models-runtime`, `models`,
`models-rag`, `models-semantic-order`, `backend-java`, `backend-native`,
`backend-apple`, `models-langchain4j`, `models-spring-ai`, and
`models-embedding`. Benchmark applications, documentation tooling, and modules
containing only package scaffolding are not published.

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

The workflow uses the same Maven Central and GPG secrets as `mfcqi-java`:
`MAVENCENTRAL_USERNAME`, `MAVENCENTRAL_PASSWORD`, `GPG_PUBLIC_KEY`,
`GPG_SECRET_KEY`, and `GPG_PASSPHRASE`.
