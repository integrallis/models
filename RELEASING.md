# Releasing models

The release path mirrors `integrallis/mfcqi-java`: Gradle stages publications,
JReleaser signs and validates one Maven Central bundle, and the workflow creates
the GitHub release.

Version `0.1.x` publishes only `models-api`, `models-runtime`, and
`models-backend-purejava`. Modules containing only planned package scaffolding
are intentionally excluded.

## Cut a release

1. Set a non-snapshot version in `gradle.properties`.
2. Update `CHANGELOG.md`, commit, and push to `main`.
3. Run **Actions → Release** with `dry_run` enabled.
4. After validation succeeds, rerun with `dry_run` disabled.

`models-backend-purejava` depends on `vectors-core` for JDK Vector API numeric
kernels. Local development resolves that dependency through the sibling
composite build; releases consume the published `vectors-core` artifact.
Placeholder modules still declare no dependencies until they contain an
implementation.

The workflow uses the same Maven Central and GPG secrets as `mfcqi-java`:
`MAVENCENTRAL_USERNAME`, `MAVENCENTRAL_PASSWORD`, `GPG_PUBLIC_KEY`,
`GPG_SECRET_KEY`, and `GPG_PASSPHRASE`.
