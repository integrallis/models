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

The three published modules have no dependency on `vectors`. Placeholder
modules also declare no dependencies until they contain an implementation, so
models CI and release builds are standalone.

The workflow uses the same Maven Central and GPG secrets as `mfcqi-java`:
`MAVENCENTRAL_USERNAME`, `MAVENCENTRAL_PASSWORD`, `GPG_PUBLIC_KEY`,
`GPG_SECRET_KEY`, and `GPG_PASSPHRASE`.
