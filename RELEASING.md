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

## When a native platform job fails, decide before you touch anything

A failed native platform job stops the release. The bundle is assembled only after **every**
native succeeds, so this cannot produce a partial publish — observed on 0.3.56, run
`38008419296`: `Native windows-x86_64` failed, `Release to Maven Central and GitHub` was
**skipped**, every artifact returned 404, and no tag was created. That is worth knowing as
behaviour and not merely as intent, because it means a failed run is safe to diagnose calmly
rather than urgently.

The question is only whether the failure is the code or the infrastructure. Read the root cause
before re-dispatching, and check these four things:

1. **Where in the job did it die?** `BUILD FAILED in 34s` is not a compilation failure. Grep the
   log for `What went wrong` rather than reading the Gradle stack trace, which is always the
   bottom of the log and never the cause.
2. **Did the other platforms pass?** Five of six passing on the same commit rules out most code
   explanations immediately.
3. **Did the dry run pass on effectively the same tree?** Compare the two runs' shas with
   `git diff --name-only <dry> <real>` and look for anything on the path that failed. If the delta
   is docs and scripts, the code did not change.
4. **Is the thing it could not find actually there?** Fetch it yourself.

Worked example, 0.3.56. The cause was
`Plugin [id: 'com.integrallis.mfcqi', version: '0.7.0'] was not found`, at 34 seconds, during
plugin resolution before any compilation. The plugin's POM returned **200** from
`repo1.maven.org` when checked by hand; five other platforms resolved the same plugin in the same
run; the dry run had passed the identical job fifteen minutes earlier; and the only commit delta
was `scripts/fleet/**` and `docs/**`, nothing touching Rust. Four signals, all pointing at
transient resolution on one runner. Re-dispatched unchanged as run `38009139958` and
`Native windows-x86_64` passed.

**Re-dispatch is the right action only once that case is made.** Re-running a release because a
job is red, without knowing why it was red, is how a real defect gets published on the second
attempt. Equally, do not "fix" the workflow after a single flake: adding retries or a repository
mirror to the release path on one data point is speculative surgery on the one pipeline that must
be trustworthy. Wait for a second occurrence and a reason.
