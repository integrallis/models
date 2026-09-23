# Capability probe — does the qualified Granite base support architecture A?

**This is a probe, not a measurement. No number here should be quoted as performance.** It exists
so the architecture was known to be viable before any harvest was paid for.

Run 2026-09-20 on Hetzner `decisions-bench-noul-20260920` (ccx33, 8 vCPU AMD EPYC-Milan, AVX2,
30 GiB, Ubuntu 24.04.4), Temurin 25.0.4.1, `:models-decisions:integrationTest`.

## Artifact provenance

| | |
| --- | --- |
| Model | `granite-4.1-3b-Q4_K_M.gguf` |
| Source | `huggingface.co/ibm-granite/granite-4.1-3b-GGUF` at pinned revision `ab47014810…` |
| Size | 2,099,501,664 bytes — matches the ModelJars catalog exactly |
| SHA-256 | `662b0626cd58f443baea23559b469df6576a81d349649c59413b36a9fb32eb29` — **verified equal** to the catalog's recorded `baseArtifactSha256` |
| Licence | Apache-2.0 |

## GGUF metadata, read from the artifact

```text
general.architecture                    granite      (pure transformer; no ssm.* keys, not a Mamba hybrid)
granite.block_count                     40
granite.embedding_length                2560
granite.attention.head_count            40
granite.attention.head_count_kv         8            (GQA)
granite.context_length                  131072
granite.vocab_size                      100352
```

`granite` routes to `DecoderArchitecture.GRANITE` and therefore to `LlamaDecoder`, which declares
`supportsHiddenState()` and implements `freezePrefix`.

## What the probe established, on the real weights

- `PureJavaBackend` loads the artifact and **is** a `SharedPrefixInferenceBackend`.
- `supportsHiddenState()` is true. `supportsSharedPrefixes()` is true.
- Two forks of one frozen prefix return **true** from `sharesPrefixStorage` — physical sharing
  proven by the backend, not inferred from timing.
- The hidden state is **2560 wide** and entirely finite. That is the head's input width.
- Two identical question suffixes over one shared prefix produce **exactly equal** hidden states.
- `SharedInferencePrefix.sharedBytes()` is positive.
- `SharedPrefixDecisionEvaluator` drives the real backend end to end and returns a verdict inside
  its declared answer space, having frozen the prefix exactly once.

**Conclusion: architecture A is viable on this base.** A decision head can read a 2560-wide hidden
state off a physically shared prefix without a decode loop.

## What this probe does not establish

Nothing about quality, agreement with the decode verdict, calibration, or speed. The qualified
answerability evidence is on the **`rust-ffm`** backend; this probe used pure Java, so its figures
would not be comparable to that evidence even if it had produced any. A comparable measurement needs
`backend-native` built on the host.

## A trap worth recording

The first two attempts reported `SKIPPED` and then `UP-TO-DATE`: Gradle does not forward `-D` to the
forked test JVM, and does not treat an environment variable as a task input. The probe skipped
visibly rather than passing vacuously, but a cached skip could still have been mistaken for a run.
`integrationTest` now declares `DECISIONS_GRANITE_MODEL` as an input so changing it invalidates the
task.

## rust-ffm toolchain on the host (build only, no measurement)

The qualified answerability evidence is on `rust-ffm`, so cycle 3 needs `backend-native` present for
its figures to be comparable with that evidence.

- Ubuntu 24.04's packaged Cargo is too old for this repository's Rust 2024 crate, as the runbook
  records. Installed the rustup bootstrap; the repository's pin selected **rustc 1.96.0**.
- **`:backend-native:assemble` does not build the native library.** It completed in 10 s and
  produced only the Java jars. The cdylib is built by `prepareNativePlatformResources`, which
  compiled `jmodels-kernels v0.3.42` and emitted `libjmodels_kernels.so` (458,656 bytes) into
  `META-INF/models/native/linux-x86_64/`.
- `:backend-native:test` passes, including the bundled-library poll-budget case, so the FFM binding
  resolves and loads the library actually built on this host.

**Why this is written down:** had `assemble` been trusted, a cycle-3 arm labelled `rust-ffm` would
have run with no native library present. Depending on fallback behaviour that either fails outright
or quietly executes the pure-Java path under the wrong label, which is the ablation hazard the
working agreement names — a stage that never ran, reported as a stage that did not matter.
