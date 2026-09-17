# Component clean-host run — Granite 4.1 3B answerability aLoRA

What this directory establishes, and how to re-run it on a fresh host for either kernel runtime.

**Established here (measured on a clean host):** the Models 0.3.42 artifacts resolve from Maven
Central onto a machine with no `~/.m2`, `~/.jbang` or `~/.gradle`, and the pinned base GGUF plus the
pinned adapter bundle open and produce, on the selected backend, outputs byte-identical to that
backend's committed qualification evidence, physically sharing the KV prefix on all six cases.

**Not established here:** anything about accuracy, latency or the crossover. Six cases is an
identity check on a fresh machine, not a benchmark. The qualification numbers come from the
window arms in `../evidence/`.

## 0. Which backend, and why it matters

The component `modeljars_granite_4_1_3b_answerability_alora_integrallis_f32` is republished as
`1.0.0-f32.2` binding **rust-ffm** evidence, because the base
`ibm_granite_granite_4_1_3b_gguf_q4_k_m` is qualified on rust-ffm only — a pure-java-bound component
can never be opened with it. The clean-host proof therefore has to exist on rust-ffm. The pure-java
arm is kept because it is the arm the identity gate pairs against, and because it is the run the
already-committed `evidence/` directory records.

| Backend | Selection | Expected outputs taken from | Evidence directory |
| --- | --- | --- | --- |
| `pure-java` (default) | `PureJavaBackend.loadActivatedAdapter` | `../evidence/pj1/window-squad-v2-dev-specialist-pure-java.json`, `../evidence/pj3/window-msmarco-v2.1-validation-specialist-pure-java.json` | `evidence/` |
| `rust-ffm` | `RustFfmBackend.loadActivatedAdapter`, reached reflectively exactly as `ActivatedAnswerabilityQualificationCli` (models-bench, v0.3.42) reaches it | `../evidence/main/window-squad-v2-dev-specialist-rust-ffm.json`, `../evidence/main/window-msmarco-v2.1-validation-specialist-rust-ffm.json` | `evidence-rust-ffm/` |

The two expected tables are held separately in `AnswerabilityCleanHost.java` even though they
currently agree entry-for-entry. That agreement is the *measured* result the identity gate asserts,
not an assumption this run is entitled to make: if a future kernel diverged, the Rust arm must fail
rather than be scored against pure-Java expectations.

Only the base matrix products move to the native kernel. The transformer, the adapter delta, the
activation boundary and the physically shared KV prefix stay in Java, which is exactly why the two
arms are expected to emit the same tokens.

## 1. Preconditions

| Input | Value |
| --- | --- |
| Base GGUF | `granite-4.1-3b-Q4_K_M.gguf`, sha256 `662b0626cd58f443baea23559b469df6576a81d349649c59413b36a9fb32eb29`, pinned HF revision `ab4701481089b58a082ef63cc1cee738887293ff` |
| Adapter bundle | ModelJars release `granite-4.1-3b-answerability-alora-integrallis-v1` (4 files, each sha256-pinned in the program) |
| Frozen window v2 | sha256 `dfb8cd7203418c79bae27306ffc4857f0ab02be7f9f8ddcae793af556e9cab37`, pinned raw GitHub at `cb2f42624d6cb75e8caa9fbe1715233ea7148eb2` |
| Models artifacts | `com.integrallis:backend-java:0.3.42`, `com.integrallis:models-runtime:0.3.42`, and for rust-ffm `com.integrallis:backend-native:0.3.42` |

Central coordinates the classpath resolves to (verified 2026-09-17, all group `com.integrallis`
except the two third-party jars):

```
backend-java-0.3.42.jar     531326e385b2c39700f04859719e3b9b7f2a842edb61c28afb25f15995f8a979
models-runtime-0.3.42.jar   b520e8fdb643709493cd7b9b3024d12a3b195e64e54347c087070feb940cc652
models-api-0.3.42.jar       49d3416ceddde6a1d9da117ff88856eed8805c76625d6d6992eadf2d4e4e6ae7
vectors-core-0.1.22.jar     984f3897c511a52a821b3eb27f78d1747651f99b13ca7d9dc471ea7340304376   com.integrallis
jackson-core-2.21.4.jar     4b40a06396f239f8de2da57419adde6e94e5edc18a2171d471ea05eeed4e5c2d   com.fasterxml.jackson.core
slf4j-api-2.0.17.jar        7b751d952061954d5abfed7181c1f645d336091b679891591d63329c622eb832   org.slf4j
backend-native-0.3.42.jar   3e62b2e59df079428f3b90802b48674fcac22dc256617ab5e90156a4b16c95d3   rust-ffm only
```

`vectors-core` is published under **`com.integrallis`**, not a `com.integrallis.vectors` group; it
arrives as a runtime dependency of `backend-java`'s (and `backend-native`'s) POM, so nothing has to
be pinned by hand.

`backend-native:0.3.42` carries the native kernels as classpath resources for all six platforms
(`linux-x86_64`, `linux-aarch64`, `macos-x86_64`, `macos-aarch64`, `windows-x86_64`,
`windows-aarch64`) under `META-INF/models/native/<platform>/`. There is no per-platform artifact to
select and no extra coordinate to add.

## 2. The fresh-host commands

Fresh Ubuntu 24.04 (x86_64 or aarch64), run as root, with `AnswerabilityCleanHost.java`,
`write_clean_host_run.py` and `run-clean-host.sh` in one directory. Nothing else may have populated
`~/.m2`, `~/.jbang` or `~/.gradle`, or `freshMachine` is recorded false and the run cannot pass.

**rust-ffm** (the arm `1.0.0-f32.2` binds):

```
sudo BACKEND=rust-ffm bash run-clean-host.sh
```

**pure-java** (unchanged; this is what `evidence/` already holds):

```
sudo bash run-clean-host.sh
```

Either command may take an explicit output directory as its one argument; it must be a directory
inside `release-pilot2/clean-host/`, because `outputLog.uri` is derived from its name.

The wrapper pins and checksum-verifies its own toolchain (Temurin `jdk-25.0.4.1+1`, JBang 0.141.0),
downloads the 2.1 GB base and the bundle with fail-closed hash checks, resolves from Maven Central,
fingerprints the classpath, tees the program output, writes `clean-host-run.json` and finishes with
`SHA256SUMS` over everything it wrote. Budget roughly 10–20 minutes of measurement on top of the
download, on the pilot-2 host geometry.

`backend-native` is added on the JBang command line (`--deps`), not in `//DEPS`, so the pure-java
arm still resolves exactly the six jars its committed evidence records
(`resolvedClasspathSha256` `880b4f45a473abc15e937bafd29fcf2c4be79733281c0d1a39b17545a535ab47`). The
rust-ffm arm resolves those six plus `backend-native` and gives
`6c4649cb093f6fdaa0e6b8327dad12ce29f0501ef1a0a9c70180dd91f2807667`; a different value on the fresh
host is a finding, not a formality.

## 3. What the rust-ffm run records beyond the pure-java one

`clean-host-run.json` keeps every field `requireCleanHostRun` reads and adds two the gate ignores:

```
"backend": "rust-ffm",
"nativeLibrary": {
  "platform": "linux-x86_64",
  "abi": 5,
  "library": "libjmodels_kernels.so",
  "sha256": "b74d6e386d8494455b10698c1042e8d9e746cc60c8dce3b936b7cb5c203589d8",
  "source": "bundled-classpath",
  "kernelPlan": "rust-ffm-v13"
}
```

On `linux-aarch64` the library hashes to
`a6aad94358bb931b1c8199e079e21a470d9d4b6a87555080a33dc6b45ccac83b`. The digest in the record is
measured over the classpath resource on the host, not copied out of the payload's
`native.properties`, and a mismatch aborts the run before any inference happens.

Neither field is taken on trust. `write_clean_host_run.py` derives both from the program log and
refuses to record `pass: true` unless every backend statement the program made agrees with the
declared backend:

* the `loaded backend=` line names it;
* all six `CASE ... backend=` lines name it;
* a `native-library` line is present for rust-ffm and absent for pure-java.

A pure-Java log relabelled as a Rust run therefore fails instead of being believed.

**The switch is observable, both ways.** The program reads the loaded execution plan back and
refuses to print `PASS` unless the selection actually changed it:

```
rust-ffm    injectedGroupedProjections=true   matrixKernel=rust-ffm-quantized-v13
pure-java   injectedGroupedProjections=false  matrixKernel=vector-api
```

A native backend that was selected but silently inert would otherwise produce a passing record that
means nothing.

The final verdict line is still exactly `PASS cases=6 passed=6`, byte-for-byte what the pure-java
evidence carries, so the gate's log check is unchanged; the backend is stated on every other line.

## 4. Outputs to commit

Everything the wrapper writes under `evidence/` (pure-java) or `evidence-rust-ffm/` (rust-ffm):

| File | Role |
| --- | --- |
| `clean-host-run.json` | the gate record |
| `clean-host-output.log` | `outputLog` — its sha256 and size are bound in the run JSON |
| `resolved-classpath.txt` | `resolvedClasspathSha256` is the sha256 of these bytes |
| `fresh-check.txt` | the `freshMachine` derivation |
| `jbang-resolve.log`, `program-sha256.txt`, `toolchain.txt`, `java-version.txt`, `host-baseline.txt`, `wrapper.log` | provenance |
| `SHA256SUMS` | checksums of every other committed evidence file |

Then replace the literal `<EVIDENCE_REVISION>` in `clean-host-run.json`'s `outputLog.uri` with the
`integrallis/models` commit that contains the log.

## 5. The already-committed pure-java evidence is now older than the sources

`evidence/` was produced by the program as it stood before the backend switch existed. It still
satisfies the gate — every field `requireCleanHostRun` reads is present and unchanged, and the gate
ignores keys it does not know — but two things about it are now stale and should not be mistaken for
a mismatch found on a host:

* `evidence/program-sha256.txt` records `AnswerabilityCleanHost.java`
  `7f730564f49f03e6291179d884f6902e501d0aa5318f5fb2eafc4caa2ef48d8f`; the file in the tree is
  `0a15f489535a9fb31d42c2191c3c99b8e90d9a66b3f1713dd84011024c193148`.
* `evidence/clean-host-run.json` predates the `backend` and `nativeLibrary` fields, and
  `evidence/clean-host-output.log` predates the enriched `loaded` and `CASE` lines.

`resolvedClasspathSha256` is *not* stale: it is still `880b4f45...`, which is exactly why
`backend-native` is added with `--deps` instead of `//DEPS`. Re-running `sudo bash run-clean-host.sh`
on a fresh host regenerates `evidence/` against the current sources; the outputs it must match are
unchanged.

## 6. Unit tests

```
python3 -m unittest write_clean_host_run_test -v   # from this directory, 22 tests
```

The last test re-runs a passing record through `assemble_component_report.validate_clean_host_run`
as fetched from `origin/main`, for both backends, and skips if that file is unreachable.

## 7. Local verification already done (dev machine, not a clean host)

Measured on 2026-09-17 on a developer macOS host (Intel i7-9750H, `os.arch=x86_64`, Temurin 25.0.3),
against the same pinned inputs, resolving `backend-native:0.3.42` from Maven Central:

* **rust-ffm** — all six cases PASS, byte-identical to
  `../evidence/main/window-*-specialist-rust-ffm.json` with the same `sharedPrefixTokens`, native
  library resolved from the bundled `macos-x86_64` payload (sha256
  `80375639bd0379a91c684492ebc061a244ea4b72013661d54f1b6fe80a9b25ea`, ABI 5, plan `rust-ffm-v13`),
  `injectedGroupedProjections=true matrixKernel=rust-ffm-quantized-v13`.
* **pure-java** — same run on the default path, `injectedGroupedProjections=false
  matrixKernel=vector-api`, no `native-library` line. This is the negative half of the ablation: the
  switch is visible in the loaded plan in both directions.
* **negative paths** — `--backend rust-ffm` without `backend-native` on the classpath fails with the
  mirrored "rust-ffm qualification requires the optional backend-native runtime" message;
  `--backend <anything else>` exits 64.

That is a *correctness* check of the program, not the clean-host proof: the machine has a populated
`~/.m2` and `~/.jbang`, so `freshMachine` is false there by construction and no
`clean-host-run.json` from it can pass. The fresh Ubuntu 24.04 run is still required.

`injectedGroupedProjections=true` on the Rust arm is measured on `macos-x86_64` only. It is
*expected* on Linux — same model, same planner, same ABI 5 kernel, only a different build target —
but it is not measured there. If it turned out false on the fresh host the run fails loudly rather
than passing with an inert kernel, which is the direction that error should take.

## 8. What this run does not, and cannot, establish for the gate

* **The evidence revision.** `outputLog.uri` carries `<EVIDENCE_REVISION>` until the log is
  committed; the gate then fetches the bytes and checks size and sha256 against the record.
* **That the report is rust-ffm-bound.** `requireCleanHostRun` never looks at a backend — the new
  `backend` and `nativeLibrary` fields are carried through untouched and nothing downstream
  cross-checks them against `taskCorrectness.backend`. What makes a report rust-ffm is the
  `--window` arms handed to `assemble_component_report.py`; the clean-host record follows, it does
  not decide.
* **`kernelIdentity`, which becomes required the moment `taskCorrectness.backend != "pure-java"`**
  (≥10 cases per suite per arm, `identicalOutputs` true). Measured over the committed
  `../evidence/main/identity10-*` pairs: the two **specialist** pairs are identical 10/10, and the
  two **base** pairs are not (squad-v2-dev 2/10 differ, msmarco-v2.1-validation 3/10). Since
  `kernel_identity()` fails if *any* supplied pair differs, only the specialist pairs can be passed
  as `--identity`. The already-assembled `component-report/component-qualification.json` has
  `kernelIdentity.pass: false` with `casesPerSuitePerArm: 0` — harmless while the backend is
  pure-java, fatal once it is not.
* **That the two backends agree beyond those ten cases.** They do not, and nobody should assume it:
  over the full 200-case windows the outputs differ on 2/200 (squad specialist), 6/200 (msmarco
  specialist), 17/200 and 25/200 (the base arms). `sharedPrefixTokens` agrees everywhere. This is
  why a rust-ffm report has to be assembled from the rust-ffm window arms, and why the six-case
  identity check here is pinned to the rust-ffm evidence rather than the pure-java table.
* **`taskCorrectness` on the raw window numbers.** The gate needs `balancedAccuracy >= 0.8` per
  suite; the rust-ffm window arms report 0.865 (squad) and **0.725** (msmarco) as scored against the
  dataset labels. The pure-java report clears the bar only through `--labels`, which re-scores both
  arms against confirmed labels (msmarco 0.735 → 0.825). The same `--labels` files must be applied
  to the rust-ffm arms, and the result re-derived rather than assumed to carry over.
