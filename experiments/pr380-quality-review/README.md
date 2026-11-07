# PR 380 quality review

This review compares the pinned PR base, the original PR head, and the reviewed source commit. See [generated score tables](COMPARISON.md), [exact revisions and tool settings](results/provenance.json), and the unmodified reports under `results/{main,before,after}`. MFCQI measures production Java; it does not score Rust, tests, or model accuracy.

**Coverage caveat discovered in the subsequent CLI pass:** MFCQI 0.7.0's default AST parser skips files using modern Java syntax. The scores below are reproducible but incomplete as a full-source quality assessment. See the [CLI hotspot review](cli-hotspots/README.md) for measured parser coverage and method findings with modern syntax enabled. A corrected main/PR score comparison requires rerunning both arms after fixing analyzer language configuration.

The aggregate is slightly down from main: **67.1447 → 67.1400 (−0.0047 score points)**, effectively flat at two decimal places. The original PR was 67.0386; the review improves it by 0.1014 points. Java backend, native backend and API scores exceed main. Benchmark tooling remains below main by 0.4145 points, although the review recovered 0.1480 points there. Eighteen other scopes are unchanged. Added validation slightly lowers the native module score relative to the original PR; its correctness benefit takes precedence over that metric movement.

The remaining release decreases are concentrated in complexity, coupling and smell components; documentation, duplication and cohesion improve relative to main. No further changes were made merely to force the aggregate across main's score.

## Findings addressed

| Priority | Finding | Resolution and evidence |
|---|---|---|
| High | The added Qwen3.5 `forwardEmbeddings` entry point advanced the session without recording its embedding prefix. Rewind replayed zero-filled token history and changed subsequent logits. | Removed the unwired entry point and its isolated tests; restored this class to main. The supported LLaVA path uses the Llama decoder. A tiny synthetic token-3 prefix reproduces the defect on the original PR; see the saved [test patch](results/qwen-rewind-reproducer.patch) and [expected failing run](results/rewind-reproducer.log). Token-zero tests had hidden the problem. A future Qwen multimodal API must implement replay and position semantics before exposure. |
| High | Native causal attention accepted aliased output/input arrays; retained attention eligibility could overflow shape arithmetic. | Reject aliasing before native dispatch or state changes, and check shape products using long arithmetic. New native tests cover alias rejection, oversized dimensions and successful continuation after rejection. |
| Medium | Cleanup failure could mask a model-load failure or skip arena cleanup. | Use try-with-resources on load failures so both resources close and secondary failures remain suppressed on the primary exception. |
| Medium | The rejected Q8 conversion left an unused converter, tests, matrix copies and dispatch plumbing in production. | Removed those remnants. The earlier performance record already measured the Q8 path losing both fidelity and speed; no new fallback was introduced. |
| Medium | The benchmark entry point mixed option parsing, encoder measurement and decoder measurement. | Extracted encoder and decoder measurement methods and a measurement-options record. Prompt, sampling, timing boundaries and report fields are unchanged. |
| Low | The public encoder constructor depended on an inaccessible internal weight type; several public API contracts lacked units or ownership guidance. | Made the internal encoder package-private and documented supported projector/configuration APIs, ownership, image layout and scratch-buffer concurrency limits. Restored the text-load Javadoc. |

The Rust pass inspected packed Q4 bounds and reduction order, AVX2 dispatch, ragged tails, Q6 tiling, dense causal pointer contracts and disjoint head ownership. This review changes no Rust implementation or numerical kernel. Existing kernel tests and prior remote performance evidence remain relevant; this is not a formal proof of native memory safety.

## Validation and limits

`models-api:check`, `backend-java:check`, `backend-native:check`, and `models-bench:check` passed after the main fixes. After removing the final unused Java encoder dispatch, `backend-java:check` and `models-bench:check` passed again. These checks include formatting and configured static-analysis/coverage rules. Logs and a per-suite XML-derived [test summary](results/test-summary.json) are retained. Tests tagged `model-fixture` were excluded: no real-model inference ran on the development Mac.

The Qwen rewind reproduction is deliberately a failing test on the original PR snapshot. Its production sources were unmodified during the original-PR MFCQI run, and test code is outside MFCQI's source scope.

No new full-model performance claim is made for the review commit. The prior paired remote performance result remains pinned to `0f2bb7d6a4b30c6b5297d86456b9320388d86707`; its measured numerical kernels are unchanged. Published-task VQA/OCR qualification and catalog packaging remain separate work.

## Reproduce the comparison

Use the exact three source revisions in `results/provenance.json`, MFCQI 0.7.0, Gradle 9.4.1 and Java 25.0.3. For each checkout, stage into a fresh directory (the script rejects an existing destination):

```sh
python3 experiments/pr380-quality-review/stage.py CHECKOUT /tmp/quality-source
```

Run from that checkout, passing the review's init script by absolute path:

```sh
JAVA_HOME=/path/to/java-25.0.3 ./gradlew --no-daemon --no-parallel --max-workers=1 \
  '-Dorg.gradle.jvmargs=-Xmx4g -XX:MaxMetaspaceSize=512m' \
  -I /path/to/analyze.init.gradle mfcqiAnalyze \
  -Pmfcqi.sourceDir=/tmp/quality-source -Pquality.output=/tmp/quality-results
```

The release source set matches `.github/workflows/mfcqi.yml`: ten published modules. Gradle also emits individual reports for all source-bearing Java modules, including benchmark tooling. Analyzer parallelism is fixed at one. Existing repository metric weights and exclusions are unchanged; `bytecodeSecurity=false` and `failOnGate=false` match the repository. The initial default-heap run failed and was discarded; every retained arm uses the same 4 GiB heap. Source manifests and the plugin JAR hash are included for auditability.

Copy each arm's reports into `results/main`, `results/before`, and `results/after`, then run `python3 experiments/pr380-quality-review/compare.py`. The comparison includes every reported scope and release metric component, including unchanged low components; no metrics were excluded to improve the result.
