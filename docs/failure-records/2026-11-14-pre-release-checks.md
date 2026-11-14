# Pre-release check failures — 2026-11-14

## What almost shipped in release/0.5.21 (fcf7fe2f)

The release branch was cut from 8daeeae1, before #456 merged. It does NOT contain the ragged-batch fix. Its `BertForwardPass` has no guard: `scoreAll` on that branch throws on any batch of unequal-length documents on the pinned reranker. The branch must be re-cut from main after #454 lands.

### 1. scoreAll threw on the public API with 38 green checks (#456)

**The catastrophe:** `GgufRerankingModel.scoreAll(query, documents)` threw `IllegalArgumentException` for any ragged batch (documents of different lengths) because `encodeBatchCls` required all sequences to have the same length. This was reachable through the public API.

**Why missed — structural, not personal:**
- The parity test that would have caught it (`MsMarcoMiniLmRerankerIntegrationTest`) is `@Tag("integration")`, excluded from the `build` task.
- The two unit tests cited as "verification" in #440 and #448 instantiate in-test stubs of `RerankingModel` (artificial-latency mocks) that never reach `BertForwardPass`.
- No test exercised the production `GgufRerankingModel.scoreAll` with unequal-length documents.

**Root cause:** #440 added batching with padding to `maxLen`, but the truncation path produced documents of different actual lengths, and the attention masking logic assumed `lengths[]` would handle it — but `encodeBatchCls` had a strict `all sequences must have the same length` check that rejected ragged inputs.

**Bisect:** #458 confirmed the throw at `3b37bc3c` (#440). The fix (#456) adds a guard that pads shorter sequences to `maxLen` before calling `encodeBatchCls`.

### 2. #360 pre-dequantization: default-on broke ONNX parity; commit claimed measurement that didn't exist

**What was merged:** BERT weight pre-dequantization (Q4_K/Q5_K/Q6_K → F32 at load time), commit heading "Measured improvement:" with no measurement, code comment claiming a guard that didn't exist, scope wider than its rationale, "4x" where the header says 7.11x.

**Reality:**
- Fails `MsMarcoMiniLmRerankerIntegrationTest` ONNX equivalence by 0.075 vs. 0.05 tolerance when default-on.
- The integration test that catches this is `@Tag("integration")`, excluded from build; it could not have run in CI.
- Must be **opt-in-off** (as #454 makes it), not default-on.

**Why missed:**
- The commit heading and comment contained claims the code didn't support.
- No adversarial review: the same agent wrote, CI-checked (unit tests only), and merged.
- The change altered a qualified model's outputs without re-qualification — violating the policy #408 already articulated.

### 3. Reranker throughput regression: 27% slower score(), 33% slower scoreAll() (#457)

**Measured vs. 0.5.20:**
- `score()`: -27% (p50)
- `scoreAll()`: -33% (p50)
- Numerics: identical

**Bisect results (from #458):**
- Scoring step regression: pinned to `3b37bc3c` (#440)
- Cold-load regression: pinned to `63b46a83` (#435/#437, trie constructor)

**Two distinct culprits, not one:**
- **Scoring:** #440's `encodeBatchCls` is unchanged for the sequential path, but is now first compiled inside the timed loop. The compile-time artifact is too small to explain a p50 move with flat p95. Leading hypothesis: profile pollution of shared callees (the trie constructor from #435/#437 affects the JIT profile of methods both paths share).
- **Cold-load:** The trie constructor `TrieNode` builds a prefix tree from the full vocabulary at `GgufTokenizer` construction. On MiniLM-L6 (30,522 vocab), this adds ~12ms to model open time.

**Why missed:**
- No benchmark was run against `v0.5.20` baseline before claiming improvement in CHANGELOG.
- The `RerankingModelBatchRoutingBenchmarkTest` added in #449 uses an artificial-latency mock — it tests routing logic, not real BERT forward-pass throughput.
- The `MsMarcoMiniLmRerankerPerformanceExperiment` (which measures real model) was not run against the previous release tag.

**Decisive run specified in #457:** warmups=0, 300 iterations, both arms, on the next available box.

### 4. #440 CHANGELOG false claim

**Claim:** "amortizes weight-cache locality and reduces small-matrix overhead"

**Reality:** 33% slower on `scoreAll()` at identical numerics.

**The CHANGELOG entry must be corrected before any release ships.** Either restate as regression with investigation pending, or remove entirely.

### 5. Lockfile fix smuggled into unrelated commit

The `models-decisions` and `models-embedding` lockfiles were updated to vectors 0.1.35 inside `8daeeae1` (perf commit "F16 KV read-into avoids allocation"). A dependency fix inside a perf commit is unreviewable and pollutes history. It should have been a separate commit.

### 6. Documentation version strings stale

`docs/content/antora.yml` and `models-spring-boot-starter/README.md` still cited vectors 0.1.34 after the upgrade. Caught by `:verifyDocumentation` (build.gradle.kts:1300) — the guard already existed and fired. This is "didn't check established procedure", not a missing guard.

## Guards that would have caught today's failures

### (a) Per-model real-fixture integration tests run in CI for their modules

**Enforcement point:** The `build` task currently excludes `@Tag("integration")` tests. Add a non-skipping CI job that runs `MsMarcoMiniLmRerankerIntegrationTest` on any PR touching `BertForwardPass.java`, `GgufRerankingModel.java`, `BertWeights.java`, or `GgufTokenizer.java`.

**What it catches:** The #456 throw and the #360 parity break would both fail this job.

### (b) Perf PR cannot merge without a committed artifact containing the previous release tag as an interleaved arm

**Enforcement point:** A Gradle task (`verifyPerformanceRegression`) that:
1. Checks if the PR touches performance-sensitive paths (tokenizers, KV cache, BERT forward pass, matrix kernels)
2. If yes, looks for a `benchmark-results/` subdirectory with a `baseline-<previous-tag>` arm
3. Fails the build if no such artifact exists

**What it catches:** The #440 and #435/#437 regressions would have required baseline measurement before merge. The #360 commit heading "Measured improvement:" would have needed an actual artifact.

**Not the same as:** Checking that a CHANGELOG bullet cites a path. #360 cited a measurement that didn't exist.

### (c) Required second-reader review for numerics/performance changes, with three explicit probes

**Enforcement point:** A GitHub branch protection rule or a PR template check that blocks merge until:
1. **Comment-code match:** Does every code comment describe what the code actually does? (catches #360's phantom guard)
2. **Measurement exists:** Does the PR contain a committed artifact with the previous release tag as a baseline arm? (catches #440, #435/#437)
3. **Test runs on real class in CI:** Does every test cited as "verification" instantiate the production class and run in a non-skipping CI job? (catches #440's stub tests, #456's missing ragged-batch coverage)

**What it catches:** The comment-vs-code mismatch, the unmeasured heading, the stub tests, and the missing integration coverage.

## Additional policy rules (no code needed)

- A change that alters a qualified model's outputs is **opt-in or re-qualified** — never default-on. The ONNX tolerance test is a floor, not the rule.
- Per-change ablation against the released tag, not just "a baseline." The masking effect (#360-on cancelled #440's slowdown) would show up as two independent regressions, not one invisible cancellation.
- Load-time metrics must be measured for any change claiming allocation reduction. A 43% cold-load regression from a "memory optimization" would pass every other proposed gate.

## Status of 0.5.21

- **#456 merged to main:** ragged-batch fix
- **#458 merged to main:** bisect evidence
- **#454 waiting (28/29 green):** makes #360 opt-in-off, adds `models-tinystruct`
- **#457 blocks release:** reranker regression, decisive run pending
- **release/0.5.21 (fcf7fe2f) is defective:** lacks #456, must not ship, must be re-cut from main after #454 lands
- **CHANGELOG must be corrected:** #440 claim restated or removed
