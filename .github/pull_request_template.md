<!--
Required second-reader review for PRs touching numerics or performance.
A single author writing, CI-checking, and merging is how comment-code mismatches,
unmeasured headings, and stub tests slip through.

Before requesting review, answer the three probes below.
The reviewer must verify each answer, not just read it.
-->

## Second-reader probes (numerics / performance PRs only)

- [ ] **Comment-code match:** Every code comment in the diff describes what the code actually does. No phantom guards, no "measured improvement" headings without measurement artifacts, no "4x" where the data says 7.11x.
- [ ] **Measurement exists:** The PR contains a committed artifact in `benchmark-results/` with the previous release tag (`git describe --tags --abbrev=0`) as an interleaved baseline arm. Not a CHANGELOG citation — an actual JSON/CSV/raw output with both arms run under identical conditions.
- [ ] **Test runs on real class in CI:** Every test cited as "verification" in the commit message or CHANGELOG instantiates the production class (not an in-test stub or artificial-latency mock) and runs in a non-skipping CI job. `@Tag("integration")` tests that are excluded from the `build` task do not satisfy this probe.

## Performance checklist

- [ ] The **Reranker canary** workflow is green on this PR, or its red is explained here with the measurement on the registered host class that supersedes it

- [ ] Baseline measurement against previous release tag completed (`benchmark-results/`)
- [ ] No CHANGELOG claim contradicts the measured result
- [ ] Cold-load metrics measured for any change claiming allocation or memory reduction
- [ ] Per-change ablation against the released tag (not "a baseline" — the actual tag)

## Numerics policy

- [ ] A change that alters a qualified model's outputs is opt-in or re-qualified — never default-on
- [ ] ONNX/oracle tolerance test passes (this is a floor, not the rule)
- [ ] If outputs shift within tolerance, the pinned record is updated or the change is opt-in

---

## What this PR does

<!-- One sentence per logical change. Link the issue. -->

## Files changed

<!-- List the production files (not tests, not docs) that changed. -->

## Verification

<!-- For each test cited as evidence: name, whether it runs in CI, and what it actually exercises. -->

## Known gaps / follow-up

<!-- What this PR does NOT do, and the issue that tracks it. -->
