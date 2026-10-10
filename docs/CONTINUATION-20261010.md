# Router measurement continuation, 2026-10-10

Base: `ae1cd15e405a2d8748fc9c404be3b06ed388bed6` (current Models 0.3.56 main).

This branch collects the uncommitted `RouterObservationCollectorCli` and its four-line command
registration from `models-router-enhancements`, rebased onto current main. The source worktree and
an exact workspace backup remain untouched. Formatting was applied here.

`./gradlew :models-bench:test` passes and the collector compiles. No model was downloaded and no
quality/performance campaign was run on the development Mac. This is **not ready to merge**:

- Persist the seed, sampling options, backend/version, artifact hashes and classifier hash in the
  evidence, rather than retaining hashes in memory or printing them only to stdout.
- Validate unique model IDs and case IDs; a duplicate model ID currently replaces a loaded backend.
- Close the classifier and embedding backend on every exceptional path. Scope or restore the
  process-wide context-length property.
- Validate each model's chat template instead of imposing CHATML_NO_THINK on every model.
- Add focused parser/lifecycle/receipt tests; then collect the paired campaign using the runbook.

The separate open `docs/candidate-inventory` branch/PR #258 is retained for focused review. Its
filename-only evidence matcher and missing sibling-catalog behavior need correction before merge.
Keeping that PR separate avoids mixing an inventory repair with unfinished router measurements.

Historical experiments are preserved under local `refs/archive/audit-20261010/heads/…`:

| Original branch | Disposition |
| --- | --- |
| `exp/decisions-gpu` | Preserve SemIf/cascade raw evidence and unproved conclusions; start any new campaign from this branch's current-main base. |
| `feat/system-one-named-questions` | Preserve the dated Laya comparison and its limits. |
| `feat/cross-model-kv-sharing` | Preserve the rejected Qwen3 translation evidence; no claim of a usable cross-model cache. |
| `feat/alora-cache-sharing`, `perf/worker-poll-budget`, `exp/granite-alora-first-candidate` | Preserve historical adapter/checkpoint provenance and rejected/partial measurements; released implementations have moved on. |
| `exp/gpu-large-model-campaign` | Preserve the rejected dispatch hypothesis and resident-activation preregistration; current main contains later device work. |
| `exp/tornado-attention-oracle`, `feat/tornado-attention` | Preserve historical oracle code; the later 0.3.53 release deliberately removed Tornado. Do not restore that backend implicitly. |

The workspace audit's preservation manifest gives every full SHA and restoration command. Archives
retain the original complete tree and evidence; they are not claims that an old experiment passes
current qualification. No old backend code or historical qualification numbers were copied over main.
