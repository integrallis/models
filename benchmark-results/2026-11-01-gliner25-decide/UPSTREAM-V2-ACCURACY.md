# Original GLiNER2.5-Decide: upstream fidelity and disclosed accuracy

On 2026-11-01 the user approved “upstream fidelity plus performance and disclosed accuracy.”
The new `catalog-policy-v2.json` requires at most 0.5 percentage points of loss against the
same original checkpoint in Fastino’s runtime. All existing numerical and performance gates
remain. This supersedes the local absolute accuracy floors only for admitting the original
publisher model; the earlier failed verdicts and fine-tuning targets remain historical records.

Measured on the pinned original F32 weights, with 8 CPU threads per arm:

| Cohort / score | Models | Fastino runtime |
| --- | ---: | ---: |
| Publisher public development, one head per call, mean domain exact match | 63.455882% | 63.455882% |
| Publisher public development, joint heads, mean domain head exact match | 63.715686% | 63.715686% |
| Publisher public development, joint heads, all-heads-correct document exact match | 48.823529% | 48.823529% |
| Complete GoEmotions test, micro-F1 | 36.076701% | 36.076701% |
| Complete GoEmotions test, macro-F1 | 32.094050% | 32.094050% |

The selected labels match on every one of 10,027 requests, with no inference errors.
Those requests cover **7,127 unique source documents**, because the 1,700 publisher documents
are evaluated jointly and as 2,900 separate heads. This is the publisher's public development
split, not a reproduction of its private 5,100-document headline test. The README's one-head
recipe was evaluated independently from the joint-head recipe. Scores were independently
recomputed with scikit-learn 1.7.2, and emotion text, gold annotations and label order were
checked against the official 5,427-document test split.

The earlier rare-label-enriched 259-case emotion subset scored 43.686007% micro-F1 in both
runtimes. The larger official test is lower, not higher. We found no port-induced prediction
loss; the upstream model is weak on this zero-shot emotion schema. No publisher-reported
GoEmotions F1 was found. Older entity-extraction F1 claims are a different task and model.

`upstream-v2-quality.json` binds all 56 disclosed metrics, raw predictions, independent score
reports and numerical parity reports by SHA-256. It passes fidelity, but deliberately does not
claim performance or catalog admission. Accuracy workers ran concurrently; their timing is not
used for performance qualification. Both audit instances and their root volumes were deleted;
cleanup receipts are retained in their respective run directories.

Three sequential, counterbalanced full performance pairs are running under
`gliner25-eval-20261101T234435Z`. Released-runtime verification and catalog publication remain.
