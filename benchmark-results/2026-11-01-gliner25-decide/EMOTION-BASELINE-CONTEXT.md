# Emotion quality: upstream baseline versus our admission targets

Checked 2026-11-01. Fastino's pinned model card and its current main card do not
report a GoEmotions F1 result. They report 60.2% exact-match accuracy averaged over
17 domains of fast-decisions, with 300 held-out examples per domain. That figure
cannot be compared numerically with GoEmotions micro-F1.

Sources:
- https://huggingface.co/fastino/GLiNER2.5-Decide/blob/9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90/README.md
- https://huggingface.co/fastino/GLiNER2.5-Decide/raw/main/README.md
- https://pub.towardsai.net/gliner2-5-decide-fine-tuned-local-system-one-model-jev-level-accuracy-7-faster-367f82c01bf7

The linked article's accessible introduction describes duplicate-detection
fine-tuning and a comparison with Jev, not an emotion benchmark. The remainder is
paywalled; no claim is made here about uninspected content.

Our original 259-case emotion cohort produced identical selected labels in the
publisher's CPU Python runtime and Models: micro-F1 0.4368600683, macro-F1
0.4395110220. See results-restart-1/quality-1-reference-report.json and
results-restart-1/quality-1-models-report.json. This is a measured same-input,
same-weight comparison, not a vendor claim. It does not show accuracy lost in the
Java/Rust port.

The completed general-v3 fine-tune selected step 9000 with 0.562701 micro-F1
and 0.562993 macro-F1 on a different, 243-case development cohort reserved from
published training data. These scores
are development diagnostics, not final qualification and not an apples-to-apples
improvement estimate against the original 259-case cohort.

GoEmotions has 27 emotion labels plus neutral, and examples may have more than
one label. Its original BERT baseline reported approximately 0.46 average F1:
https://arxiv.org/abs/2005.00547 . That published baseline provides task context;
it is not our comparator and uses a different evaluation setup.

The 0.70 micro-F1 / 0.60 macro-F1 gates in experiments/gliner25-decide/policy.json
are locally adopted admission targets. They are not Fastino-published guarantees
and were not established by reproducing a Fastino emotion result. Improving the
checkpoint to meet those targets is additional model development beyond proving
faithful runtime support. The targets remain unchanged in the recorded policy;
this note does not grant qualification or retroactively alter any verdict.


The independent scikit-learn audit reproduces the original 259-case metrics
exactly and verifies text, original label IDs, gold annotations and label order.
However, that cohort deliberately enriches rare labels: it includes all six
`grief` examples but only 23 of 1787 `neutral` examples in the official test.
Its micro-F1 must not be presented as the score on the full test population.
The direction and size of that sampling effect require the full-test run; rare
label enrichment alone does not establish that the sample understates quality.

The publisher's public fast-decisions data has 1700 development examples; its
5100-example official test is private. The public data is being evaluated using
original weights in both runtimes, with document and classification-head exact
match reported separately. Neither score is a reproduction of the private-test
headline. All 5427 official GoEmotions test rows are also being evaluated in both
runtimes. Inference timings from these concurrent accuracy jobs are not valid
paired performance measurements.

The original GLiNER2 paper reports F1 for CrossNER entity extraction, while its
classification table reports accuracy for seven other datasets. Those results
use an earlier model and do not provide a GLiNER2.5-Decide emotion baseline:
https://arxiv.org/html/2507.18546v1 .
