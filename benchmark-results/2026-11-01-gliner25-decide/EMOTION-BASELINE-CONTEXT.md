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

The running general-v3 fine-tune is around 0.55–0.56 micro-F1 on a different,
243-case development cohort reserved from published training data. These scores
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
