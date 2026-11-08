# Rejected attention dispatch experiment

Same-host baseline/candidate/baseline, three warmups and seven measured repetitions.
Candidate 0a000e7c preserves all output tokens but regresses median decode by
0.63–0.85% for the caption and 0.55–0.76% for the short answer, beyond the
0.21–0.22% baseline bracket variation. No full qualification repeat was triggered.
The production experiment was removed; only diagnostic source and evidence remain.

The short-answer candidate/reference ratio of 1.0613 is not an optimization win:
the unchanged baseline is faster. This diagnostic cannot qualify either model.
The subsequent reference-attention component probe runs only after all timing ends.
