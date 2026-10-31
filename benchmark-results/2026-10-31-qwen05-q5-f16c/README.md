# Qwen2.5 0.5B Q5_K_M: F16C decode experiment

The two-conversion native change did **not** produce repeatable qualification. The
three candidate repetitions returned QUALIFIED, QUALIFIED, FAILED_RELATIVE_GATE.
All raw generated answers, token counts, prompts and evaluations matched the
baseline; no token-ID parity claim is made. Both source variants use the same
compiler and debug-symbol settings and released Java dependencies.

| Repetition | Baseline tok/s | Candidate tok/s | Candidate / reference decode | Candidate / reference p95 E2E | Candidate verdict |
|---|---:|---:|---:|---:|---|
| 1 | 61.607 | 62.143 | 0.8041 | 1.4587 | QUALIFIED |
| 2 | 61.147 | 62.914 | 0.8129 | 1.4478 | QUALIFIED |
| 3 | 62.423 | 61.208 | 0.7966 | 1.5064 | FAILED_RELATIVE_GATE |

Exact machine values are authoritative in `results/summary.json`. The candidate's
small changes versus baseline (approximately +0.9%, +2.9%, -1.9%) do not establish
a reliable performance improvement. Both quality metrics remain 66.7% model
contribution and 100% correctness among contributed answers on this nine-case RAG
corpus. This is not broad accuracy evidence.

Native release unit tests passed on Intel macOS and AMD Linux. Finite half values
and infinities retain their exact bits, as do scalar/AVX2 test rows. Source native
overrides are experimental; these results cannot admit a released catalog model.
The unproven runtime change was removed from the working branch. PR 319 will be
closed after its existing CI finishes; no competing runtime path is retained.

The next separate experiment tests released fused grouped attention. It uses no
F16C override and will not alter any qualification thresholds or existing model
qualification records.
