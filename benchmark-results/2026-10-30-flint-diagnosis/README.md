# Flint Q4_K numerical diagnosis

Runtime: `23e4e083ec311a216a8d19d78084dfefe82f3133`.
Diagnostic source and worker: `f68555dee4c811ce19f659906220834ca1028808`.
This diagnostic followed the failed parity gate in `../2026-10-30-flint-improvement/`.
It does not qualify accuracy or performance.

On the pinned completion prompt and its first three teacher-forced predictions:

| Execution | Third token | Maximum absolute logit difference from Java |
|---|---:|---:|
| Java | 2635 | 0 |
| Native, grouping disabled | 63509 | 2.9800568 |
| Native Q8_0 only | 2635 | 0 |
| Native Q4_K only | 63509 | 2.9800568 |
| Native projections checked, Java results propagated | 2635 | 0 |

Q8_0 projections matched Java bit for bit on identical inputs. Q4_K differences reached
0.00048828125 per projection and propagated to changed predictions. Disabling grouping
reproduced the discrepancy, so grouped expert scheduling is not required to trigger it.
The source comparison identifies the different floating-point reduction order in Q4_K.
The ordered-kernel correction requires a fresh independent full parity run.

Timing includes different cache/JIT states and diagnostic overhead; it is not a speed comparison.
Only three continuation positions were checked here, not a general accuracy sample.
The worker completed successfully, requested shutdown, and subsequent EC2 queries returned
neither instance `i-0911789873d1ff527` nor root disk `vol-0bdddc0e14f572545`.
Raw reports are unchanged, logs are losslessly compressed, and deployment inputs are retained.
