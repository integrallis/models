# V12 release-gate result

Decision: **REJECTED — not releasable**.

| Gate | Frozen requirement | Observed | Result |
|---|---:|---:|---|
| Python evaluator tests | all pass | 90/90 | PASS |
| Java / Hugging Face prompt bytes | exact | 1,072/1,072 | PASS |
| Java / Hugging Face prompt token IDs | exact | 228/228 | PASS |
| Activated invocation | `[151644, 77091, 198]` | exact | PASS |
| Exposed syntax | 75/75 | 75/75 | PASS |
| Exposed schema | 75/75 | 75/75 | PASS |
| Exposed adapter tool exact | at least 43/50 | 42/50 | **FAIL** |
| Adapter vs same-policy base exact | no worse | 42/50 vs 45/50 | **FAIL** |
| Exposed false tool calls | at most 1/25 | 5/25 | **FAIL** |
| Live-development screen | exposed must first pass | not run | STOPPED |
| Sealed 300-case qualification | exposed and live must first pass | not opened | STOPPED |

The policy-only hypothesis is falsified for this candidate. Perfect syntax and schema do not
compensate for worse selection correctness or a 20% false-call rate. The unchanged V9 weights,
policy, prompt oracle, evaluator outputs, and raw log are retained as reproducible rejection
evidence; no artifact may be published from V12.
