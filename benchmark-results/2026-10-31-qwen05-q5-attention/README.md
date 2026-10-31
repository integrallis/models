# Released fused attention experiment

Fused attention improved observed decode and preserved raw answers, counts, prompts
and evaluations. Both fresh confirmations passed, but the initial fused screen
missed the p95 E2E ceiling. The registered all-three-pass criterion therefore
remains false. No catalog qualification was changed.

| Trial | Fused | Decode tok/s | Decode ratio | p95 E2E ratio | Verdict |
|---|---|---:|---:|---:|---|
| screen-head-by-head | False | 59.160 | 0.7921 | 1.4443 | FAILED_RELATIVE_GATE |
| screen-fused | True | 64.951 | 0.8316 | 1.5195 | FAILED_RELATIVE_GATE |
| confirm-1 | True | 64.380 | 0.8289 | 1.3665 | QUALIFIED |
| confirm-2 | True | 64.452 | 0.8330 | 1.4248 | QUALIFIED |

The next registered experiment combines fused attention with the independently
measured smaller native prefill pool. All runs use Maven Central product bytes.
Native grouped attention capability is advertised, but Qwen does not activate
that route: its architecture selects the Java head-by-head or fused implementation.
No runtime kernel or library default is modified by this experiment.
