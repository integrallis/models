# Released fused attention and native pool qualification

The preselected native8/decode8/fused configuration passed its screen and both
counterbalanced fresh confirmations. All raw text, token counts, prompt hashes,
grounding and evaluations match the control. All default correctness smokes pass.
No native override, new runtime path or relaxed qualification gate was used.

| Trial | Decode tok/s | Decode ratio | p95 E2E ratio | Verdict |
|---|---:|---:|---:|---|
| screen-pool16-fused | 63.658 | 0.8094 | 1.4458 | QUALIFIED |
| screen-pool8-fused | 64.808 | 0.8317 | 1.2317 | QUALIFIED |
| confirm-1 | 64.824 | 0.8275 | 1.2540 | QUALIFIED |
| confirm-2 | 64.169 | 0.8209 | 1.0855 | QUALIFIED |

This comparison isolates native pool size while fused attention is fixed. The
separate 2026-10-31-qwen05-q5-attention experiment isolates fused attention while
pool size is fixed; both raw paired controls are retained. The profile combines
these independently measured recommendations, scoped to the measured CPU/JVM.
No cross-host timing comparison is used. The optional 0.85 decode margin target
was not reached; the unchanged production floor is 0.80.

Canonical catalog evidence uses the final confirmation, not the fastest result.
Quality scope: nine general RAG cases, 66.7% model contribution, 100% correctness
among model-contributed answers. This does not establish broad model accuracy.
