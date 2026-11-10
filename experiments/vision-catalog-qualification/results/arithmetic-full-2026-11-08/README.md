# Full arithmetic-candidate attempt — 2026-11-08

Candidate c7acc590147032ae78adad01ae1e869b64922274, independently measured on two
m6a.4xlarge workers. Complete 80-case runs, fresh pinned reference on each host,
and both initial and longer integration confirmations. Per-directory receipts
and SHA-256 manifests bind the machine, runtime, cases and results.

Neither candidate qualifies in this attempt:

* Q4_K_S passes every performance gate, but answers `Rectangle` instead of
  `Rectangles` on VQAv2 454129000. Nonblank token agreement is 63/64 and VQA
  score loss is 1.25 percentage points, exceeding the frozen limits.
* Q4_K_M matches all 80 answer sequences and passes accuracy. Its first task
  has prefill/total latency ratios 1.11242/1.11381 versus the strongest reference,
  exceeding 1.10. The other three timed tasks pass.

Both longer integration confirmations pass against F32 and F16 references.
Those integration tests use the Q4_K_M decoder on both hosts; they do not
establish Q4_K_S decoder performance. Encoder pixels and embeddings are exact.

Full raw evidence is retained by the bounded worker archive collector under
`audit-artifacts/pr380-vision-arithmetic-qualification-2026-11-08` in the workspace.
No catalog admission or threshold is changed by these results.
