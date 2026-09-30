# Certified RAG evidence, 2026-09-30

The first qualifications produced by the two-arm fleet worker. Both arms ran on the same
`m6a.4xlarge` in one session: the candidate through `rust-ffm` on the downloaded GGUF, and the
comparator through Ollama on a model imported from **that same file**, so `artifactSha256` matches and
the comparison is not excluded. Declared thread budget 16 for both arms, context 2048, 256 max output
tokens, 1 warmup and 3 iterations, general workload.

Every number here is *measured on that host*. `CertifiedRagEvidenceTest` re-runs
`RagProductionQualificationPolicy.assess` over these exact files and pins the verdict and both
ratios, so a policy change that would alter either has to alter the test too.

| model | decode ratio | end-to-end ratio | verdict |
| --- | --- | --- | --- |
| `qwen3.5-2b-q4_k_m` | 0.9016 | 1.2264 | QUALIFIED |
| `gemma-3-4b-it-q4_k_m` | 0.8740 | 0.9582 | QUALIFIED |

Gemma 3 4B is faster than the comparator end to end, not merely within the ceiling.
