# All-activation-only candidate: rejected by extended fixtures

GitHub run38321902032 uses Models4f3d3e9a and exact Vectors8495fbf9; the frozen
protocol and full artifact identities accompany the raw outputs. All eight jobs
(x86/ARM, Bonsai/Qwen, two fixed shards) failed the candidate's repeated fresh-session
check: `fresh session changed after compilation`. Baseline failures are retained too.
Every job stopped before timing; no performance result exists for this candidate.

The unchanged repository Bonsai JUnit tests had passed in run38321333816. That evidence
is narrower: the separate process fixture exercised a compilation/session history that
still failed. The JUnit pass does not qualify this artifact. These results are not pooled
with the broad BF16-plus-activation candidate's separate run38321278750.

Bonsai2 is Qwen3.5 (`qwen35`), established directly from the exact model's GGUF header
and complete SHA256, not inferred from its name. Its sigmoid/softplus and convolution
activation paths are part of this ablation. Both baseline and candidate fixtures emitted
actual runtime origins; provenance records every JAR entry and native-library hash.
