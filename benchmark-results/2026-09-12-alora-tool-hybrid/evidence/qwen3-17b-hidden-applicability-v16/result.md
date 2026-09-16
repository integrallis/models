# Qwen3 1.7B hidden-applicability V16 result

V16 is rejected. It is not a releasable hybrid model.

The frozen production-Java Q4 gate completed three of its 75 exposed development observations:

| Case | Expected | Predicted | Score | Physical shared prefix |
| --- | --- | --- | ---: | --- |
| `irrelevance_126` | no call | call | 2.03176 | yes |
| `irrelevance_165` | no call | call | 8.34644 | yes |
| `irrelevance_169` | no call | no call | -16.29031 | yes |

Two false calls already make the fixed requirement of at least 24 correct no-call decisions out of
25 impossible. The run therefore stopped immediately instead of spending more compute. The
production path did prove that each observed activated branch referenced the same physical KV
prefix as its base branch, but physical sharing alone is not sufficient for qualification.

The offline BF16 validation result—389/390 calls, 79/81 no-calls, 0.986372 balanced accuracy—does
not transfer unchanged to the Java Q4 runtime. V16's fixed zero threshold cannot be adjusted after
seeing these results. Any quantization-aware calibration must be a new, separately frozen
experiment with its own development split and untouched qualification gate.

The exposed dual-generation gate, sealed 300-case qualification, packaging, catalog publication,
and release were not run. The sealed data remains unopened.
