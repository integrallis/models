# Gemma 4 26B-A4B Q4_K_M library-default smoke

This directory proves that the exact qualified Gemma artifact passes the
nine-case guarded-RAG correctness workload before any model- or host-specific
performance tuning is applied.

- Models commit: `37dec2680637eb22b5ee59001e53af11d8d3946b`
- artifact SHA-256: `88f4a13b0bb95f031a7fad973e10854122fb67ebc34d214d39a2f65053046abc`
- checked-in report SHA-256: `74739c5166a445ba97cbb05701cddea47dfda03d8af119836b53f374e04b77d5`
- result: 9/9 successful, 100% correct-answer rate, 100% abstention accuracy
- prompt cache: longest-common-prefix
- `models.native.quantizedDecode`: `false`
- `models.native.loadWarmup`: `false`
- tuning system properties: none

The run used Java 25.0.3 on the controlled 8-vCPU AMD EPYC-Milan Linux host.
It is a correctness and onboarding gate; the separately retained tuned report
remains the performance qualification evidence.
