# Qwen3 1.7B hybrid live-decision V21 preflight

Status: **frozen before any V21 model score**.

V20 was rejected after its complete 75-case Java generation run. This distinct V21 experiment
tests a drift-tolerant decision threshold on the separate, already-exposed BFCL-live development
window from V18. It is not the sealed 300-case qualification window and cannot authorize a release.

## Frozen identity and procedure

- Models scorer revision: `e2dab1e2749199792237bd388e70fba04076335b`.
- Base Qwen3 1.7B Q4_K_M GGUF SHA-256:
  `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`.
- V9 activated adapter weights SHA-256:
  `f1a4a1bc15e3942ed5a3fbe2c4750f39a8b24804e20566242ec740c5ce851c21`.
- BFCL-live records SHA-256:
  `4d70a040341480ccbae9af5a00d83bc4e7365ebead67e390cb5c3763ed83b30a`.
- BFCL-live manifest SHA-256:
  `f1815fa298e8b4e205b061a26ed8a1043e7aa688f7a4b40d034e7c329d14507a`.
- Cases: 25 simple, 25 multiple, 25 irrelevant; query overlap with the static evaluation and
  complete training sources is zero per the pinned V18 manifest. V18 already exposed scores from
  some of these cases, so this is development screening, not an untouched independent test.
- Java 25 `PureJavaBackend` loads the exact GGUF and adapter directly. Qwen's tool prompt is
  rendered in-process from each case's messages and declared tool schemas after the existing
  control-marker injection check. Each turn forces and verifies one physical shared KV prefix.

The V20 live-margin drift moved `multiple_53` from a prior specialist margin `4.8653` to
`4.7051`, below V20's frozen `4.8179874` threshold. V21 sets the primary specialist threshold
to `4.5`, retaining the same rescue conditions: specialist margin strictly above `1.3521204`,
base-minus-specialist margin strictly above `33.705593`, and base margin strictly below
`39.42147`. Contextual call/no-call token IDs remain `4913` and `19536`. The V21 threshold was
selected from exposed V20 data **before** any V21 score; applying it retrospectively to V20's
recorded margins yields 48/50 calls and 24/25 no-calls. That is not a V21 result or qualification.

## Fixed stop/go gate

The V21 decision screen passes only with all of:

- exactly 75 cases with the specified 50/25 class split;
- at least 48/50 correct applicable-call decisions;
- at least 24/25 correct irrelevant/no-call decisions; and
- 75/75 physical shared-prefix identity.

No threshold, rescue condition, corpus, class floor, or source identity may change after V21
scoring begins. If a floor becomes mathematically unreachable, the run may stop early and is
rejected. A pass would allow a separately frozen, stricter-schema generation experiment; it would
not open the sealed 300-case set without that subsequent generation and product gate.

The initial host is the local Intel i7-9750H Mac with 32 GB memory and Java 25.0.3. It incurs no
cloud charge. No external inference server is permitted. A Rust kernel is not used; future Rust
FFM work would require a measured narrow JVM bottleneck and retain Java as reference/fallback.
