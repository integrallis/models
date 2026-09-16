# Qwen3 1.7B physically shared hybrid-generation V20 preflight

Status: **frozen before any V20 model output is generated**.

V20 is the last development gate before the sealed 300-case qualification window may be opened.
It runs entirely inside Java 25 and tests the complete composition: one exact Qwen3 base model, one
activated tool specialist, a fixed two-branch decision policy, real structured generation, and one
physical KV prefix shared by both branches. The 75 inputs were exposed by earlier experiments, so a
pass permits independent qualification but is not itself release evidence.

## Immutable implementation and inputs

- Models scorer revision: `beea98aa922fd45915883213e84a10911e2e98e4`.
- Base model: `Qwen/Qwen3-1.7B`, Q4_K_M GGUF SHA-256
  `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`.
- Activated V9 adapter SHA-256:
  `f1a4a1bc15e3942ed5a3fbe2c4750f39a8b24804e20566242ec740c5ce851c21`.
- Exposed 75-case records SHA-256:
  `944d137ba1e325e0f3daa922a73108de0da5ccf0292bcf4dbe4a565aa11fcb4c`.
- Records: 25 simple calls, 25 multiple calls, and 25 irrelevant/no-call prompts.
- Contextual call/no-call token IDs: `4913` / `19536`.
- Generation: temperature `0`, maximum `128` new tokens, unrestricted token constraint.
- Prompts are consumed from the pinned record bytes after recursively rejecting tokenizer and tool
  control markers from every untrusted message and tool value. Reconstructing them is prohibited.

The Java evaluator is a test-covered port of the pinned BFCL matching rules. It requires complete
`<tool_call>` blocks with no surrounding prose, declared tools and arguments, schema-valid values,
and unordered exact matching against the recorded accepted alternatives. No Python or external
runtime participates in scoring.

## Fixed branch decision

At the same contextual boundary, the runtime measures `callLogit - noCallLogit` on both the exact
base branch and activated specialist branch. It selects a tool call when either:

1. the specialist margin is strictly greater than `4.8179874`; or
2. the specialist margin is strictly greater than `1.3521204`, the base-minus-specialist margin is
   strictly greater than `33.705593`, and the base margin is strictly less than `39.42147`.

All other prompts stay on the exact base branch. These thresholds were derived only from previously
exposed development margins and cannot change after this preflight. Every turn forces the shared
strategy and must prove that the two sessions reference the same physical KV prefix.

## Frozen development gates

The complete 75-case run passes only with all of:

- at least 48/50 correct call decisions;
- at least 24/25 correct no-call decisions;
- at least 43/50 exact generated tool calls;
- 75/75 strict-syntax results;
- 75/75 schema-valid results; and
- 75/75 physically shared prefixes.

A selected one-case run is a mechanics diagnostic only and cannot return a qualification pass. The
complete run must finish and write its report. A failed or partial run cannot authorize opening the
sealed set.

## Release boundary

The V11 300-case window remains sealed and unread. It may be opened exactly once only after V20
passes unchanged. The independent window must then satisfy its frozen correctness gates plus the
existing product, conversation, framework-adapter, exact-base, memory, native-memory-tracking, and
performance gates before Models or ModelJars can publish a hybrid artifact.

Production inference is in-process and Java-first. External inference servers are prohibited. A
future Rust/FFM kernel would be permitted only for a narrow, profiled JVM primitive; it may not own
the model graph, tokenizer, sampling, state, or generation, and Java must remain the reference and
fallback. No Rust kernel participates in V20.
