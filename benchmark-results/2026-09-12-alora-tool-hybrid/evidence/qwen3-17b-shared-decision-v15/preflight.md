# Qwen3 1.7B shared-decision V15 preflight

V15 tests a runtime composition over the unchanged V9 Qwen3 1.7B adapter. It does not retrain,
select a checkpoint, translate KV values, or copy equivalent caches. The base and activated branch
must reference the same immutable physical KV blocks before the adapter invocation.

## Immutable inputs

- Base: `Qwen/Qwen3-1.7B` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- GGUF SHA-256: `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`.
- V9 adapter SHA-256: `f1a4a1bc15e3942ed5a3fbe2c4750f39a8b24804e20566242ec740c5ce851c21`.
- Activated invocation: `[151644, 77091, 198]`.
- Contextual decision prefix: `<tool_call>\n`.
- Contextual call token: `4913`; contextual no-call token: `19536`.
- Already-exposed V9 development records SHA-256:
  `944d137ba1e325e0f3daa922a73108de0da5ccf0292bcf4dbe4a565aa11fcb4c`.

The records store their auxiliary `tools` tree with sorted JSON keys, while their pinned `prompt`
field preserves the original upstream schema member order used by V9. Reconstructing prompt bytes
from that auxiliary tree is therefore prohibited. The profiler consumes the exact pinned prompt.
Before treating it as template control, it recursively rejects any `<|`, `<tool`, `</tool`,
`<tools`, or `</tools` marker in every untrusted message and tool value. The frozen 75-case source
contains zero such values, so recognizing the template's real special tokens cannot accidentally
promote user or schema text into control tokens.

External Hugging Face/Python execution remains an oracle and training aid only. The scored and
releasable path is the Java 25 in-process runtime over the exact GGUF and adapter bytes.

## Fixed call-decision procedure

The Java runtime evaluates the rendered Qwen tool prompt plus the common `<tool_call>\n` output
prefix on the activated branch and records `callLogit - noCallLogit`. Scoring is performed on the
same activated session used for any subsequent generation. The session retains the exact physical
prefix shared with the base branch; a copied or independently recomputed prefix fails the run.

The 75 already-exposed V9 cases are development data, not independent qualification evidence.
Within each of `simple`, `multiple`, and `irrelevance`, the lowest ten SHA-256 values of
`qwen3-17b-shared-decision-v15:partition-v1\0<kind>\0<id>` form the 30-case calibration partition.
The remaining 45 cases form a margin screen whose logits cannot influence threshold selection.

Threshold candidates are the value immediately below the smallest distinct calibration margin,
every midpoint between adjacent distinct margins, and the largest margin. The selected candidate
maximizes balanced call/no-call accuracy, then correct no-calls, then correct calls, then the higher
threshold. A call requires a margin strictly greater than the threshold. The screen passes only
with all of:

- at least 28 of 30 correct call decisions;
- at least 14 of 15 correct no-call decisions; and
- balanced accuracy strictly above 0.93.

No threshold, tie-break, partition, token ID, or screen gate may change after margins are read.

## Fixed base/adapter arbitration

V9's already-exposed generations are permitted for policy design, but cannot qualify V15. They
show the useful complementarity that motivates the composition: the activated branch supplies
strict tool syntax when the base emits prose, while the base supplied three correct optional
arguments that the activated branch omitted. The fixed oracle-free policy is:

1. When the decision margin does not pass, return the canonical empty tool-call sentinel and use
   the exact base branch for conversational generation.
2. When it passes, generate a candidate on both physically shared branches.
3. A candidate is usable only when it parses as tool calls and names only declared tools. Prefer
   the sole usable candidate when the other is unusable.
4. When both are usable and have the same ordered tool names, prefer the candidate containing more
   unique, schema-declared top-level arguments. This favors explicit user-supplied optional values,
   but cannot reward invented or undeclared fields.
5. All remaining cases, including exact ties or different tool selections, use the activated tool
   specialist.

Applied retrospectively to the exposed records, this rule selects 47/50 exact positive calls versus
45/50 for base and 42/50 for the adapter. That is development evidence only. The Java dual-branch
implementation must reproduce it before the sealed set is opened.

## Qualification and release gates

If the margin screen passes, the fixed Java hybrid must reproduce all 75 development cases with
100% parseable syntax, 100% schema validity, at least 47/50 positive exact calls, and no more than
1/25 false calls after the frozen decision threshold. It must also prove the base-only and
adapter-only comparison arms from the same model bytes.

Only then may the existing sealed 300-case window be opened. It retains the original independent
gate: 100% syntax, 100% schema validity, at least 85% exact positive calls and no worse than the
same-prompt base, and at most 5% false calls. The unchanged 14-case product suite, zipcode
regression, six-turn conversation, natural-language tool-result synthesis, exact base behavior,
disabled-adapter no-op, projection oracle, Spring AI, LangChain4j, long-context, physical identity,
memory, NMT, and performance/crossover gates must all pass afterward.

Models and ModelJars releases remain prohibited until every gate passes. If any correctness gate
fails, V15 is rejected and no downstream artifact or catalog entry is published.
