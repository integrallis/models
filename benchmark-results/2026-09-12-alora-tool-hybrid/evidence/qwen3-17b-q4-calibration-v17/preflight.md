# Qwen3 1.7B production-Q4 calibration V17 preflight

Status: **frozen at 2026-09-14T02:52:27Z, after V16 stopped and before any additional production
Q4 score was collected**.

V16's affine applicability head passed untouched BF16 validation but failed its fixed-zero Java Q4
transfer gate: the first two no-call observations were false positives. V17 does not reinterpret or
repair V16. It is a new experiment whose sole causal change is calibration of the decision boundary
on a production-Java Q4 calibration partition that was deterministically defined by V15 before V16
existed. The affine head, adapter, model, representation, prompts, physical-sharing implementation,
generation policy, and all independent release gates remain unchanged.

Three V16 calibration observations are already known and are disclosed in `../qwen3-17b-hidden-
applicability-v16/result.md`. No additional V17 calibration score and no screen score was observed
before this protocol was frozen.

## Immutable inputs

- Base model, Hugging Face revision, GGUF, adapter, V9 training data, V16 training/validation
  features, and V16 affine head are byte-for-byte identical to V16.
- Production GGUF SHA-256:
  `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`.
- Activated adapter SHA-256:
  `f1a4a1bc15e3942ed5a3fbe2c4750f39a8b24804e20566242ec740c5ce851c21`.
- V16 head file SHA-256:
  `67b1f5fd231f924554d02daf450ce049fbccd24c084f7730d4ce2724ca234d1e`.
- V16 canonical head artifact SHA-256:
  `acece5d8e0188b874ef8a919597ba7a359d876bc20fd828af5fbfab00984445a`.
- Exposed 75-case records SHA-256:
  `944d137ba1e325e0f3daa922a73108de0da5ccf0292bcf4dbe4a565aa11fcb4c`.
- Partition seed: `qwen3-17b-shared-decision-v15:partition-v1`.
- Per kind (`simple`, `multiple`, `irrelevance`), the lowest ten SHA-256 values of
  `seed\0kind\0id` are calibration; the remaining fifteen are the untouched screen. This yields a
  30-case calibration partition (20 call, 10 no-call) and 45-case screen (30 call, 15 no-call).

The sealed V11 300-case qualification window remains prohibited until both V17 phases and the
exposed dual-generation gate pass.

## Phase 1: production-Q4 calibration

The Java 25 in-process runtime scores only the 30 pre-existing calibration cases. Each score uses
the V16 head over the final normalized activated hidden state at `<tool_call>\n`, from the same
activated session that physically shares immutable KV blocks with the base branch.

Candidate thresholds are `nextDown(minimum score)`, every midpoint between consecutive distinct
scores, and the maximum score. A call requires `score > threshold`. Selection maximizes balanced
accuracy, then correct no-calls, then correct calls, then the higher threshold. No feature, weight,
bias, prompt, case, ordering, or tie-break may change.

Calibration advances only with all of:

- 30/30 observations and physical shared-prefix identity;
- at least 19/20 correct call decisions;
- at least 9/10 correct no-call decisions; and
- balanced accuracy at least 0.925.

The command writes a calibration artifact containing all scores, the selected threshold, identities,
environment, and verdict. Its file SHA-256 is captured and committed before the screen begins.

## Phase 2: untouched production-Q4 screen

The screen command accepts the calibration artifact only with its explicit recorded file SHA-256,
verifies every immutable identity, and uses its threshold without adjustment. It evaluates only the
45 untouched cases. V17 passes only with all of:

- 45/45 screen observations and physical shared-prefix identity;
- at least 28/30 correct call decisions;
- at least 14/15 correct no-call decisions;
- screen balanced accuracy strictly above 0.93; and
- combined calibration plus screen results of at least 47/50 correct calls and 24/25 correct
  no-calls.

Failure rejects V17 immediately. A different threshold, split, head, or quantization is a new
experiment.

## Gates after the screen

A passing screen permits V15's already frozen oracle-free dual-branch generation policy on the 75
exposed cases. It still requires 75/75 parseable syntax, 75/75 schema validity, at least 47/50 exact
positive calls, and at most 1/25 false calls, with base-only and adapter-only comparison arms from
the same bytes and physical shared-prefix identity on every hybrid case.

Only then may the original sealed 300-case qualification run. Its unchanged gate is 300/300 syntax,
300/300 schema validity, at least 170/200 exact positive calls and no worse than the same-prompt
base, and at most 5/100 false calls. The 14-case product suite, zipcode regression, six-turn
conversation, natural-language tool-result synthesis, exact base behavior, disabled-adapter no-op,
projection oracle, Spring AI, LangChain4j, long-context, physical identity, memory, native-memory
tracking, performance/crossover, packaging, clean-host, and published-artifact tests remain
mandatory.

Models and ModelJars publication remains prohibited until every gate passes.
