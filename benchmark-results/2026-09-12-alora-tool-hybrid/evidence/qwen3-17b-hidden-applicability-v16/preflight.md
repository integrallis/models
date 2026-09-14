# Qwen3 1.7B hidden-applicability V16 preflight

Status: **frozen at 2026-09-14T00:54:17Z, before feature extraction, head training, or any V16
candidate output**.

V15 proved that a base Qwen3 1.7B branch and its V9 Activated-LoRA tool specialist can retain the
same immutable physical KV-cache blocks. Its two-vocabulary-logit applicability rule nevertheless
missed the unchanged development floor. V16 preserves the exact base, adapter, prompts, shared
cache, generation policy, and release gates. Its sole causal change is a small affine applicability
head over the activated branch's final normalized hidden state at the same causal decision
boundary.

The head is evaluated in Java in the same process as inference. Python, Transformers, PEFT, and a
GPU are permitted only to extract training features and fit the offline artifact; they are not a
production inference dependency.

## Immutable ancestry and data

- Base: `Qwen/Qwen3-1.7B` revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- Production GGUF SHA-256:
  `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`.
- V9 rank-32 adapter SHA-256:
  `f1a4a1bc15e3942ed5a3fbe2c4750f39a8b24804e20566242ec740c5ce851c21`.
- V9 training manifest SHA-256:
  `d0620df4860c879f6d3f6e5573168bb08afc0b48d954040825e1317972f7de47`.
- Prepared-data manifest SHA-256:
  `6abf29b6040a2831f52ad0b77a7276a2213e5c6bc775c4c68b40e3a3dc9b788c`.
- Train JSONL SHA-256:
  `c46b606c74a3d1bda38c6f50a55854bf9112f0ae9d92993eb32e577c211f3002`;
  2,840 usable rows after the frozen 1,024-token filter: 2,356 call and 484 no-call; ordered
  source-line SHA-256
  `ceab03a876d41ada5e7a3f99cad4357e0b70a478b6f8c20f82da7aa0a1626941`.
- Validation JSONL SHA-256:
  `7006d89f1be432424889f8fbc846ff348cf2770a56f409c9e38361ae4857cfb0`;
  471 usable rows: 390 call and 81 no-call; ordered source-line SHA-256
  `1ce289514ff5ddbe72a34e240196925b8d341058e66278d627b477f5fc622f51`.
- Already-exposed 75-case development records SHA-256:
  `944d137ba1e325e0f3daa922a73108de0da5ccf0292bcf4dbe4a565aa11fcb4c`.

No V11 sealed qualification row may be opened until every training, validation, Java-development,
and generation prerequisite below passes.

## Fixed representation

For each row, use the exact V13-compatible tokenization and canonical Qwen tool prompt. Append the
shared output prefix `<tool_call>\n`, whose frozen token IDs are `[151657, 198]`. Do not append the
first call/no-call token. The feature is the 2,048-coordinate final RMS-normalized activated hidden
state at the last shared-prefix position—the same state that the vocabulary projection consumes to
produce V15's call/no-call logits.

The base and V9 adapter remain frozen. Dropout is disabled. Feature extraction is batch size one,
BF16, SDPA, TF32 disabled, using Torch `2.6.0+cu124`, Transformers `4.53.3`, and the already proven
PEFT environment. Inputs longer than 1,024 tokens retain the historical V13 exclusion; no new
filtering is allowed. Features are persisted as ordered float32 arrays with row IDs, labels,
source-line identities, extraction-environment metadata, and SHA-256 values.

The production Java path uses the normalized activated hidden state from the quantized GGUF at the
same token boundary and from the same request-scoped session used for later generation. It may not
recompute or copy the common prefix. Every development and qualification observation must report
physical base/tool prefix identity.

## Frozen affine-head training

The classifier score is:

```text
score(h) = dot((h - mean) / scale, weight) + bias
call iff score(h) > 0
```

`mean` and `scale` are computed only from the applicable training fold. `scale` is the population
standard deviation clamped to at least `1e-6`. Training uses float64 full-batch, class-balanced
binary cross-entropy plus `lambda * sum(weight^2)`. The bias is not regularized. Each optimizer
starts from all-zero weight and bias and uses deterministic CPU L-BFGS with strong-Wolfe line
search, history size 20, at most 200 iterations, gradient tolerance `1e-10`, and parameter/change
tolerance `1e-12`.

Select `lambda` only from `[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]` by deterministic five-fold
out-of-fold performance on the training split. Within each label, sort rows by SHA-256 of
`qwen3-17b-hidden-applicability-v16:fold-v1\0<source-line>` and assign folds round-robin. Score at
the fixed zero threshold. Selection maximizes pooled balanced accuracy, then correct no-calls,
then correct calls, then the larger `lambda`. Refit once on all training rows with the selected
`lambda`, all-training normalization, and the same optimizer. No checkpoint selection, feature
selection, nonlinear transform, prompt change, threshold calibration, or post-output hyperparameter
change is allowed.

The saved artifact contains schema/version identity, all immutable input hashes, extraction and
training provenance, hidden dimension, normalization vectors, affine parameters, fixed threshold
zero, selected `lambda`, cross-validation counts, validation counts, and a canonical artifact
SHA-256. Save/reload scores must be bit-identical in float64. Java evaluates the persisted float32
parameters with a documented accumulation order and must reproduce serialized arithmetic fixtures
within `1e-6` absolute score error and with identical decisions.

## Frozen gates

The offline head advances only when the untouched validation split has all of:

- at least 371/390 correct call decisions;
- at least 77/81 correct no-call decisions;
- balanced accuracy strictly above 0.95;
- finite features, normalization, parameters, and scores; and
- exact row counts, order, labels, source identities, and source hashes.

A passing artifact then runs the 75 already-exposed Java development cases. Before generation it
must produce at least 47/50 correct call decisions, at least 24/25 correct no-call decisions, and
75/75 physically shared prefixes. Failure rejects V16 without dual generation or sealed data.

If the decision gate passes, use V15's already frozen oracle-free dual-branch arbitration unchanged:
accept only parseable/schema-valid declared tools; take the sole usable branch; when both branches
select the same ordered tools, prefer more unique declared top-level arguments; otherwise prefer
the activated specialist. The Java development result must have 75/75 parseable syntax, 75/75
schema validity, at least 47/50 exact positive calls, and no more than 1/25 false calls. Base-only
and adapter-only arms must come from the same model and adapter bytes.

Only that complete pass may open the existing sealed 300-case window. Its original gate remains
unchanged: 300/300 syntax, 300/300 schema, at least 170/200 exact positive calls and no worse than
the same-prompt base, and at most 5/100 false calls. The unchanged 14-case product suite, zipcode
regression, six-turn conversation, natural-language tool-result synthesis, exact base behavior,
disabled-adapter no-op, projection oracle, Spring AI, LangChain4j, long-context, physical-identity,
memory, native-memory tracking, and shared-versus-recomputed performance/crossover gates all remain
mandatory afterward.

Models and ModelJars publication is prohibited until every gate passes. Any failed correctness
gate rejects V16. A later nonlinear head, different representation, or different threshold would
be a separately preflighted experiment, never a reinterpretation of V16.
