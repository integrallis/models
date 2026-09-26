# SemIf-faithful arm — retained data, no verdict yet

Committed 2026-09-26 from untracked workstation files produced 2026-09-22. **Provenance: measured
here** on the decisions-GPU host at the commits recorded in this directory's other notes. Nothing
was re-run to write this file, and it deliberately draws no accuracy conclusion — see "What is
missing" below.

## What the arm is for

`prepare_semif.py` renders JevBench rows exactly as the upstream `jevbench/adapters/semif_direct.py`
does, so our arm differs from SemIf's **only in the runtime**: our Q4_K_M Java engine against their
BF16 PyTorch. Per the working agreement, an external comparator is only meaningful when we run every
arm ourselves under one protocol; this script exists so that condition can be met rather than
assumed.

The script's own docstring records four fidelity corrections over a first attempt that was not
faithful — option-description prefixes, `noul` option order and its true→yes/false→no mapping,
choice options following criteria key order rather than the labels array, and evidence carrying
`task.state` as a nested JSON value instead of a pre-stringified copy. The chat wrapper is the
model's own `tokenizer.chat_template` rendered under `add_generation_prompt=True`,
`enable_thinking=False`.

## Files

| file | rows | contents |
|---|---|---|
| `prepare_semif.py` | — | the renderer, faithful to the upstream adapter |
| `semif-flat.tsv` | 231 | rendered prompts, one per JevBench item, with kind and option ids |
| `tokens-semif.tsv` | 232 | per-item letter-token and candidate-token counts (header row included) |
| `q8-semif.tsv` | — | Q8 activations, SemIf-faithful rendering |
| `qwen4b-semif.tsv` | — | Qwen 4B, SemIf-faithful rendering |
| `q8-ours.tsv` | — | Q8 activations, our own rendering — the control for the rendering change |
| `semif-std-out.tsv` | 72 | `noul` distributions, options in `no,yes` order |
| `semif-std2-out.tsv` | 72 | the same items with options in `yes,no` order |
| `semif12out.tsv` | 12 | a 12-item choice subset |

`semif-std-out.tsv` and `semif-std2-out.tsv` are an **option-order ablation**, not a duplicate run:
the same items scored with the two option orders, which is how a positional artefact in the letter
distribution would show up. Keep them paired.

## What is missing before any claim

1. **No scored verdict.** These files are distributions and token counts. Nothing here computes an
   accuracy for the SemIf-faithful arm, and the sibling `SCORES.md` predates this data and does not
   cover it.
2. **One dataset.** JevBench alone cannot settle a general claim under the working agreement.
3. **No upstream arm run by us.** The point of the fidelity work is to compare against SemIf's own
   runtime; that side has not been run here, so the comparison this data was built to enable is
   still not available. A published SemIf number would be `read`, and must not be placed beside
   these as though both were measured here.
4. **The option-order ablation is unanalysed.** The pair exists; no positional-bias result has been
   derived from it.

Until at least 1 and 3 are done this directory holds a faithfully prepared instrument and its raw
output, which is worth keeping and is not a result.
