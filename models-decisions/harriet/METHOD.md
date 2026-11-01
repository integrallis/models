# What Harriet is

A typed decision is a whole question — evidence, a criterion, and a closed set of options — that
must come back as one of those options with a probability attached. Harriet answers it with **one
forward pass and no generated tokens**.

## The readout

The model's own chat template wraps a system instruction and a JSON payload:

```
<|im_start|>system
Apply the supplied criterion to the supplied evidence. Choose exactly one listed option.
Respond with only its uppercase letter, with no explanation or reasoning.<|im_end|>
<|im_start|>user
{"evidence": ..., "criterion": "...", "options": [{"letter": "A", "description": "id: text"}, ...]}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

One forward pass. Read the last-position logits at the token ids for `A`, `B`, ... and softmax
over just those. The answer is a distribution over the declared options and nothing else: no JSON
to repair, no parse to fail, no decoding loop to time out.

Every answer slot is verified to be exactly one round-trip token before the run, and a slot that
is not is refused rather than scored. A merged or multi-token slot reads the wrong logit and
would fail silently.

## Three types

* **Noul** — a proposition, true or false.
* **Choice** — 2 to 255 named options.
* **Score** — 2 to 10 ordered levels.

All three reduce to the same letter readout, so there is one code path and one failure mode.

## What it does not do

No generation, ever. No fine-tuning, no LoRA, no distillation — the weights are frozen and are
someone else's, under Apache-2.0. Harriet is a prompt, a readout, a calibration, and a release
format.

## Calibration, and what we refuse to do

A decision model that cannot say how sure it is has not finished the job. Harriet ships a
temperature in its artifact and applies it at inference.

**That temperature must never be fitted on the benchmark.** We measured what that is worth:
fitting one on half the JevBench hard tier and reporting on the other half moved a Qwen3-0.6B arm
from 58.1 to 68.6 composite — from roughly 18th to 5th on a field of 24. It is disclosed tuning
on the evaluation set and we do not publish it. Harriet's headline number runs with no fitted
temperature at all, because the 4B is natively calibrated (hard-tier ECE 0.1295).

## The artifact

`DecisionArtifact` is the release unit: standardizer, head, temperature, base model name and
**base weight digest**. Magic `IDSN`, version 2, big-endian, byte-reproducible. Loading it against
weights whose sha256 disagrees throws rather than scoring. That check has already caught a
same-byte-size, different-hash download.

## Engine

Pure Java on the Panama Vector API, with small Rust shims via FFM where Java alone will not do —
the K-quant kernels. No vendor math library, no inference runtime, no external engine. Model
parsing, tokenizer, graph, KV cache, and the scoring loop are all Java.

Sessions share a frozen KV prefix: `freezePrefix` / `fork` / `sharesPrefixStorage`. Many questions
against one document pay for that document once. Measured consequence: candidate scoring costs
only 1.25x a single pass, because the k option forwards reuse the prefix.
