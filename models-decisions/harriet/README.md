# Harriet

**Typed, calibrated decisions from open weights. One forward pass, no generated tokens, on the JVM.**

Harriet answers a *typed decision* — evidence, a criterion, and a closed set of options — with a
probability distribution over exactly those options. It never generates text, so there is no JSON
to repair, no parse to fail, and no decoding loop to time out.

It trains nothing. The weights are **Qwen3.5-4B, frozen, Apache-2.0**. Harriet is a prompt, a
readout, a calibration, and a release format — all of it Java on the Panama Vector API with small
Rust shims for the K-quant kernels.

* [`METHOD.md`](METHOD.md) — what it does and how the readout works
* [`ATTRIBUTION.md`](ATTRIBUTION.md) — whose weights, whose method, whose benchmark
* [`REPRODUCE.md`](REPRODUCE.md) — every command behind every number here

## Results

<!--RESULTS-->

## What we refuse to do

**No temperature fitted on the benchmark.** We measured what that trick is worth: fitting one on
half of JevBench's hard tier and reporting on the other half moved a Qwen3-0.6B arm from 58.1 to
68.6 composite, which on a field of 24 is roughly 18th place to 5th. It is disclosed tuning on the
evaluation set. Harriet's headline runs with no fitted temperature.

**No training on another system's output.** Nothing here is derived from Jev or any closed
service — not for training, distillation, prompt search, or model selection.

**No borrowed numbers.** Every figure above was produced on our own harness on a dedicated host.
Where we quote another system we say so and link the source.

## Honest limits

* The **judge tier** carries 28% of the Intelligence weight and is not in the public release, so
  the harness renormalises it away. Checked against the 25 published systems that ran both tiers:
  judge sits 0.033 below standard on average, but for systems below 0.70 standard it sits a median
  0.037 *above*. Our number is therefore mildly pessimistic, not optimistic.
* **Calibration is half-measured.** The axis is the mean of an ECE term and a TVD-to-gold-distribution
  term. The public sets carry no gold distributions, so only the ECE half is computed.
* **Speed is standard-tier only.** The spec draws p50/p95 from a standard+judge run.
* **Cost uses a size-class reference price**, as every self-hosted row in the published results
  does. Input tokens are counted with the model's own tokenizer, not estimated.
* This is **not comparable to a published leaderboard row** for the reasons above. It is a
  like-for-like recomputation under one protocol, with the protocol stated.

## Licence

Apache-2.0 for this code. Qwen3.5-4B weights are Apache-2.0 and are not redistributed here.
Independent project; not affiliated with or endorsed by TypeSafe, Qwen, IBM, or SemIf.

## Name

Harriet Ann Jevons, née Taylor, married William Stanley Jevons in 1867. He built the Logic Piano,
the first machine to solve logical inference mechanically. After he drowned in 1882 she selected
and edited *Letters & Journal of W. Stanley Jevons* (Macmillan, 1886) — his work reached the
public because she published it. An open-weights release is named for the publisher.
