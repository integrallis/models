# Qwen3 cross-model KV qualification evidence

This directory records every positive and negative result for the Qwen3 0.6B to 1.7B cache
handoff. A result belongs here even when it disproves an implementation path.

## Raw-copy negative control

The first experiment copied the complete 28-layer, 8-head, 128-dimension source prefix directly
into the equal-shaped target cache and recomputed the last prompt token with the target. It failed:
the first greedy token differed and only one of eight teacher-forced top-1 predictions agreed.

This rules out raw tensor sharing between these independently trained family members. Any valid
handoff needs a learned transformation, strict model/tokenizer/artifact identity checks, a
recomputed target tail, and downstream correctness gates. See
`raw-copy-negative-control.json` for the exact artifacts, host, and measurements.

The experiment is implemented by `QwenKvHandoffExperiment` and its copy path has unit coverage in
`QwenKvHandoffExperimentTest`.

## Learned-mapper experiments

The calibration corpus is 600 pinned FineWeb-Edu sequences: 500 for fitting, 50 for mapper
validation, and 50 held back for the final token-level test. Both Qwen checkpoints use the same
tokenizer files. Keys are transformed into content coordinates before fitting and returned to the
target's rotary coordinates at handoff.

The first learned implementation was a head-local ablation: each target KV head could see selected
source layers, but only the corresponding source head. That restriction is not present in the
stronger reference implementation and is therefore retained only as a negative ablation. Across
the 50 held-out sequences, its best `k=8` result reached 60% first-token agreement, 68.25%
teacher-forced top-1 agreement, mean final-token KL 0.3731, and mean final-logit cosine 0.9395.

The corrected full-cross-head `k=8` mapper lets every target KV head use all source KV heads from
each selected layer. Its BF16 artifact is 939,640,032 bytes with SHA-256
`7bc77eb86f61a39ac7c0a521367a5d877d7d498392e241865ac62d80e148c18b`. Held-out mapper R2 is
0.78143 for keys and 0.65404 for values. On the 50-sequence token-level test it reached 70%
first-token agreement, 76.25% teacher-forced top-1 agreement, mean final-token KL 0.23355, and
mean final-logit cosine 0.96001.

Those improved token metrics did **not** qualify the mapper. The first pinned downstream smoke
retained the native target's normalized HellaSwag score on 20 examples, but its mapped/native
WikiText perplexity ratio was 1.71835, failing the predeclared 1.10 maximum. The immutable local
result is `full-cross-head-k8-quality-smoke.json` (SHA-256
`913b9376943238ab2bcf81d15a7b0b4ba67f5ca75790d5f251cf5929812ba848`). This candidate is rejected
as configured; no catalog or runtime claim may be based on its token-level metrics.

The follow-up corrected both departures from the reference protocol. Source-layer probes used all
500 calibration sequences, and the full-cross-head mapper was increased to `k=12`. The resulting
BF16 artifact is 1,409,402,536 bytes with SHA-256
`37a231af96485682a02f866298777e420addb04ab8e90eee398dd5e673c6965b`. Held-out mapper R2 improved
only modestly to 0.78536 for keys and 0.65953 for values. The independent 50-sequence token test
remained at 70% first-token and 76.25% teacher-forced top-1 agreement; mean final-token KL improved
to 0.22683 and mean final-logit cosine to 0.96618.

At a 1K context, no practical recomputed suffix met the 1.10 WikiText perplexity-ratio gate. The
first passing setting recomputed 768 of 1,024 target tokens, which erases the intended handoff
advantage. A bounded-memory evaluator then made 4K contexts possible without materializing both
full caches in FP32. At 4K, recomputing a 4-token sink and a 512-token suffix passed the perplexity
gate on 32 held-out samples with a mapped/native ratio of 1.02085.

Passing perplexity was still insufficient. The pinned CacheBridge exact key/value retrieval task
was run at 4K, where all eight native-target answers were correct. The mapped cache retained only
five of eight answers with a 512-token suffix. Increasing the target-native suffix to 768, 1,024,
and 1,536 tokens retained only four of eight in each run: answers stored in the translated middle
of the context were lost while answers in the native sink or suffix survived. This decisively
fails the predeclared 100% native-correct retention gate. The natural-language needle task retained
eight of eight native-correct answers, but one passing task cannot override the failed exact gate.

## Attention-weighting protocol correction

An audit against CacheBridge v1 (arXiv:2609.00891, equations 15–19) invalidated the first
attention-weighted calibration exports. They measured ordinary same-token attention and summed
raw sensitivity across boundaries. The paper instead measures the target model's first future
query at each causal prefix boundary, excludes the query and all later positions, normalizes K and
V sensitivity separately to unit mass for every target layer, KV head, component, and boundary,
and then gives every boundary equal mass.

Consequently, `attention-smoke`, `attention-n50`, and `attention-n500` on the qualification host,
and every `mapper-attention-*` or `mapper-full-attention-*` artifact derived from them, are invalid
evidence and cannot support any conclusion about Attn-Repair. They are retained only as provenance
of the failed experiment. The first attempted corrected export also stopped before finalization on
a GPU-to-host serialization defect; its `.partial` output was never promoted to an artifact.

The corrected exporter is schema version 2 and has synthetic coverage for the sensitivity
formula, causal boundary mask, per-boundary normalization, GQA head grouping, and GPU-to-host
materialization. The fitter refuses older schemas and binds the sensitivity file to its SHA-256,
target checkpoint revision, calibration-corpus hash, exact sequence and sampled-position order,
tensor geometry, paper version, and aggregation rules. Derived shrink weights live inside the
mapper's evidence directory rather than modifying the sensitivity export. Its layer-selection path
also follows the paper's calibration-only centered same-head OLS probes and held-in R2 ranking; a
held-out ridge selector is treated as a separate variant rather than mislabeled as CacheBridge.

The corrected 500-sequence export contains 128,000 observations and has SHA-256
`c8715a63f761d24f235abed9c74a8182aea9e3c3ad549d330f0638037edf2d36`. The resulting head-local
`k=8` mapper is 117,556,336 bytes with SHA-256
`cab1cc1e71025f0a4586bfe82e4d63833af1f51d3deaa05d2b42db24558ae3bb`. On the disjoint
50-sequence token test it reached 62% first-token and 69.5% teacher-forced top-1 agreement, with
mean final-token KL 0.34245 and mean final-logit cosine 0.93168.

The exact downstream gate rejected it. At 4K with a four-token native sink and 512-token native
tail, the target answered all eight exact key/value probes correctly but the mapped cache retained
only two. The `k=8` mapper is rejected as configured and cannot support runtime or catalog work.
The first `k=12` fit then exhausted the 8 GiB GPU because the generic solver allocated a second
1.97 GiB centered covariance tensor beside its existing sufficient statistics. It produced no
mapper. The corrected bounded solver centers the owned moments in place and solves one
layer/head equation at a time; a synthetic test proves numerical equivalence and verifies that
the storage is reused.

The separately predeclared `k=12` capacity point then completed under the new evidence identity.
Its BF16 mapper is 176,277,040 bytes with SHA-256
`ce5259787c339829378b477ae9818ef9cae01d56fbbf7ccffce3c960c8d99e9a`; held-out mapper R2 was
0.72982 for keys and 0.57564 for values. On the independent 50-sequence token diagnostic it reached
60% first-token and 70% teacher-forced top-1 agreement, with mean final-token KL 0.30923 and mean
final-logit cosine 0.93853. More fitting capacity did not repair downstream behavior. At the same
4K exact-retrieval gate, the native target again answered all eight probes correctly and the mapped
cache retained only two. The `k=12` CacheBridge mapper is therefore rejected. Its fit report,
mapper, token diagnostic, exact-retrieval evaluation, and logs are preserved under
`remote-evidence/cachebridge-v4-k12/`.

## Current decision

The Qwen3 0.6B to 1.7B raw-copy, predictive ridge, and corrected CacheBridge mapper families are
**rejected** as releasable cross-model KV handoffs. They are not wired into production, listed in
the catalog, or eligible for a capability claim. The negative result establishes that
shape-compatible Qwen3 caches plus strong token-level similarity do not guarantee exact
long-context behavior.

The next mechanism must use a truly shared base with adapter-compatible activations so that cached
prefix blocks are exact, not translated approximations. Qualification thresholds will not be
relaxed to manufacture a positive result.

## Evidence notes

The immutable artifacts copied from the controlled host are under `remote-evidence/`; their hashes
are recorded in `remote-evidence/SHA256SUMS`. Rejected mapper binaries are ignored by Git because
of their size, but are retained locally until the experiment archive is finalized.

The bounded-memory mapper translates token chunks directly into BF16 target-cache blocks. It is
mathematically equivalent to the whole-cache transform, but GPU GEMM accumulation changed slightly
with batch shape: on a 1K cross-check the whole-cache and chunked perplexity ratios were 1.2588 and
1.2661 respectively, with identical native loss. These oracle runs are therefore screening
evidence, not a substitute for the final Java/GGUF implementation and published-artifact tests.
