# MS MARCO MiniLM L12 v2 reranker evaluation: 2026-09-06

This experiment asks whether the corrected L12 cross-encoder adds enough retrieval quality to earn
a second catalog slot beside the qualified 19 MiB MiniLM L6 reranker.

## Pinned inputs

| Input | Pin |
|---|---|
| Upstream checkpoint | `cross-encoder/ms-marco-MiniLM-L12-v2@7b0235231ca2674cb8ca8f022859a6eba2b1c968` |
| Corrected Q4_K artifact | `cstr/ms-marco-MiniLM-L-12-v2-GGUF@2b6301cfa22006cb9eb5563c2e673e7f3976f00e` |
| Artifact SHA-256 | `064ed6e944326cb2af6c211f48984dcfb41a09fcb27544327def1e2022678f51` |
| Artifact size | 26,083,360 bytes |
| Workload | `../workloads/berlin-population-six-documents-v1.json` |

The retained `transformers-reference.json` was produced by the pinned Transformers oracle. A
separate CrispEmbed build at `e4cfd3dd5353c4a4e9d48c3e6657491d38e797d3` supplied a same-artifact
Q4_K reference. Models does not load either external runtime.

## Correctness

The Java path reproduced the exact pair tokens and token-type IDs, the full ranking
`[0, 2, 5, 4, 1, 3]`, and the same-artifact scores within `0.05`. Its maximum difference from the
unquantized Transformers logits was `0.3272`, inside the explicit `0.40` Q4_K envelope.

The repo's Q8_0 artifact was rejected before Models qualification. It carries a one-layer
classifier head and produces compressed scores around `-0.35..-0.04`; the corrected checkpoint
requires the two-layer classifier head and produces logits around `-11..+9`. Higher nominal tensor
precision cannot repair an incomplete graph.

## Local runtime comparison

Both arms used the same six documents, Java 25.0.3, the same 2019 Intel Mac, two warmups, 20
single-pair iterations, and eight six-document batch iterations.

| Model | Size | Pair p50 | Batch p50 | Documents/s |
|---|---:|---:|---:|---:|
| qualified MiniLM L6 | 19.1 MiB | 109.910 ms | 807.589 ms | 7.430 |
| candidate MiniLM L12 | 24.9 MiB | 239.078 ms | 1,572.799 ms | 3.815 |

The upstream evaluation table reports L6 at `74.30` NDCG@10 / `39.01` MRR@10 and L12 at
`74.31` / `39.02`. The candidate is therefore approximately twice as slow in this Java workload for
an upstream quality difference of only `0.01` on each metric.

## Decision

`PASS` for pure-Java graph compatibility. `REJECT` for ModelJars catalog promotion: it does not
create a meaningful quality tier and substantially weakens throughput. The pinned test remains an
explicit, slow compatibility gate; normal integration builds do not download it.

The next useful reranker tier should change the product envelope: either a materially faster tiny
English model or a genuinely multilingual model. It should not be another near-duplicate point on
the MiniLM L6 curve.

