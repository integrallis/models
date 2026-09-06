# MS MARCO TinyBERT L2 v2 reranker qualification: 2026-09-06

TinyBERT L2 is a useful target for a distinct low-latency tier: the upstream checkpoint is only
4.39 million parameters and its published evaluation reports 9,000 documents/s, `69.84` NDCG@10,
and `32.56` MRR@10 on the upstream GPU benchmark.

The official Apache-2.0 checkpoint was pinned at
`cross-encoder/ms-marco-TinyBERT-L2-v2@81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc`. The retained
Transformers oracle records exact pair inputs, logits, and ranking for the common six-document
workload.

Three community GGUF repositories were inspected. The Q4_K_M artifact
`c7666e78709a1ac4c9ed709d9b7ceb3779aedf3b0a38ee19be97ee6acd32cd20` and Q8_0 artifact
`67ba8eb5303ebd8ba398d08656d106e9cdea5ec9e85ed2201459f7c81ed524b8` both load the two-layer,
128-wide BERT encoder but omit the sequence-classification head. An independent GGUF runtime
therefore rejects both as “not a cross-encoder reranker.” Models must not publish an artifact that
silently turns a reranker into an embedding-only encoder.

## Corrected standard GGUF

The official checkpoint does contain the missing tensors: the 128-wide BERT pooler and its scalar
sequence-classification output. A controlled conversion at llama.cpp commit
`a5822222909b785f23ddc74ce3c8f85bd0e38562` retained the pooler as standard `cls.weight` and
`cls.bias`, retained the classifier as `cls.output.weight` and `cls.output.bias`, and declared GGUF
rank pooling. The converter checkout was restored after the experiment; no external inference
engine is part of Models.

| Artifact | Size | SHA-256 | Result |
|---|---:|---|---|
| corrected F32 | 17.0 MiB | `f63a33e59119488ec46e453e05e0962c07687cb208cd5677cf6ad1e64dabd8ef` | conversion control |
| corrected Q4_K_M | 5,351,264 bytes | `6498399a6d6837ce04805aac972caf07aab8cc4db3eedd914fc5392c7505f409` | ranking preserved; maximum Transformers logit delta `0.617` |
| corrected Q8_0 | 5,498,208 bytes | `3c6701e5cd30548a68f5ce0cbbc1ad0833f414021f3a8d5e1436f80738f71ef4` | ranking preserved; maximum Transformers logit delta `0.061` |

Models previously recognized a project-specific corrected naming scheme but not the standard GGUF
rank-pooling contract. The retained regression now accepts both schemes and tests the Q8_0 bytes
through the plain Java, LangChain4j, and Spring AI APIs. The exact Java scores are
`[7.222041, -11.288418, 7.379485, -11.584701, -9.988857, -5.657200]`; both Java and Transformers
rank the documents `[2, 0, 5, 4, 1, 3]`.

## Local runtime comparison

The retained performance result uses the same query and six documents as the L6/L12 comparison.
Each of three fresh Java 25.0.3 processes used two batch warmups, 20 single-pair iterations, and
eight six-document batch iterations on a 2019 Intel Core i7-9750H Mac. The backend is pure Java;
the median pair p50 is `13.778 ms`, the median batch p50 is `67.728 ms`, and median batch throughput
is `88.590 documents/s`. The maximum observed pair p95 was `40.485 ms`; maximum batch p95 was
`141.439 ms`.

Against the existing L6 measurement on the same machine and protocol, TinyBERT is about 8.0 times
faster for a single pair and 11.9 times faster for the six-document batch. This is a real speed
tier, with the expected quality tradeoff: the upstream evaluation reports `69.84` NDCG@10 and
`32.56` MRR@10 versus L6's `74.30` and `39.01`.

Decision: `PASS` for pure-Java graph compatibility, equivalence, framework behavior, and the local
performance envelope. Catalog promotion is `BLOCKED ON DURABLE ARTIFACT URL`: the corrected Q8_0
bytes must be published immutably before ModelJars can bind and independently download them.
