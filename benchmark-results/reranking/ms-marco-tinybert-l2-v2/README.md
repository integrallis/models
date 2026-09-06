# MS MARCO TinyBERT L2 v2 reranker preflight: 2026-09-06

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

Decision: `BLOCKED ON ARTIFACT`, not rejected as a model. Produce or obtain an immutable corrected
GGUF containing the classifier head, then rerun exact token, score, ordering, framework-adapter,
latency, and memory gates. Until there is a durable download URL, it cannot enter ModelJars.

