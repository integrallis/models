# Logit fusion in ~50 lines of Java

`LogitFusion.java` fuses two Qwen3 GGUF members token by token with the product-of-experts rule
`s = w·log_softmax(logits_A) + (1 − w)·log_softmax(logits_B)`, greedy, each member keeping its own KV
cache, using only the public Models API (`InferencePipeline.prefill` / `forward`, `ChatTemplate`).
It prints each fused token id, both members' argmax ids and whether the fused token agreed with
either, then the fused text and the members' argmax agreement count.

```bash
jbang LogitFusion.java Qwen3-0.6B-Q8_0.gguf Qwen3-1.7B-Q8_0.gguf "What is 17 + 25?" 48 0.5
```

It is a readable illustration, not the measured runner. The study's runner (parallel member
stepping, mixture and article rules, sampling, voting, reranking, gates and full reports) is
`models-bench logit-fusion`; see
[`benchmark-results/2026-09-17-logit-fusion-study/REPRODUCE.md`](../../benchmark-results/2026-09-17-logit-fusion-study/REPRODUCE.md).

Run gate G0 (tokenizer identity) before fusing two files; this example assumes it passed and
tokenizes with member A only.

## What the public API does not expose (so the runner reaches below it)

- **Per-position logits for a multi-token continuation.** `InferencePipeline.prefill` returns only
  the last position's logits. Reranking and option log-likelihood need every position; the runner
  uses the backend's `SpeculativeInferenceBackend.verify` when available and otherwise teacher-forces
  with one `forward` per token. A public `prefillAllLogits` (or `score(continuation)`) on
  `InferencePipeline` would remove that.
- **Tokenizer identity.** There is no public tokenizer fingerprint. Gate G0 hashes the GGUF
  tokenizer metadata through `backend-java`'s GGUF reader.
