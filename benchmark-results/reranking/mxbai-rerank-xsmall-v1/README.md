# mxbai-rerank-xsmall-v1 qualification: 2026-09-06

The official, public Apache-2.0 checkpoint was pinned at
`mixedbread-ai/mxbai-rerank-xsmall-v1@b5c6e9da73abc3711f593f705371cdbe9e0fe422`.
It is a 12-layer, 384-wide DeBERTa-v2 cross-encoder with disentangled relative attention and an
F16 Safetensors classification head.

Models implements the complete graph in Java: the Hugging Face Unigram/precompiled-character-map
tokenizer, sentence-pair boundaries, learned relative-position bucketing, content-to-position and
position-to-content attention, ContextPooler, and scalar classifier. No Python, ONNX Runtime, or
external inference service is used at runtime.

## Correctness

The pinned Transformers 4.38.1 oracle supplies exact token IDs, token-type IDs, and logits. The
pure-Java logits across all six documents differ by at most `0.000003100`, well inside the retained
`0.001` gate, and reproduce the complete `[0, 2, 5, 1, 4, 3]` ranking. The same real checkpoint also
ranks the relevant document first through the LangChain4j `ScoringModel` and Spring AI
`DocumentPostProcessor` adapters.

## Storage and execution decision

The 135.1 MiB F16 checkpoint expands once into F32 arrays at load time. The companion Vectors
experiment found direct mapped F16 projection `12.50x` to `49.11x` slower than reused F32 across
the exact 384/1536-wide DeBERTa shapes. The resulting measured heap increase here is about 350 MiB.
That is an explicit latency-for-memory choice; a direct F16 mode is not exposed as a misleading
"optimization."

Independent token rows are parallelized inside each layer. Three fresh-process runs on the 2019
Intel Mac measured a `58.660 ms` median pair p50 and `17.114` documents/s median six-document
throughput with ten workers, versus `145.608 ms` and `6.331` documents/s with one worker. Scores
are bit-identical across the schedules. `-Dmodels.deberta.threads=N` makes the choice explicit;
the default leaves two processors available to the host on machines with more than four.

Decision: `PASS` for pure-Java graph compatibility, reference equivalence, framework behavior,
and the measured local runtime envelope. The exact artifact is eligible for ModelJars catalog
qualification as a multi-file Safetensors model.

Reproduce the artifact gate with:

```bash
scripts/download-mxbai-reranker-fixture.sh
./gradlew \
  :backend-java:mxbaiDebertaRerankerIntegrationTest \
  :models-langchain4j:mxbaiDebertaLangChain4jIntegrationTest \
  :models-spring-ai:mxbaiDebertaSpringAiIntegrationTest \
  -Dmodels.fixtures.mxbaiRerankerDirectory="$HOME/.jvllm/models/mxbai-rerank-xsmall-v1"
```
