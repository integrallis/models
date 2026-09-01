# MS MARCO MiniLM L6 v2 reranker compatibility: 2026-09-01

This experiment asks whether Models can execute a useful cross-encoder reranker entirely in Java,
through the public API and framework adapters, while preserving the reference model's logits and
ordering. CrispEmbed is an independent oracle only; it is not a Models runtime dependency.

## Pinned inputs

| Input | Value |
| --- | --- |
| Upstream model | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Upstream license | Apache-2.0 |
| Corrected GGUF repository | `cstr/ms-marco-MiniLM-L-6-v2-GGUF` |
| Repository revision | `1a9ef5ce8cb08936338233731314f3ff61ce0930` |
| Artifact | `ms-marco-MiniLM-L-6-v2-q4_k-imatrix-g7c-f7.gguf` |
| Bytes | 19,986,112 |
| SHA-256 | `0752a92bc33289f3fe230d1a33cce471f1a7fe1c8d5a491ca7fff6b068b0fc83` |
| CrispEmbed revision | `e4cfd3dd5353c4a4e9d48c3e6657491d38e797d3` |
| CrispEmbed ggml revision | `890278a8342c620197c90e702e1188bcab94f510` |

The artifact URL is revision-pinned in `backend-java/src/test/resources/model-fixtures.properties`.
The integration task downloads it only when the expected file is absent and verifies its SHA-256
before execution.

## Workload

Query: `How many people live in Berlin?`

| Index | Document |
| ---: | --- |
| 0 | Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers. |
| 1 | Paris is the capital and most populous city of France. |
| 2 | Berlin is well known for its museums and its metropolitan area of about six million people. |
| 3 | Domestic cats sleep for a large part of the day. |
| 4 | New York City had an estimated population of 8,804,190 in 2020. |
| 5 | The Berlin Wall divided the city from 1961 until 1989. |

The retained test first compares the exact `[CLS] query [SEP] document [SEP]` token sequence and
segment IDs. It then checks each Java logit against both the unquantized ONNX reference and the
same Q4_K artifact executed by the independent oracle.

## Correctness result

| Index | ONNX | Q4_K oracle | Models Java | Java vs ONNX | Java vs Q4_K |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8.846 | 8.821 | 8.815807 | 0.030193 | 0.005193 |
| 1 | -10.886 | -10.985 | -10.987034 | 0.101034 | 0.002034 |
| 2 | 7.401 | 7.313 | 7.304770 | 0.096230 | 0.008230 |
| 3 | -11.225 | -11.271 | -11.269738 | 0.044738 | 0.001262 |
| 4 | -5.200 | -5.149 | -5.152791 | 0.047209 | 0.003791 |
| 5 | -4.944 | -4.892 | -4.855608 | 0.088392 | 0.036392 |

Result: `PASS`. The maximum absolute delta is 0.101034 against ONNX and 0.036392 against the
quantized oracle. Both implementations rank documents 0 and 2 first, in that order. The retained
policy is 0.15 per logit against ONNX, 0.05 against the same-artifact oracle, exact top-two order,
deterministic repeated scores, and a changed score when query and document are reversed.

## Defects the experiment found

1. The GGUF declares BERT's `cls_token_id` and the standard GGUF spelling
   `seperator_token_id`, not generative BOS/EOS IDs. Treating it as a generative tokenizer silently
   produced the wrong boundary tokens.
2. Sentence pairs require token-type embeddings and longest-first truncation. When both sides are
   the same length, the reference removes a token from the first side.
3. The corrected conversion contains `classifier.dense.*` followed by tanh and
   `classifier.out_proj.*`. Earlier, uncorrected conversions omitted this pooler and emitted logits
   around plus or minus 0.2 rather than the reference scale near plus or minus 11. Those artifacts
   are rejected.
4. BERT uses the erf-exact GELU. The common tanh approximation is a different graph. Java 25's
   scalar math and Vector API expose no `erf`, so the Java implementation currently owns a tested
   scalar approximation.
5. The scalar output projection can be encoded as a rank-one tensor. Requiring a two-dimensional
   matrix rejects a valid corrected artifact.

## Public and framework gates

- `RerankingModel` scores and stably reranks documents without framework types.
- LangChain4j's `ScoringModel` adapter preserves query/document order and reference scores.
- Spring AI's `DocumentPostProcessor` adapter preserves document identity and metadata, attaches
  the reranker score, and returns the configured top-N documents.

Commands:

```bash
./gradlew \
  -Dmodels.fixtures.directory=/private/tmp/model-fixtures \
  :backend-java:msMarcoMiniLmRerankerIntegrationTest \
  :models-langchain4j:msMarcoRerankerLangChain4jIntegrationTest \
  :models-spring-ai:msMarcoRerankerSpringAiIntegrationTest
```

The first retained run used Temurin 25.0.3+9 on macOS x86-64, a six-core 2.6 GHz Intel Core i7,
and the 256-bit Panama Vector API provider. The three backend integration checks completed in
4.579 seconds, including three separate model mappings and 15 score operations. This is a
correctness and adapter smoke, not a throughput benchmark; ModelJars qualification must measure
cold load, warm per-pair latency, batch latency, and memory before catalog promotion.

## Product and JVM consequences

Models gains a Java-owned BERT cross-encoder graph and public reranking surface without importing
an external inference engine. The missing standard scalar/vector `erf` is now request `JVM-AI-9`
in `JVM_NATIVE_INFERENCE_GAP.md`, with this exact artifact and score gate as its acceptance test.
