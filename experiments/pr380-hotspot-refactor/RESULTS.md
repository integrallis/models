# Measured refactor results

## Cognitive complexity

MFCQI 0.7.0 with the same modern-Java parser setting in both arms. Overloads use the maximum method score. This is not a comparison against the default parser’s partial-coverage aggregate.

| Method | Before | After |
|---|---:|---:|
| `ClipImagePreprocessor.java:preprocess` | 40 | 3 |
| `ClipVisionWeights.java:fromGgufFile` | 15 | 2 |
| `VisionQualificationCli.java:measureDecoder` | 28 | 0 |
| `InferenceBenchmarkCli.java:main` | 71 | 2 |
| `LlamaForwardPass.java:advancePrefillBatch` | 47 | 3 |
| `LlamaForwardPass.java:forwardSequenceInternal` | 42 | 13 |
| `LlamaForwardPass.java:prefillEmbeddings` | 16 | 9 |

## Extracted helpers

All newly named methods in the four analyzed modules, including the command dispatcher.

| Method | Cognitive complexity |
|---|---:|
| `LlamaForwardPass.java:advancePrefillLayer` | 12 |
| `LlamaForwardPass.java:canDiscardFinalLayer` | 1 |
| `LlamaForwardPass.java:executeDensePrefillFfn` | 2 |
| `LlamaForwardPass.java:executeDenseTokenFfn` | 2 |
| `LlamaForwardPass.java:executePrefillAttention` | 0 |
| `LlamaForwardPass.java:executePrefillFfn` | 6 |
| `LlamaForwardPass.java:executeTokenAttention` | 11 |
| `LlamaForwardPass.java:executeTokenFfn` | 2 |
| `LlamaForwardPass.java:notifyPrefillLayer` | 3 |
| `LlamaForwardPass.java:projectPrefillLogits` | 0 |
| `LlamaForwardPass.java:projectPrefillOutput` | 3 |
| `LlamaForwardPass.java:projectPrefillQkv` | 2 |
| `LlamaForwardPass.java:projectTokenOutput` | 2 |
| `LlamaForwardPass.java:projectTokenQkv` | 2 |
| `LlamaForwardPass.java:validateEmbeddingRows` | 7 |
| `ClipImagePreprocessor.java:padSquare` | 14 |
| `ClipImagePreprocessor.java:resizeHorizontal` | 10 |
| `ClipImagePreprocessor.java:resizeVerticalAndNormalize` | 10 |
| `ClipImagePreprocessor.java:validateNormalization` | 6 |
| `ClipVisionWeights.java:attentionProjection` | 4 |
| `ClipVisionWeights.java:loadLayer` | 3 |
| `ClipVisionWeights.java:patchEmbedding` | 2 |
| `InferenceBenchmarkCli.java:dispatchCommand` | 0 |
| `VisionQualificationCli.java:bestToken` | 5 |
| `VisionQualificationCli.java:decodeGreedy` | 5 |
| `VisionQualificationCli.java:measureDecoderRuns` | 5 |
| `VisionQualificationCli.java:preparePrompt` | 4 |

## Synthetic JMH regression check

Same host/JDK, two forks per arm; three warmups and five measurements per fork, 500 ms per iteration. Time is mean microseconds per operation; ratio is after / before (lower is faster). This small synthetic measurement does not establish full-model catalog performance.

| Operation | Fixture | Before µs | After µs | Ratio |
|---|---|---:|---:|---:|
| embeddingPrefillAndDecode | buildNanoModel | 170.051 | 148.586 | 0.8738 |
| embeddingPrefillAndDecode | buildQ4KNanoModel | 9194.572 | 10576.695 | 1.1503 |
| imagePreprocess | image | 1033.820 | 1339.589 | 1.2958 |
| tokenPrefillAndDecode | buildNanoModel | 137.570 | 140.806 | 1.0235 |
| tokenPrefillAndDecode | buildQ4KNanoModel | 8912.034 | 9142.689 | 1.0259 |

## Longer-warmup investigation

The initial short run crossed the 10% investigation threshold for these two operations. The repeat used five 1 s warmups and five 1 s measurements per fork, two forks, and reversed arm order (after, then before). Both initial and repeat data are retained. Intervals are JMH 99.9% confidence intervals, not equivalence bounds.

| Operation | Before µs | After µs | Ratio | Before CI | After CI |
|---|---:|---:|---:|---|---|
| imagePreprocess | 1003.827 | 945.047 | 0.9414 | 843.8–1163.9 | 882.3–1007.8 |
| embeddingPrefillAndDecode | 8282.438 | 9045.786 | 1.0922 | 6743.8–9821.1 | 7727.2–10364.3 |

The repeat means are within the predeclared 10% investigation threshold, but the decoder intervals overlap widely. These noisy local measurements do not establish zero overhead or replace a dedicated full-model performance repeat.


Exact differential check: **39 output fingerprints matched**. Fingerprints include prefill logits, successive decode logits, repeated inference and every preprocessed pixel. Route logs confirm sequential F32 and batched/grouped Q4_K execution.

Full unit, narrow-vector, formatting, static-analysis and coverage checks passed for backend-java and models-bench. Raw JMH distributions, output fingerprints, logs, source revisions and hashes are retained in this directory.
