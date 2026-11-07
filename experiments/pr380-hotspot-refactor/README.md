# PR 380 hotspot refactor

MFCQI syntax/coverage defect: [integrallis/mfcqi-java#1](https://github.com/integrallis/mfcqi-java/issues/1).

The refactor separates image padding/resampling/normalization, CLIP layer and projection loading, Llama batch and token execution stages, qualification prompt/decode orchestration, and CLI command dispatch. Arithmetic order, scratch buffers, kernel selection, observer timing, pruning policy, gate exit statuses and benchmark timer boundaries must remain unchanged. Redundant optional-vector validation delegates to the mandatory-vector validator.

## Validation protocol

Before performance collection: compare the pre-refactor `8a34f31a` production sources with the final refactored sources under the same JDK and dependencies. Use existing synthetic F32 and Q4_K Llama test fixtures, not downloaded models. Compile original production classes into an isolated directory and place it first on the baseline JVM classpath; the candidate uses the normal freshly built production classes. No alternative runtime is shipped.

`RefactorBenchmark` reuses the existing deterministic GGUF test builders. The differential check hashes every float in prefill and successive decode outputs, repeated inference, and image preprocessing. All hashes must match exactly. Existing unit/narrow-vector checks additionally cover MoE, adapters, pruning, observers and alternative kernels; the small paired probe does not replace those checks.

JMH 1.37 measures token prefill plus decode, embedding prefill plus decode, and image preprocessing. Use two forks, three 500 ms warmups and five 500 ms measurements per fork, one benchmark thread. Run arms sequentially with no concurrent builds/analyzers. Treat an observed slowdown above 10% as a reason to investigate before accepting; retain all runs, including negative or inconclusive ones. This is a local synthetic regression check, not catalog qualification or a substitute for the prior full-model remote performance evidence.

For structural validation, use the same MFCQI 0.7.0 diagnostic extractor with modern Java parsing enabled before and after. Report extracted helpers as well as original methods. Do not compare a corrected-coverage analysis with an old partial-coverage aggregate.
