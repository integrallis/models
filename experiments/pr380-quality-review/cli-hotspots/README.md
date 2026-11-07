# MFCQI CLI hotspot follow-up

Ran the installed `mfcqi-java 0.7.0` native CLI against all four changed modules' `src/main/java` trees. All four runs exited successfully and reproduce the preceding Gradle scores to its reported precision. [Generated findings](FINDINGS.md) contain the scores, parser coverage and method-level counts. Raw CLI JSON, diagnostics and exact source/tool hashes are in `results/`.

## Important measurement limitation

The default MFCQI parser rejects records, switch expressions and other modern Java constructs. MFCQI's AST analyzers silently skip the rejected files. The diagnostic pass reproduced this and then enabled JavaParser's `BLEEDING_EDGE` language level: all source files parsed in that mode. The existing scores are reproducible, but their AST coverage is incomplete. They should not be treated as a full-source quality verdict. Some non-AST metrics still scan those files, so this is not a uniform whole-file exclusion from every metric.

The installed CLI's metrics-only JSON does not expose method findings. `ExtractHotspots.java` calls the published MFCQI 0.7.0 cognitive complexity and Java smell analyzers directly to expose them. Both default and modern-parser results are retained, together with modern-parser diagnostics on pinned main. This does not replace the CLI scores with an unreported corrected aggregate. No LLM service or model inference was used.

## Review priorities

1. **Image preprocessing:** `ClipImagePreprocessor.preprocess(BufferedImage, ...)` combines normalization validation, square padding, horizontal bicubic resampling, vertical resampling and CHW normalization. Extract these meaningful stages, keeping Pillow's integer rounding and clamping order exact. This is the clearest production refactor. The generated findings show its cognitive count; the raw smell report also flags its length.
2. **Vision benchmark orchestration:** `VisionQualificationCli.measureDecoder` combines prompt/BOS handling, repeated multimodal prefill, greedy selection, timing and report assembly. Separate token preparation and greedy selection from the measured repetition loop. Preserve timer boundaries and output artifacts; the modern parser finds this method, whereas the default parser skips its enclosing file because of a record declaration.
3. **Llama prefill:** `advancePrefillBatch` and `forwardSequenceInternal` carry branch-heavy projection, attention, FFN and cache logic. Factor layer stages with explicit state and buffer ownership. These are numerical hot paths; a refactor must retain exact-output checks and paired performance measurements. The new `advancePrefillBatch` method contains extracted existing work, so its absence on main does not mean all its complexity was newly introduced.
4. **Vision weight loading:** `ClipVisionWeights.fromGgufFile` mixes weight resolution, conversion and shape checks. Extract coherent validation/loading helpers while preserving strict profile rejection. Its cognitive score reaches the default hotspot threshold.
5. **Native backend:** there is no cognitive-complexity hotspot at the default threshold. The large, existing `RustGgufBatchedMatrixKernel.multiply` still merits structural attention for length and its many dispatch cases; its cognitive score is unchanged from main. Long parameter lists on FFM interfaces are often intentional. Do not add allocations or abstraction layers to hot paths solely to improve a smell score.
6. **API:** no cognitive hotspots at the default threshold; the new image inference interface adds no flagged implementation. No immediate cleanup is indicated.

`InferenceBenchmarkCli.main` is also an obvious dispatcher hotspot. Most of that complexity predates the PR; this PR adds one vision dispatch branch. A command registry is a broader tooling cleanup, not the main new vision defect. The module's highest cognitive method is the pre-existing router evaluation CLI; the generated report shows whole-module leaders to avoid attributing existing problems to this PR.

The two CLI `Secrets Exposure` components that score zero are heuristic scan results, not confirmed leaked credentials; this pass does not claim a validated secrets finding. It focuses on actionable structural hotspots.

## Reproduction

Use the revisions and hashes in `results/provenance.json`. Run from each source checkout:

```sh
mfcqi analyze MODULE/src/main/java --skip-llm --parallelism=1 \
  --format=json --output=MODULE.json
```

For method details, put the five published JARs listed in provenance on `MFCQI_DIAGNOSTIC_CP` (MFCQI core, metrics and smells 0.7.0; JavaParser 3.26.4; SLF4J API 2.0.17). With Java 25.0.3:

```sh
java -Xmx2g --class-path "$MFCQI_DIAGNOSTIC_CP" /path/to/ExtractHotspots.java default \
  backend-java/src/main/java backend-native/src/main/java \
  models-api/src/main/java models-bench/src/main/java > default-details.tsv
```

Repeat with `modern` instead of `default`. Repeat `modern` on pinned main for attribution. Gzip the TSVs into the corresponding `results/` files, preserve CLI JSON, and run `python3 summarize.py` to regenerate `FINDINGS.md`.

No production code was changed in this follow-up. Before drawing stronger score comparisons, fix the analyzer's Java-language configuration and make parse failures visible, then rerun both main and PR with the corrected analyzer.
