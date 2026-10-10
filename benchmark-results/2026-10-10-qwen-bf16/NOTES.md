# Qwen2.5 BF16 snapshot repair — real candidate validation

The repaired smoke path passed with the actual four-file Qwen2.5-0.5B-Instruct BF16 snapshot.
The previous worker saved the weights as GGUF and omitted the tokenizer and configuration. The
catalog URI and format were already correct. The runtime's existing directory loader needed no change.

The user explicitly requested local installation and real-model testing. These runs were performed
on the developer Mac, overriding the campaign runbook's usual remote-only instruction for this work.
No remote instance was launched. All four files were downloaded by the new worker's actual embedded
helper and verified against their pinned SHA-256 and byte size. See `inputs.json`.

## Actual fixture tests

Both selected tests executed; **2 passed, 0 skipped, 0 failures**. JUnit XML and the Gradle log are in
`fixture-tests/`.

- `HuggingFaceTokenizerTest.matchesPinnedOfficialQwen25Tokenizer`: configuration/tokenizer hashes
  and the real tokenizer's expected encodings.
- `Qwen2HuggingFaceBackendIntegrationTest.matchesPinnedHuggingFaceFloat32ReferenceLogits`: loads the
  BF16 directory, checks ChatML token IDs, and compares the first 16 logits plus the winning logit
  against the recorded Hugging Face float32 reference within 1e-3. The argmax token is exactly 3966.

## Default correctness smoke

The actual candidate JARs passed all **9/9** cases, with `correctAnswerRate=1`,
`abstentionAccuracy=1`, `failures=[]`, zero warmups and one iteration. Prefix-cache reads were exercised
(807 tokens). The shipped worker's gate returned `PASS` (`gate.txt`).

This is pipeline correctness: the model contributed on 5/9 cases and all five contributions were
correct; 3/9 used extractive fallback. There was no truncation. Settings are `general`, `chatml`,
2048 context, 8 threads, 64 output tokens, and the current default grounding policy v23. No
`-Dmodels.*` tuning property was used; sampling controls stayed at their defaults.

The raw report is `default-correctness/models-pure-java.json`, SHA-256
`5ad785faee6fb91ae6a70d4d121e6276a81a4992fea700bd989cda990fc68df6`.
Console output is retained beside it. This single-arm regression run is not a new comparative
qualification or a speed comparison with earlier reports. `peakRssBytes=0` means RSS was not measured.

## Candidate provenance and reproduction

Models **0.3.57 candidate**, with the verified staged Vectors **0.1.29** artifacts. These were not
downloaded from Central as released versions. `run-inputs.json` records the exact Java command,
JDK version, runtime JAR hashes and sizes, source base commit, working-tree patch digest and timestamps.
The source base is `be4493e1`; decompress and apply `candidate.patch.gz`, then copy `candidate-build-inputs/` into that
checkout to reconstruct the candidate source/build inputs. The patch includes the pending audit
repairs and dependency updates; it is retained as evidence, not an additional implementation branch.
The recorded patch digest applies to the decompressed bytes.

The real tests used:

```sh
./gradlew :backend-java:test \
  --tests '*HuggingFaceTokenizerTest.matchesPinnedOfficialQwen25Tokenizer' \
  :backend-java:integrationTest --tests '*Qwen2HuggingFaceBackendIntegrationTest' \
  -Dmodels.fixtures.qwen25HuggingFaceDirectory=/path/to/verified/snapshot \
  -PnotebookRepository=/path/to/vectors/build/staging-deploy --max-workers=4
```

The benchmark distribution was freshly prepared with `:models-rag-bench:installDist` using the same
staged repository. `run-inputs.json` supplies the exact smoke command. After publication, repeat the
smoke using the final Central JARs and retain a separate report before binding released evidence.
Historical 0.3.54 outcomes remain unchanged.
