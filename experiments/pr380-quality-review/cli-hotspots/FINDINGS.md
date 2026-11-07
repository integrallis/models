# CLI scores and hotspots

Generated from MFCQI 0.7.0 CLI JSON and the same version’s cognitive/smell analyzers. Scores are out of 100. Cognitive complexity is a raw count; its default hotspot threshold is 15.

| Module | CLI score | Source files | Default parser failures | Modern parser failures |
|---|---:|---:|---:|---:|
| backend-java | 66.4612 | 221 | 98 | 0 |
| backend-native | 81.7774 | 15 | 5 | 0 |
| models-api | 94.5758 | 46 | 18 | 0 |
| models-bench | 65.5531 | 117 | 80 | 0 |

## Selected PR hotspots with modern syntax enabled

Main values match by file and method name; overloads are listed together. “Absent” means no same-named method in that file on main, not necessarily entirely new logic.

| Method | Branch line | Main cognitive | Branch cognitive |
|---|---:|---:|---:|
| `ClipImagePreprocessor.java:preprocess` | 41 | Absent | 40 |
| `VisionQualificationCli.java:measureDecoder` | 105 | Absent | 28 |
| `LlamaForwardPass.java:advancePrefillBatch` | 1311 | Absent | 47 |
| `LlamaForwardPass.java:forwardSequenceInternal` | 1946 | 33 | 42 |
| `LlamaForwardPass.java:prefillEmbeddings` | 738 | Absent | 16 |
| `ClipVisionWeights.java:fromGgufFile` | 117 | Absent | 15 |
| `InferenceBenchmarkCli.java:main` | 73 | 68 | 71 |
| `RustGgufBatchedMatrixKernel.java:multiply` | 917 | 11 | 11 |
| `NativeCausalAttention.java:attend` | 82 | Absent | 9 |

## Highest cognitive complexity per module (whole module, including existing code)

| Module | Method | Line | Cognitive |
|---|---|---:|---:|
| backend-java | `LlamaWeights.java:fromGgufFile` | 156 | 56 |
| backend-java | `ModernBertDecisionForwardPass.java:attention` | 262 | 49 |
| backend-java | `ActivatedLoraAdapter.java:validate` | 819 | 48 |
| backend-java | `LlamaForwardPass.java:advancePrefillBatch` | 1311 | 47 |
| backend-java | `PrismHadamard.java:load` | 37 | 44 |
| backend-native | `NativeKernelLibrary.java:openExclusively` | 406 | 12 |
| backend-native | `NativeKernelLibrary.java:groupedGatedDeltaNetF32` | 1034 | 11 |
| backend-native | `RustGgufBatchedMatrixKernel.java:multiply` | 917 | 11 |
| backend-native | `NativeDebertaKernels.java:causalAttention` | 248 | 10 |
| backend-native | `RustGgufBatchedMatrixKernel.java:close` | 1093 | 10 |
| models-api | `BatchInferenceBackend.java:prefillBatch` | 106 | 13 |
| models-api | `ActivatedAdapterMetadata.java:requireRelativeFile` | 308 | 6 |
| models-api | `PcmAudio.java:PcmAudio` | 30 | 6 |
| models-api | `SamplingOptions.java:build` | 199 | 6 |
| models-api | `ModelPrompt.java:append` | 114 | 4 |
| models-bench | `RouterSelectionEvaluationCli.java:evaluate` | 48 | 94 |
| models-bench | `InferenceBenchmarkCli.java:main` | 73 | 71 |
| models-bench | `FusionDecoder.java:fuse` | 60 | 37 |
| models-bench | `ArmRunner.java:resolve` | 76 | 31 |
| models-bench | `FusionCli.java:runArm` | 280 | 31 |

The default parser rejects modern syntax and the analyzer silently skips affected files. Modern-mode diagnostics change only JavaParser’s language level to BLEEDING_EDGE; they are supplemental findings, not newly calculated aggregate CLI scores. No LLM service was invoked.
