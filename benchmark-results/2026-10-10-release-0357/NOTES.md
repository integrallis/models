# Models 0.3.57 released-library correctness

Measured locally on 2026-10-10 after Maven Central publication. This directory is distinct from
the earlier candidate Qwen evidence; none of that historical evidence was overwritten.

## Artifact and source identity

- Source/tag: `7bde1c2ef31c608069294e47c17d2efccbc58afb`, `v0.3.57`.
- Successful release: https://github.com/integrallis/models/actions/runs/38087706900.
- Central deployment: `50b08370-7fbd-4826-ada2-cf4521ac0307`; publication and all 16 independent
  Maven/Gradle consumer pairs passed. Receipts are under `inputs/`.
- Runtime libraries came from Maven Central through a fresh Gradle consumer/cache, without
  source substitution, local Maven fallback or added consumer dependency-policy rules.
  The complete combined graph was checked against the shared policy and embedded manifests.
- `qwen-bf16/run-inputs.json` records 58 classpath JARs: 57 Central artifacts and the unpublished
  `models-rag-bench` harness built from the clean tagged source. Embedding checks add the
  unpublished `models-bench` harness from the same source. Every JAR has a SHA-256 receipt.
- The native check used the Central `backend-native` bundle, including its embedded macOS Intel
  library. No separately supplied native library or tuning system property was used.
- Actual environment: macOS 26.6.2, x86_64, Temurin JDK 25.0.3. Other native platforms passed the
  release build/test matrix; these local inference reports measure this host only.

## Measured results

| Check | Result |
| --- | --- |
| Qwen2.5-0.5B-Instruct BF16, pure Java | 9/9 pipeline cases correct, correct abstention, no failures/truncation; actual smoke-worker gate PASS |
| Qwen2.5-Coder-0.5B Q4_0, rust-ffm | 9/9 pipeline cases correct, correct abstention, no failures/truncation; native-quantized-decode=true; actual smoke-worker gate PASS |
| MiniLM Q4_K_S | 8 oracle probes pass; minimum cosine 0.9992974227 |
| Granite 107M Q4_K_S | 8 oracle probes pass; minimum cosine 0.9998214817 |
| Granite 107M Q5_K_S | 8 oracle probes pass; minimum cosine 0.9997886055 |
| Granite 107M Q5_K_M | 8 oracle probes pass; minimum cosine 0.9997759365 |
| LFM2.5 350M Q4_0 | 8 oracle probes pass; minimum cosine 0.9997358092 |
| LFM2.5 350M Q8_0 | 8 oracle probes pass; minimum cosine 0.9997379672 |
| LFM2.5 350M BF16 | 8 oracle probes pass; minimum cosine 0.9999914692 |
| LFM2.5 350M F16 | 8 oracle probes pass; minimum cosine 0.9999993816 |

Embedding probes use the existing exact-artifact llama.cpp oracle vectors and the existing 0.999
cosine floor. All eight catalog artifacts were hash/size verified before loading. Oracle data and
acceptance thresholds were not changed for this release.

The pipeline's 9/9 does not mean nine model-authored answers. Qwen BF16 used grounded model answers
on 5/9 cases and extractive fallback on 3/9; the native coder used model answers on 4/9 and fallback
on 4/9. Both correctly abstained on the remaining unsupported case. The model-answer correctness
rate is 1.0 in both reports, while their separate raw-answer correctness field is 0.0. Preserve this
distinction when citing the results.

## Limits and reproduction inputs

These are default-correctness and embedding regression checks, not a new comparative qualification
or performance campaign. Some checks overlapped other build/test activity. Do not use their timing
differences to claim a speedup or rank backends. Peak RSS is recorded as zero (unmeasured), not zero
memory use. Prefix-cache token counts are observed diagnostics, not an ablation.

The two RAG reports retain exact corpus/workload, template, context/thread/output budgets, generation
controls, model hashes, runtime labels and backend diagnostics. The Qwen four-file Safetensors
snapshot and eight embedding downloads have installation receipts under `inputs/`. The native
fixture hash and settings provenance are in its run receipt. No weights or runtime JARs are committed.

The three `inputs/run-released-*.py` files preserve the actual driver code. They were run from the
workspace's `audit-2026-10-10/` directory; their recorded host-specific workspace/JDK paths must be
provided when reproducing. The Qwen driver requires the clean tagged checkout and a new output
directory; the other two require its successful, hash-verified Central runtime. Run commands are
also recorded beside each result. `inputs.json` hashes the retained raw files.

This record does not refresh the entire historical RAG catalog, remeasure composites, certify CUDA
devices or alter the older embedding qualification records. ModelJars may bind the new Qwen
default smoke separately while retaining the original qualification's provenance.
