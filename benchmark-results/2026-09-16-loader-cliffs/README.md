# Loader size assertions, named performance cliffs, species-as-constant rule

Date: 2026-09-16. Branch `feat/loader-integrity-and-cliffs`; the jars the process runs used were
built from commit `0e179163`.

Everything below was **measured here** unless it says otherwise. Nothing here measures speed. The
timings in `runs/*/probe.txt` and the `parse-ms` column in `loader-scan.txt` are uncontrolled (cold
page cache, runs executed back to back on a laptop) and back no claim.

## Configuration

| | |
|---|---|
| Host | MacBook Pro, Intel Core i7-9750H (6C/12T, AVX2), 32 GB, macOS (Darwin 25.6.0, x86_64) |
| JVM | Temurin 25.0.3+9, `--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx8g` |
| JFR | `-XX:StartFlightRecording:filename=run.jfr,settings=default`, read with `jfr print --events com.integrallis.models.PerformanceCliff` |
| vectors-core | 0.1.22 (preferred 256 bits, active 256 bits, persistent executor, 12 threads by default) |
| Native kernel | `backend-native/build/rust-target/release/libjmodels_kernels.dylib`, ABI of `rust-ffm-v13` |
| Harness | `harness/CliffProbe.java`: load, prefill a 10-token prompt, greedy-decode N tokens (16 for the Qwen3 baseline, 4 for every other run), print diagnostics. `harness/run.sh` holds the flags, `harness/batch.sh` the full matrix |
| Models | `Qwen3-0.6B-Q4_0.gguf`, `granite-4.1-3b-Q4_K_M.gguf`, `Qwen3.5-0.8B-Q4_K_M.gguf` (local copies) |

Every run's exact flags are in `runs/<label>/config.txt`, the diagnostics it printed are in
`probe.txt`, and the full JFR events with stack traces are in `jfr-events.txt`.

## 1. Per-tensor size assertions

Red then green: `evidence-loader-red.txt` (9 of 11 new tests failing, and the 2 that passed are the valid-file controls; the three GGUF overlap cases
were **accepted without any error** before the change) and `evidence-loader-green.txt`.

What was provoked, in synthetic files (`GgufTensorSizeAssertionTest`,
`SafetensorsTensorSizeAssertionTest`):

| Corruption | GGUF before | GGUF after | Safetensors after |
|---|---|---|---|
| Last tensor truncated by 3 bytes | rejected, message had no expected/available counts | `expected 8 bytes, available 5 before the end of file` | `expected 8 bytes, available 4 ... runs past the end` |
| Q4_0 tensor with one of its two blocks | rejected, no counts | `(Q4_0, shape [64]) expected 36 bytes, available 18` | n/a |
| Tensor overlaps the next one, both inside the file | **accepted**: the tensor read its neighbour's bytes | `expected 8 bytes, available 4: it overlaps tensor '...'` | `tensor b overlaps tensor a` (rejected before too, with a less specific message) |
| Tensor table out of offset order, with an overlap | **accepted** | rejected; the overlap is found after sorting by offset | n/a |
| Two tensors at the same offset | **accepted** | rejected, `available 0` | n/a |
| Offset past the end of the file | rejected, no counts | `expected 4 bytes, available 0 ... past the end of the file` | n/a |

For Safetensors, the rule was already strict before this change (ranges have to tile the buffer
exactly). The change adds the tensor name, dtype, shape, and expected and available counts to each
message. The set of files it rejects is the same by construction, because every check that was
there before is still there.

**Real files (no behaviour change on valid models):** `loader-scan.txt` parses every GGUF and
Safetensors file on this host with the new parsers: 36 GGUF files (Qwen2.5/3/3.5, Granite 3.2/4.1,
Gemma 3, SmolLM2/3, TinyLlama, DeepSeek, MiniCPM5, BERT-family embedders and rerankers, F16/F32/Q4_0/
Q4_K_M/Q5_K_M/Q8_0) and 20 Safetensors files (LoRA adapters, audio models, the mxbai reranker).
**56 accepted, 0 rejected.**

Not done: GGUF tensor offsets are **not** checked for multiples of `general.alignment`. I believe
llama.cpp's reader enforces that, but I have not checked its source here. The synthetic builder
writes unaligned offsets, so adding the check means changing that builder first. Left as a
follow-up.

## 2. Named performance cliffs

Each reason is reported from the branch that takes the slower path, at most once per process. It
shows up as a JFR event `com.integrallis.models.PerformanceCliff` (category `Models`, fields `reason`
and `detail`, with a stack trace) and in backend diagnostics as `performance-cliffs=<ids>` and
`performance-cliff.<id>=<first detail>`. If no cliff has been reported, diagnostics come back as the
same object as before.

Red then green: `evidence-cliffs-core-red.txt` / `evidence-cliffs-red.txt` (every positive test
failing against a no-op registry), `evidence-cliffs-green.txt`, `evidence-native-red.txt`,
`evidence-native-green.txt` (the whole backend-native suite, 42 tests).

### Real-process runs (JFR on, real models)

| Reason (`id`) | Emitted at | Provoked by | Events for that reason | Diagnostic key observed | Fast-path control (0 events for that reason) |
|---|---|---|---|---|---|
| `vector-api-unavailable` | `ExecutionPlanner.reportVectorWidthCliffs` | `q3-java-forceScalar`: `-Dvectors.forceScalar=true` (provider `ScalarVectorUtilSupport`, active bits 0) | 1 | `performance-cliff.vector-api-unavailable=vector-provider=ScalarVectorUtilSupport, preferred-vector-bits=256` | `q3-java-baseline` |
| `vector-width-capped` | same | `q3-java-maxBits128`: `-Dvectors.maxBits=128` (active 128 < preferred 256) | 1 | `performance-cliff.vector-width-capped=active-vector-bits=128, preferred-vector-bits=256, ...` | `q3-java-baseline` |
| `persistent-executor-not-used` | `ExecutionPlanner.persistentExecutor` | `q3-java-executor-dedicated`: `-Dvectors.gguf.executor=dedicated` | 1 | `performance-cliff.persistent-executor-not-used=executor=dedicated, gguf-parallel=true, gguf-threads=12` | `q3-java-baseline` |
| `q4-pairwise-kernel-unsupported` | `ExecutionPlanner.q4Kernel` (both pairwise branches) | `q3-java-shortPairwise-maxBits128`: `-Dmodels.purejava.q4Kernel=short-pairwise -Dvectors.maxBits=128` | 1 | `performance-cliff.q4-pairwise-kernel-unsupported` | `q3-java-shortPairwise` (256 bits: the pairwise kernel is supported, 0 events) |
| `fused-grouped-attention-not-wired` | `LlamaForwardPass` constructor | every Qwen3 0.6B run (GQA with group size 2, not Granite) | 1 per process | `performance-cliff.fused-grouped-attention-not-wired=architecture=qwen3, group-size=2` | `granite41-java-baseline` (0 events in total) |
| `native-grouped-attention-unavailable` | `LlamaForwardPass` constructor (Granite with an injected kernel) | `granite41-native-groupedAttention-false`: `-Dmodels.native.groupedAttention=false` | 1 | `performance-cliff.native-grouped-attention-unavailable=architecture=granite, kernel=rust-ffm-quantized-v13` | `granite41-native-baseline` (0 events in total) |
| `native-gated-delta-net-unavailable` | `Qwen35ForwardPass.applyGatedDeltaNetRecurrence`, called for every GDN layer on every prefill chunk and decode step | `qwen35-native-gdn-false`: `-Dmodels.native.gatedDeltaNet=false` | 1 (the branch ran on every GDN layer for the prefill and each of the 4 decode steps) | `performance-cliff.native-gated-delta-net-unavailable=architecture=qwen35, kernel=rust-ffm-quantized-v13` | `qwen35-native-gdn-true`, `qwen35-java-baseline` (0 events in total each) |

### Unit-tested in process only

| Reason | Emitted at | Provoked by | Observed | Why no real-process run |
|---|---|---|---|---|
| `batched-prefill-unsupported-tensor-type` | `LlamaForwardPass` constructor; `Qwen35ForwardPass.withExecutionPlan` | synthetic F32 Llama (`LlamaForwardPassPerformanceCliffTest`, 3 constructions, 1 event); toy F32 Qwen 3.5 through `PureJavaBackend.load` twice (1 event, diagnostic key present) | 1 event; F16 projections give 0 | No local model has an unbatched projection type (none of the real runs reported it) |
| `row-by-row-projection` | `Qwen35ForwardPass.projectBatched`, per-row branch | `Qwen35ForwardPass.fromGgufFile(file)` (default batch 32) on the toy F32 model, 3 prompts | 1 event; batch size 1 gives 0 | Only reachable through the public `Qwen35ForwardPass.fromGgufFile` graph API. `PureJavaBackend` plans batch 1 for such a model first and reports `batched-prefill-unsupported-tensor-type` instead |
| `native-poll-budget-unsupported` | `NativeKernelLibrary.applyPollBudget` (the path taken when `setPollNanos == null`) | called with a null handle 3 times | 1 event; opening the bundled library (which has the capability) gives 0 | The only ABI-matching library on the host has `POLL_BUDGET`; an older library would fail the ABI check before this branch |

The planner reasons are also unit-tested with synthetic fingerprints
(`ExecutionPlannerPerformanceCliffTest`), and so is the once-per-process guarantee: 10,000 reports
from one thread, and 16 threads x 1,000 reports, each commit exactly 1 event.

### Asked for but not implemented, and why

- **Native kernel library absent**: there is no fallback. `RustFfmBackend.load` throws, so no slower
  path is ever taken, and naming one would describe a branch that does not exist.
- **Native access restricted**: no branch checks it. On JDK 25 the FFM downcall either prints the
  restricted-method warning and proceeds at the same speed, or throws under
  `--illegal-native-access=deny`. Neither is a fast path being skipped.
- **Serial GGUF rows (`-Dvectors.gguf.parallel=false`)**: `q3-java-parallel-false` reported no cliff,
  because vectors-core still names the executor `persistent`. Parallel rows being turned off is a
  real slower path that is **not named** yet. Follow-up: add `gguf-parallel-disabled` if we treat
  that setting as a cliff and not as a choice.
- **Native grouped attention falling back per row** when the cache view has more than two spans
  (`LlamaForwardPass.attendNatively` returns false): real, not named, and not provokable without a
  multi-span native test. Follow-up.

### Honest limits

- A cliff is process-scoped. Once one model reports `vector-width-capped`, the diagnostics of every
  backend in that JVM carry it. That is deliberate, because the cause (JVM flags) is process-wide
  too, but it means the key does not tell you which model hit it.
- The price of a cliff was **not measured** here. This PR names and counts fallbacks. How much each
  one costs is a separate A/B per reason.

## 3. Species-as-constant structural rule

`VectorSpeciesConstantStructuralTest` loads (without initialising) every class that references
`jdk/incubator/vector/VectorSpecies`. It fails on a species method, constructor, or lambda
parameter, or on a species field that is not `static final`. Red then green:
`evidence-species-red.txt` (4 of 5 failing against a no-op rule), `evidence-species-green.txt`.

- **backend-java: 0 violations.** Scanned `TensorOps` and `GroupedQueryAttentionKernel`, which are the
  only two main classes that use the Vector API.
- **vectors-core 0.1.22 (the pinned jar, where most kernels live): 1 violation**,
  `PanamaConstants#preferredSpecies(VectorSpecies)`. `javap -c` shows that all 5 call sites are in
  `PanamaVectorUtilSupport.<clinit>`, so the species the kernels see is still a static final
  constant. The violation is recorded in an explicit allow-list, as a follow-up for the vectors
  project.
- The 11 MB to 162 GB allocation figure that motivated the rule is **self-reported by another
  engine and not reproduced here**. No A/B was run: our only violation sits in a static initializer,
  so no kernel pays a per-call species parameter, and a cheap experiment would have to construct
  that pathology rather than measure any code we ship. **Open measurement:** a JMH
  `-prof gc` A/B of one kernel written with its species as a constant versus as a parameter, on C2
  and on Graal.
