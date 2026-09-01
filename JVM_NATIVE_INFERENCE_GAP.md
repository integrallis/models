# Experiment-backed requests for JVM-native AI inference

Status: working research brief, 2026-09-01.

This document is the evidence base for an open request to JVM, JDK-distribution, compiler, and Java
accelerator leaders. It separates demonstrated gaps from hypotheses and gives every request a
retained reproducer and measurable acceptance criterion. It is not a claim that Java cannot run the
supported models. The opposite is already established: the
pure-Java backend parses the artifacts, tokenizes prompts, executes every supported graph, owns
model state, samples tokens, and integrates with Spring AI and LangChain4j. The optional Rust code
exists to improve a narrow performance envelope for selected CPU profiles.

## The current native boundary

Models does not embed llama.cpp, Ollama, FreeToken, Needle, or another inference engine. The
Models-owned ABI 5 crate exposes a persistent worker context and these compute operations:

| Native operation | Formats or graph | Java ownership outside the call | Current reason to keep it |
| --- | --- | --- | --- |
| Batched, grouped, and independent quantized matrix projection | Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K, and mixed K-quant groups | Artifact mapping, activation lifetime, graph scheduling, attention, KV cache, sampling, and generation | Exact x86 profiles can use AVX2/FMA integer dots, weight reuse, and a persistent worker pool more predictably than the current portable lowering. |
| Gated DeltaNet recurrence | F32 recurrent state used by Qwen3.5 | Convolution, projections, state ownership, attention, tokenizer, and generation | The Java recurrence is correct and vectorized, but the exact Qwen3.5 0.8B production profile needs the narrow FFM recurrence to meet its TTFT policy. |

The [backend contract](backend-native/README.md) is versioned, checksum-verified, optional, and
replaceable. Every native-selected model also runs a library-default Java correctness smoke. This
turns shim removal into a performance qualification problem instead of an architecture rewrite.

## What Java has already closed

- Read-only mapped GGUF, sharded Safetensors, and the Needle 2 CACT container execute in-process.
- Q4/Q5/Q8, K-quants, BF16, MXFP4, and CQ2/CQ4 have Java storage and compute paths.
- Dense and hybrid transformer graphs, MoE expert loading, recurrent state, embeddings, tool
  calling, and framework adapters remain Java code.
- Standard MXFP4 weights execute without expansion; the retained W4A8 kernel was 12.73x faster
  than exact packed decode for the isolated GPT-OSS routed-layer shape while preserving its stated
  numerical gate.
- Java-authored TornadoVM kernels passed complete Qwen output gates on NVIDIA A16 and A40 devices.
  The A40 gate improved warm prefill 4.40x and median decode 1.32x over the same host's Vector API
  path.
- Java 25 FFM is sufficient for a small, explicit, testable ABI. FFM itself is not the performance
  gap; it is the containment mechanism while the remaining kernels are replaced.

Evidence is indexed in [the experiment journal](benchmark-results/README.md), with raw JVM,
artifact, hardware, and oracle details in the linked reports.

## Demonstrated JVM gaps

| Gap | Evidence | Useful JVM/runtime capability |
| --- | --- | --- |
| Packed integer dot-product lowering is runtime- and architecture-dependent. | The [pairwise baseline](https://github.com/integrallis/vectors/blob/main/vectors-bench/jmh-results/vector-api-pairwise-jvm-baseline.md) shows Graal lowering expanded graphs to `VPMADDWD` and `VPMADDUBSW` while HotSpot C2 did not. It also records no equivalent AArch64 `SDOT`/`UDOT` lowering. | Direct, portable Vector API operations for the signed/unsigned packed dot products used by quantized inference, with consistent x86 and AArch64 lowering. |
| Correct vector code can lose to a simpler kernel because scalar replacement and inlining are fragile. | The [Graal Q4 gate](https://github.com/integrallis/vectors/blob/main/vectors-bench/jmh-results/graal-q4-short-pairwise.md) records a retained profile only after a larger inlining limit; accumulator arrays and aggressive global settings caused allocation or bimodal regressions. | Predictable scalar replacement for small vector aggregates and local, stable compiler control that does not require process-wide tuning. |
| Mapped-memory code generation changes across JDK releases and quantization formats. | The [Java 25/26 gate](https://github.com/integrallis/vectors/blob/main/vectors-bench/jmh-results/mapped-kquant-jdk25-jdk26.md) found format-specific gains, regressions, and a Java 25 Q4_K kernel faster than Java 26 on the same host. | Stable optimization of long-offset mapped `MemorySegment` access inside vector loops, with generated-code diagnostics suitable for regression testing. |
| Java-authored GPU execution still needs a specialized launcher and pays large compilation/readiness costs. | The [A16 report](models-accelerator-bench/results/vultr-a16-2q-2026-08-29.md) required the TornadoVM launcher, retained 223 plans, and spent about 14 seconds in eager readiness. Maven dependencies alone could not discover the device drivers. | Standard device discovery from an ordinary Java launch, persistent ahead-of-time device code caches, explicit device-memory lifetime, and asynchronous transfer/event APIs. |
| Device compiler coverage is not yet portable across common inference graphs. | The isolated A16 integer reduction needed graph reshaping; attention passed its numeric test but more than doubled full-model prefill, and only the NVIDIA PTX envelope passed. | Cross-vendor lowering and profiling for packed integer reduction, reductions/softmax, and bounded dynamic shapes, with failures visible before a full model is deployed. |

These are measured gaps in the tested runtimes, not permanent language limitations. The same
reports also show cases where Java wins or where a proposed native/vector formulation should be
rejected.

## Open requests to JVM leadership

The request is not “make Java as fast as native code” in the abstract. Each item below can be
discussed, reproduced, and closed independently.

### JVM-AI-1: portable packed dot products

**Audience:** Vector API, HotSpot C2, and Graal compiler teams.

**Observed result:** the expanded Vector API graph reached `VPMADDWD` and `VPMADDUBSW` on the tested
Graal development build but not on HotSpot C2. The same report found no AArch64 `SDOT`/`UDOT`
lowering. Even after successful instruction selection, the complete Q8 block remained slower than
the established widening kernel.

**Request:** provide direct or reliably recognized operations for signed-short pairwise
multiply-add and unsigned-byte by signed-byte dot products, with documented overflow/saturation
semantics suitable for exact quantized inference.

**Acceptance gate:** the existing pairwise microbenchmarks must lower without internal APIs or
global compiler flags on x86-64 and AArch64, allocate zero, and then pass the complete Q4/Q8 block
plus full-model hash gates. The full block—not the presence of an opcode—is the success criterion.

### JVM-AI-2: predictable local optimization of vector helpers

**Audience:** HotSpot C2 and Graal compiler teams.

**Observed result:** a production Q4_0 graph improved materially only after raising Graal's global
inlining limit. Vector accumulator arrays prevented scalar replacement; more aggressive global
settings caused large allocation or bimodal full-model performance, and `ForceInline` from an
internal package did not solve it.

**Request:** make small fixed-size vector aggregates scalar-replace reliably and provide a stable,
local compiler-control mechanism for a hot call chain without widening the whole process's graph
budget.

**Acceptance gate:** the retained Q4_0 benchmark should select the pairwise graph with no normalized
allocation, no internal annotations, no process-wide inlining override, and stable three-fork plus
full-model results.

### JVM-AI-3: stable Vector API over mapped MemorySegments

**Audience:** HotSpot, Graal, FFM, and JDK distribution performance teams.

**Observed result:** long-offset mapped access improved some K-quant formats and regressed others;
the measured Java 25 Q4_K kernel was faster than Java 26 on the same host. The safe production
policy therefore branches on JDK version and quantization format.

**Request:** treat read-only mapped `MemorySegment` access with long offsets as a first-class vector
loop, publish enough generated-code diagnostics to identify regressions, and run representative
mapped quantized kernels as cross-release tests.

**Acceptance gate:** Q4_K, Q5_K, Q6_K, mixed-group, vocabulary, and batch-32 gates must preserve
exact output and avoid a release-to-release regression outside their recorded confidence intervals.

### JVM-AI-4: ordinary-launch Java accelerator discovery

**Audience:** JDK distribution, TornadoVM, Project Babylon/code-reflection, and accelerator-runtime
teams.

**Observed result:** Java-authored PTX kernels passed two NVIDIA full-model gates, but an ordinary
`java` launch cannot discover the TornadoVM device runtime from Maven dependencies alone. Users need
a specialized launcher or generated argument file.

**Request:** define a supported way for an application launched with ordinary Java tooling to
discover an installed device backend, compile Java-authored kernels, and fall back to CPU without
changing its public model API.

**Acceptance gate:** the existing public `TornadoBackend.open` full-model test must run from a
standard application launch on a clean host, select the qualified device, or report a structured
CPU fallback reason. No external inference server or handwritten CUDA/Rust kernel may be added.

### JVM-AI-5: persistent device code and explicit memory lifetime

**Audience:** TornadoVM and future standard Java accelerator API teams.

**Observed result:** the qualified Qwen path retained 223 plans and spent about 14 seconds in eager
readiness because compiled PTX lives only with an execution plan. Fixed shapes removed prompt-tail
recompilation but did not remove cold compilation.

**Request:** support a validated device-code cache keyed by application code, device architecture,
driver/runtime, and shape; expose persistent device-memory ownership plus asynchronous transfer and
event primitives.

**Acceptance gate:** a second process on the same qualified host should reuse validated code, keep
the first visible TTFT within 10% of the warm request, preserve exact CPU/GPU tokens, and invalidate
the cache when any key changes.

### JVM-AI-6: cross-vendor device compiler conformance

**Audience:** TornadoVM PTX/OpenCL/SPIR-V backend maintainers and device vendors.

**Observed result:** one nested integer reduction passed ordinary JVM tests but failed PTX; a
flattened fallback was correct but slower; attention passed an isolated numerical gate and then
more than doubled full-model prefill. AMD and Intel devices remain unqualified.

**Request:** add inference-shaped conformance cases for packed integer reductions, bounded dynamic
shapes, reduction/softmax, and persistent mapped weights across PTX, OpenCL, and SPIR-V.

**Acceptance gate:** isolated output tolerance and complete model token parity must both pass. A
backend is not qualified by kernel compilation alone, and a numerically correct full-model
regression must remain a rejected result.

### JVM-AI-7: low-precision types and conversions, gated by evidence

**Audience:** Vector API, compiler, and hardware-vendor teams.

**Observed result:** Java already executes BF16, MXFP4, and quantized integer paths, but it implements
format decoding, scale application, and activation conversion manually. The Java W4A8 experiment
shows this can be viable; it does not yet prove that a new API is faster.

**Request:** evaluate explicit BF16/FP8 and packed low-precision conversion or dot-product support
against the retained MXFP4 and GPT-OSS expert shapes before standardizing an abstraction.

**Acceptance gate:** a proposed API must preserve the independent decoding fixture and numerical
floors, improve both official expert shapes in a three-fork gate, and improve the official-checkpoint
screen. Convenience without a measured end-to-end gain is insufficient.

### JVM-AI-8: common observability and a distribution qualification lane

**Audience:** OpenJDK distributions, GraalVM, TornadoVM, JFR/JMC, and Java ecosystem CI providers.

**Observed result:** the investigations combine JMH, generated assembly, JFR, Linux counters,
device timings, and complete model reports manually. Compiler and JDK changes can improve one shape
while regressing another.

**Request:** expose JFR/JMC events that connect Java call sites with compilation decisions, vector
width/instructions, device compilation, transfers, and kernel execution; provide a reproducible
qualification lane across current HotSpot, GraalVM, and accelerator-enabled distributions.

**Acceptance gate:** one command should emit a machine-readable environment plus evidence bundle
for the existing kernel and model gates, without capturing secrets. A distribution claim should be
accepted only when the corresponding output-equivalence and performance policy passes on that
runtime.

## Distribution-specific evidence today

| Runtime or distribution | What the retained experiments establish | Open issue |
| --- | --- | --- |
| Temurin/OpenJDK 25 HotSpot C2 | Portable widened and register-tiled Vector API kernels; correct mapped FFM execution; current production baseline | Missing pairwise packed-dot lowering in the tested build; some useful graphs require manual expansion and hardware-scoped dispatch. |
| Temurin/OpenJDK 26 HotSpot C2 | Long-offset mapped access improves selected K-quant workloads and complete MiniCPM metrics | Benefits are format-specific, and the measured Q4_K kernel remained slower than Java 25; release regressions need an inference-shaped lane. |
| GraalVM CE 25 development build | The tested compiler fix emits x86 pairwise instructions and enables a qualified Q4_0 profile | The complete Q8 pairwise block still loses, local inlining control is fragile, and no equivalent AArch64 dot lowering was established. |
| TornadoVM 5.2 on JDK 25 | Java-authored Q4 projection and decode kernels preserve full-model output and accelerate qualified NVIDIA A16/A40 paths | Specialized launch, cold plan compilation, graph-size constraints, and absent AMD/Intel gates limit ordinary application use. |
| Java 25 FFM with the Models-owned ABI | Safely contains optional native kernels behind a versioned, checksum-verified boundary | This is an effective transition seam, not the target execution model; qualified Java performance must replace each selected capability. |

## Hypotheses worth testing

- Low-precision vector conversions and arithmetic for BF16, FP8, MXFP4, and related formats may
  reduce decode/unpack overhead, but each needs a complete quantization-exact benchmark. Java's
  current MXFP4 result shows that a custom Java kernel can already be viable.
- A JVM-visible accelerator cache keyed by bytecode, device, driver, and fixed shape may remove
  first-user compilation without asking every library to invent a plan repository.
- Better virtual-thread or structured-worker affinity is not automatically a kernel improvement.
  The existing persistent executors should be replaced only when hardware-counter and
  full-model evidence shows scheduler overhead is material.
- Unified CPU/device profiling that connects Java frames, generated instructions, transfers, and
  kernels would shorten qualification loops, but it should not substitute for output equivalence.

## Native-removal experiment order

1. **Qwen3.5 recurrence:** compare current Java SIMD and ABI 5 recurrence in counterbalanced
   full-model runs on x86 and Apple Silicon; inspect generated code before adding another Java
   formulation. Remove the native recommendation if Java clears the existing 0.8B policy.
2. **Qwen2.5 3B quantized decode:** isolate the selected Q4_K/Q5_K/Q6_K projection shapes, test
   current HotSpot and Graal builds, then rerun the 27-case workload. The goal is the current
   Rust/FFM tier with identical output hashes, not a synthetic microbenchmark win.
3. **AArch64 packed dots:** run the same primitive and model gates on Apple M1 through M4 and an
   ARM server. Add architecture dispatch only from retained evidence.
4. **GPU launch simplification:** test whether ordinary Java launch and persistent compiled plans
   can reproduce the qualified A16/A40 path; then add AMD and Intel device gates.
5. **GPT-OSS MXFP4 experts:** profile the official checkpoint's measured expert hot loop, retain
   only Java candidates that beat the current W4A8 path, and repeat generation plus Spring tool
   oracles before catalog consideration.

Success is not the deletion of Rust at any cost. Success is a portable Java path that meets the
same correctness and performance policy, after which the corresponding native capability and
catalog recommendation can be removed without changing the public Models API.
