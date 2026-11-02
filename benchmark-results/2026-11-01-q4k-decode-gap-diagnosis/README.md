# Q4_K decode-gap diagnosis — what the repack hypothesis was worth

Date: 2026-11-01. Box: RunPod pod `xj8g254b92wgy6`, 16 vCPU of an **AMD EPYC 7702P** (Zen 2,
64 cores on the host, 1 NUMA node), **AVX2, no AVX-512** (`avx512f` count 0 in `/proc/cpuinfo`).
`nproc` = 16; `/proc/loadavg` reports the host, not our cpuset, so it is recorded and not gated on.
Raw artifacts and the exact scripts that produced every number are in `raw/`.

## 1. The repack hypothesis is dead — measured null

Earlier in this campaign I named ggml's repacked Q4_K GEMV (`block_q4_Kx8` +
`ggml_gemv_q4_K_8x8_q8_K`) as *the* cause of our decode gap, and proposed porting it into Vectors.
**That claim does not survive measurement and is withdrawn.**

`GGML_CPU_REPACK` is a CMake option (`llama.cpp ggml/CMakeLists.txt:152`), not an env toggle, so the
A/B is two builds of one commit (`785322d`), same flags otherwise, same model, 16 threads:

| arm | build | tg128 (tok/s) |
|---|---|---|
| repack **ON** (pass 1) | `b/` | 21.54 ± 0.33 |
| repack **OFF** | `bnr/` `-DGGML_CPU_REPACK=OFF` | 20.16 ± 2.89 |
| repack **ON** (re-confirm) | `b/` | 19.75 ± 1.25 |

All three overlap. Run-to-run drift on the *same binary* is ~9% (21.54 vs 19.75), which bounds what
this box can resolve. Repack is worth **nothing measurable** here.

It was genuinely active in the ON arm, not silently skipped: `repack.cpp:5011` selects
`q4_K_8x8_q8_K` for `GGML_TYPE_Q4_K` when `ggml_cpu_has_avx2()` and `ne[1] % 8 == 0`, and the x86
implementation at `arch/x86/repack.cpp:1464` is a real AVX2 kernel, not the `_generic` fallback.

**Consequence:** llama.cpp's *ordinary* per-row Q4_K dot reaches the same decode throughput as its
repacked 8-row-interleaved GEMV on this hardware. Row-major per-row traversal is therefore **not**
the thing separating us from it, and porting `block_q4_Kx8` into Vectors would buy nothing. The
earlier "named cause with reference implementation" was a code-reading inference that the
measurement contradicts.

## 2. What llama.cpp actually achieves

Weight traffic per decoded token, from file size × tg128 (`-p 0 -n 128 -t 16 -r 3`):

| model | bytes | tg128 | implied weight traffic |
|---|---|---|---|
| `mathstral-7B-v0.1-Q4_K_S.gguf` | 4,144,746,304 | 19.42 | **80.5 GB/s** |
| `mathstral` Q4_K_M (`m.gguf`) | 4,372,811,584 | 20.35 | **89.0 GB/s** |

Throughput is near-flat while bytes/token rise 5.5%, which is the signature of a
bandwidth-dominated regime rather than a compute-bound one.

### A failed measurement, recorded as failed

My own read-bandwidth probe (`raw/roofline.sh`, `bw.c`) reported a best of **30.5 GB/s** on 16
threads — *below* the 89 GB/s the same box demonstrably sustains while decoding. The probe is
therefore wrong (an un-vectorized float reduction, not a bandwidth test), and its number is **not a
result**. What stands is the 89.0 GB/s lower bound established by the decode runs themselves.
The real ceiling for this socket is not established by anything measured here.

## 3. The gap, measured in absolute terms on one box

Both sides, same model file, same 16 threads, same pod, nothing shared with a comparator engine:

| engine | decode (tok/s) | implied weight traffic |
|---|---|---|
| llama.cpp `tg128` (build `785322d`) | 19.42 | 80.5 GB/s |
| ours, `--backend rust-ffm` | 8.97 (p50 of 27 cases) | 37.2 GB/s |

That is a **2.17x** decode deficit, and it reproduces the `decodeThroughputRatio = 0.4769`
previously measured against ollama -- but without a comparator engine in the loop, so it is a
property of our decode path and not of a protocol or a cold start. Per-case decode figures and the
full report are in `raw/candidate.log` and `raw/candidate.json`.

The route was confirmed **at runtime**, not only from source: the report's own diagnostics carry
`native-quantized-decode = true` and `native-kernel-decode-threads = 16`. The native kernel served
decode. Jars: `backend-native-0.5.10`, `vectors-core-0.1.32`.

Prefill is separately and much worse off: `p50PrefillTokensPerSecond = 22.79`, giving TTFT of
**6.4-9.1 s** per case on these RAG prompts. That is not what the decode gate measures, but it
dominates `endToEndLatencyRatio`, so it is recorded and not analysed here.

## 4. The route was verified from source, not assumed

Our candidate arm runs `--backend rust-ffm` (`scripts/fleet/two-arm-pod.sh:36`). Checked, with
line references, that this reaches the fast kernel at batch = 1:

- `LlamaForwardPass.java:3138` `matmulDispatch` tests `batchedMatrixKernel.isEligible(type, 1, …)`,
  so decode (batch = 1) can dispatch to the native kernel — it is not prefill-only.
- `RustGgufBatchedMatrixKernel.java:529` advertises `Q4_K`; `:541` `isEligible` adds no batch-size
  restriction for `Q4_K` (only `Q5_1` and `BF16` are range-limited). So Q4_K **is** eligible at
  batch = 1.
- `RustFfmBackend.java:95` passes a real `RustGgufBatchedMatrixKernel` (not
  `GgufBatchedMatrixKernel.none()`) into `PureJavaBackend.load`.
- `model-kernels/src/lib.rs:4150` `compute_q4_k_batched_row_range_avx2` delegates batch = 1 to
  `dot_q4_k_q8_k_row_avx2` (`:6010`, impl at `:6060`), whose instruction mix includes
  `_mm256_maddubs_epi16` ×2 and `_mm256_madd_epi16` ×5 — **the same instruction pair ggml's Q4_K
  GEMV is built on.**
- Ordered (determinism) mode is **off** for this model: `requireSequentialQ4KReduction()` has
  exactly one caller, `FlintForwardPass.java:142`. mathstral is a llama-arch model, so it runs the
  vector-accumulation path, not the sequential one.
- The per-block counters in the Rust hot loop (`record_k_quant_weight_block_decode`, `lib.rs:47`)
  are `#[cfg(test)]` and compile out of release. Not a hot-path cost.

So the decode gap is **not** a missing kernel, **not** a mis-wired route, **not** inferior
instruction selection, and **not** the determinism mode. Those four were plausible and are now
excluded.

## 5. Where our decode time actually goes (JFR, 454 s, 27 cases)

`jdk.ExecutionSample` **and** `jdk.NativeMethodSample` both at 1 ms (ExecutionSample alone omits
threads in native state and would have made the Rust kernels look free). 289,013 native samples vs
91,094 Java samples -- **76% of all samples are in native state**, which is why the pairing matters.

On the `main` thread alone -- one thread, one sampler cadence, so the split is meaningful:

| | samples | share of main thread |
|---|---|---|
| native quantized matmul kernels | 289,011 | **79.1%** |
| Java | 76,311 | 20.9% |

Native time is essentially *only* the matmuls -- 287,418 of 289,013 samples (99.4%):

| frame | samples |
|---|---|
| `invokeGrouped` / `quantizedF32GroupedBatchedMatmul` (dual + triple projections) | 189,584 |
| `multiply` / `invokeBatched`, of which: | 97,834 |
| &nbsp;&nbsp;`q4_KF32BatchedMatmul` | 86,613 |
| &nbsp;&nbsp;`q5_KF32BatchedMatmul` | 9,646 |
| &nbsp;&nbsp;`q6_KF32BatchedMatmul` | 1,575 |
| mmap page-in (`UnixFileDispatcherImpl.read0`) | 1,241 |

There is **no** orchestration, attention-dispatch or tokenizer hotspot. All 175,970 `jdk.ThreadPark`
events are on `vectors-gguf-*` workers and **none on `main`** -- about 13 parks per token per
thread, far too few to matter.

The Java 21% is dominated by two things:

| frame | samples | share of Java samples |
|---|---|---|
| `TensorOps.swiGlu` (from `LlamaForwardPass.activateFfn`) | 39,012 | 42.8% |
| `PanamaVectorUtilSupport.matVecDotExact` | 18,028 | 19.8% |
| `PanamaVectorUtilSupport.addWeightedRowsInPlace` | 15,255 | 16.7% |
| `TensorOps.softmax` | 7,267 | 8.0% |

`swiGlu` is a single-threaded elementwise activation on `main`. `matVecDotExact` /
`batchDotProductExact` are the **`Exact` determinism variants**, which an earlier measurement
already prices at 9-12% with a quality justification that did not reproduce -- see
`activation-determinism-x86`.

### What this implies for the gate, arithmetically

Taking the 79.1/20.9 split and the measured 2.17x kernel bandwidth deficit: bringing only the
kernels to llama.cpp's throughput gives `79.1/2.17 + 20.9 = 57.3%` of current decode time, i.e.
**1.75x -> ~15.7 tok/s -> ratio ~0.81**, which lands exactly on the 0.80 `decodeThroughputRatio`
boundary with no margin. The Java 21% is where the margin would come from.

So the kernel is the larger half and is necessary; the Java work is the margin. Neither alone is
comfortably sufficient. This is a projection from a sample-share decomposition, **not** a measured
speedup, and is labelled as such.

## 6. A probe that failed, reported as failed

To split kernel throughput from per-matmul orchestration I streamed 2 GiB of **valid** Q4_K blocks
through the native kernel as one call versus 224 calls (one 7B token's matmul count) at 2/4/8/12/16
threads (`raw/Q4KThroughputProbe.java`). It is **inconclusive and no number from it is used.** At
fixed configuration across repeats the one-call arm spans 28-47 GB/s and the 224-call arm spans
3.2-48.7 GB/s -- a 15x spread on a re-run of the identical point. A first version additionally
filled weights with random bytes, producing NaN/denormal fp16 scales and measuring the FPU's
garbage handling (`checksum=NaN`); that was fixed, and the instability survived the fix.

Both logs are kept as `raw/q4k-probe-*-INCONCLUSIVE.log`. The cause is almost certainly a shared
host: this is 16 vCPU of a 64-core box, host load averaged 6-20 throughout, and a resident
`ollama serve` sat alongside. **An ordering of thread counts must not be taken from these runs** --
an earlier single-JVM pass appeared to show a clean 8-thread peak at 55.2 GB/s and it did not
replicate once each thread count got its own JVM.

## 7. A real difference that is not yet priced

The Java/Panama path (used whenever the native kernel is absent) cannot express these instructions.
Panama has no `VPMADDUBSW`/`VPMADDWD`, so `PanamaVectorUtilSupport.java:2404`
(`multiplyAddUnsignedSignedBytes128`) and `:2388` (`multiplyAddSignedShorts256`) **emulate** each
one with an even/odd `rearrange` pair plus widening multiplies. Separately, the Panama Q4_K GEMV
(`:2455` `ggufQ4_KQ8_KMatVecDot`, inner dot at `:3989`) does a `reduceLanes(ADD)` horizontal
reduction per 16 lanes inside the innermost loop, and loads each 32-byte packed chunk twice — once
for the low-nibble group, once for the high-nibble group.

Both are real inefficiencies in the *Java* kernel. Neither is a candidate explanation for the
two-arm numbers, because the two-arm candidate arm runs `rust-ffm`, not the Panama path. They are
only worth fixing for the pure-Java backend, and they are **unmeasured** — no claim is made about
their size.

## 8. ANSWERED: the cause, and the fix

Full detail and reproduce script in **`kernel-ab/RESULTS.md`**. Summary:

A head-to-head of ggml's `vec_dot_q4_K_q8_K` against our `dot_q4_k_q8_k_row_avx2` in **one process,
on identical weights, single-threaded and interleaved** gives a flat **2.0-2.2x** across working
sets of 1.1 MiB, 9 MiB and 36 MiB. Size-independence rules out cache and memory effects;
single-threading rules out scheduling. It matches the 2.17x measured end-to-end, so the entire
decode gap was inside the single-row kernel.

**The cause was per-block overhead around an already-correct inner loop.** Per 256-element block we
performed the branchy six-bit `qk_scale`/`qk_min` field extraction **sixteen times** (eight in the
pair loop, eight more in a **scalar** 8-iteration mins loop), and broadcast each scale from a GPR.
ggml unpacks all sixteen fields **once** from three u32 words, broadcasts with `shuffle_epi8` from a
register, and computes the mins term as a single `hadd_epi16` + `madd_epi16` over the precomputed
`bsums`. The inner dot -- 4 weight loads, 8 `maddubs`, 8 `madd_epi16` -- was already identical.

This is the same defect class the earlier tile port already priced at 40% ("the dominant cost was
redundant work per batch group, not instruction selection"). The GEMV never received that fix.

**Fixed**, via one shared `qk_unpack_scales_and_mins` helper (ported from ggml, MIT, attribution in
the source). Measured before/after in one session with ggml as the in-run control, 4096 rows:

| type | before | after | our speedup |
|---|---|---|---|
| **Q4_K** | 1.86 GB/s (0.454x ggml) | **3.66 GB/s (0.858x)** | **1.97x** |
| Q5_K | 2.84 GB/s (0.700x) | 3.15 GB/s (0.758x) | 1.11x |

**Numerically exact**: the harness's agreement figures are unchanged to all printed digits
(1.471e-08 at this size) because the only reordering is of integer sums, which is associative with
no lane overflow. The crate's 43 tests pass, including the bit-for-bit ordered-reduction ones.

This covers the decode route for both entries: `compute_q4_k_batched_row_range_avx2:4162` delegates
`batch_size == 1` to this function, and so does the grouped path (`:4682`) -- together the 65.6%
grouped and 88.5% of the single-matmul native time from section 5.

### Still open

- **End-to-end tok/s has not yet been re-measured** with the fix. Section 5's decomposition
  projects 79.1% of decode time shrinking by ~1.97x, which would give roughly `79.1/1.97 + 20.9 =
  61%` of current time, i.e. ~1.64x -> ~14.7 tok/s -> ratio ~0.76. That is a projection; the
  measured number is what decides the gate, and it is pending.
- The remaining ~14% to ggml on Q4_K is the GPR scale broadcast (ggml uses `shuffle_epi8` from a
  register). Not yet attempted.
- **Prefill at 22.79 tok/s** -- larger than the decode gap in wall-clock terms and untouched.
- Q5_K's two horizontal reductions per block, and Q6_K, were not examined.
