# Granite 4.1 3B Q4_K_M decode: where the last 15% against llama.cpp goes

Measured on the same AWS c7a.4xlarge (16 vCPU, AMD EPYC 9R14), Models fcd38d3, tuned profile
(16 native workers, 8 for single-token projections, native quantized decode, native grouped
attention, batched Java attention): **26.02 tok/s** against llama.cpp b10012 **30.42 tok/s** and
Ollama v0.32.0 **29.55 tok/s** — 38.4 ms per token against 32.9 ms, a 5.6 ms gap. Earlier
JFR work put the native matmuls at bandwidth parity with llama.cpp, so the gap is in what the
runtime does around them. This note lists what ggml's CPU backend (llama.cpp b10012,
`ggml/src/ggml-cpu/`) does differently, read from its source, and the measurements that decide
which differences matter. Nothing here is a result until the measurement is in.

## What the token costs, from the code

Per layer on the single-token path (`LlamaForwardPass.forwardSequenceInternal`): 5 native
dispatches — fused q/k/v, grouped attention, o_proj, fused gate/up, down — plus Java glue on
the main thread: two `rmsNorm`, q/k head norms and RoPE over 48 heads, two residual FMAs,
swiGlu, and the Q8 activation quantization inside each dispatch. 40 layers plus the tied
Q6_K LM head: **201 downcalls per token** and roughly 200 single-threaded glue segments between
them. Weight bytes per token: 1.52 GB Q4_K (q, k, o, gate, up) + 0.58 GB Q6_K (v, down, the
211 MB embedding that serves as the LM head); every type runs in the Rust kernel.

## Differences against ggml's CPU backend (read from source)

1. **Thread pool never parks inside a token.** `ggml_graph_compute_poll_for_work` spins
   `1024 * 128 * poll` relax rounds before sleeping, with `poll = 50` by default (`ggml.c`), i.e.
   ~6.5 M rounds — far longer than a token. Between ops threads meet at `ggml_barrier` (atomic
   counter + `cpu_relax`), never a futex. Ours: `WORKER_SPIN_ITERS = 4_000` rounds, then park on
   a condvar; every dispatch that follows a glue segment longer than the spin budget is a futex
   wake for each worker. Test: `JMODELS_KERNELS_WORKER_SPIN` (branch `perf/worker-poll-budget`),
   interleaved A/B 4,000 vs 6,553,600 with `perf stat` context switches.
2. **Dynamic row chunking with work stealing.** `ggml_compute_forward_mul_mat` splits the
   output rows into chunks of 16 (64 for large operands) and threads claim chunks through an
   atomic `current_chunk`; a straggler thread only loses its last chunk. Ours:
   `execute_matrix_job_partition` gives worker `i` the static range
   `[rows*i/n, rows*(i+1)/n)`, so one delayed vCPU (SMT sibling, noisy neighbour, the caller
   thread arriving late from its glue) stalls the whole dispatch. Test: chunk-stealing mode in
   the Rust pool behind an env knob, same A/B protocol.
3. **Parallel activation quantization.** ggml converts `src1` to `q8_K` inside the op with all
   threads (`from_float` over row blocks split by `ith/nth`) before the barrier; ours quantizes
   on the main thread in Java before the downcall. Small vectors (2,560 and 8,192 floats), so
   this is microseconds per dispatch; it matters only as part of a fused native layer.
4. **Small ops are graph ops.** rms_norm, rope, add, glu run as ggml ops with `n_tasks =
   n_threads` where the tensor is large enough, inside the same non-parking pool; ours run on
   the main thread while workers idle. For 2,560-wide vectors these are microseconds each, but
   there are ~200 of them per token; a fused per-layer native step removes them from the Java
   side and removes 4 of the 5 downcalls per layer.
5. **Everything else** — Q4_K×Q8_K AVX2 dot structure (`_mm256_maddubs_epi16` /
   `_mm256_madd_epi16` with folded scales), KV layout, attention — is already at parity or
   native on our side by measurement (native projections at bandwidth parity; native grouped
   attention landed at f6252cc).

## Measurements

Reference host: Hetzner cpx62 (16 shared vCPU), interleaved A/B, decode tok/s from the RAG
harness models arm (`--iterations 1`, tuned profile), `perf stat -e context-switches`. Shared
vCPUs make absolute numbers noisy; interleaving makes the comparison usable. Anything that wins
here is confirmed on a c7a.4xlarge before it changes a certified number.

(Results appended below as they land.)

### Result 1 — worker spin budget (reference host, interleaved, 2026-09-16T16:20Z)

| run | spin rounds | decode tok/s | p50 TTFT | p50 e2e | context switches (process) |
|---|---:|---:|---:|---:|---:|
| r1 | 4,000 | 15.92 | 1,623 ms | 3,073 ms | 760,841 |
| r1 | 6,553,600 | 23.98 | 1,269 ms | 2,248 ms | 177,340 |
| r2 | 4,000 | 15.61 | 1,658 ms | 3,143 ms | 773,128 |

Same binary, same model, same 16/8 worker profile; only `JMODELS_KERNELS_WORKER_SPIN` differs.
The parking workers cost ~4.3× the context switches and a third of the decode rate on this
shared-vCPU host. Difference 1 is real and is the first thing to productise: a per-token
non-parking regime (poll budget sized to a token, park between requests), on the Java stage
pool as well as the Rust pool. Remaining rounds and the decode-thread and chunk-stealing A/Bs
follow.

Rounds 2 and 3 (same protocol): 15.61 → 26.02 and 16.60 → 24.99 tok/s; context switches
773k → 183k and 823k → 185k. Three of three rounds agree: **+55% decode on the shared host from
the poll budget alone.** Productised as a time-based budget (`DEFAULT_POLL_NANOS` = 25 ms in the
Rust pool, `jmodels_kernels_context_set_poll_nanos` behind capability bit 21, Java property
`models.native.kernels.pollMillis`); the pure-Java executor in vectors-core gets the same
spin-then-park barrier. Confirmation on a c7a.4xlarge follows before any certified number moves.

### Result 2 — decode workers 8 vs 16 under the token-sized budget (same host)

dt8: 25.12 / 25.87 / 22.92 tok/s; dt16: 25.03 / 20.45 / 26.78. No consistent direction on 16
shared vCPUs (the extra eight are SMT siblings of busy cores on this instance); the tuned
profile keeps 8 workers on single-token projections. Chunk stealing (Difference 2) is measured
next, then the productised default against the old regime, then the pure-Java executor with the
same barrier (`vectors.gguf.pollMillis`, branch `perf/gguf-executor-poll-budget` in vectors).

### Result 3 — main-thread JFR profile under the token-sized budget (same host, 24.63 tok/s)

`settings=profile`, context 2048, 1 warm-up + 2 iterations (1,950 prefilled tokens, 440
decoded), execution and native samples on the main thread combined:

| where the main thread is                                   | samples | ≈ seconds |
|------------------------------------------------------------|--------:|----------:|
| native matmul downcalls (`invokeGrouped` + `invokeBatched`) |   2,351 |      47.0 |
| `swiGlu` (Java, single-threaded, mostly the prefill batch) |     117 |       1.2 |
| native grouped attention (critical downcall, sampled as Java)|    116 |       1.2 |
| `MemorySegment.copy` + session release around downcalls    |      73 |       0.7 |
| `rope` + `rmsNorm` + residual `addScaledInPlace`           |      64 |       0.6 |
| dispatch glue (`dualBatchedMatmulDispatch`, `multiplyTriple/Dual`, `invokeBatched` Java side) |  89 |  0.9 |
| bench CLI SHA-256 of the model file (not runtime)          |      97 |       1.0 |

The kernels own 87% of the main thread; everything ggml runs "as graph ops" (Difference 4)
is about 6% here, and half of that is `swiGlu` over the whole prefill batch on one thread.
Difference 4 is therefore worth at most a few percent of decode on this host; the remaining
decode gap is inside the matmul, not around it.

**Prefill is the larger gap.** Certified c7a.4xlarge numbers (same bundle as the decode
figures above): Models 144.7 prefill tok/s against llama.cpp 280.6 and Ollama 596.4 —
1.9× and 4.1×. Prefill is compute-bound, so that ratio points at the dot-product ISA, not
the pool: c7a is Zen 4 (AVX-512, VNNI). Checked next.

## The prefill gap, read from source (b10012 `ggml-cpu/repack.cpp`, `arch/x86/repack.cpp`, `arch/x86/quants.c`)

6. **Repacked, register-tiled GEMM for prefill.** With `GGML_CPU_REPACK` (default on), every
   Q4_K weight tensor whose row count is a multiple of 8 is repacked at load into
   `block_q4_Kx8` (eight rows interleaved per super-block) and multi-token matmuls go through
   `ggml_gemm_q4_K_8x8_q8_K`: four activation rows (`block_q8_Kx4`) × eight or sixteen weight
   rows per register tile, accumulators held in `__m256`/`__m512` across the whole K loop,
   scales and mins folded per tile. Each weight load feeds 4 activation rows and each
   activation load feeds 8–16 weight rows. Single-token decode uses `ggml_gemv_q4_K_8x8_q8_K`.
   Ours (`compute_q4_k_batched_row_range_avx2`): one weight row at a time; the block's nibbles
   are decoded once into 8 registers and reused across the batch (good), but every batch column
   reloads its 8 activation vectors from L2 and does its own `maddubs`/`madd`/`cvt`/`fmadd`
   chain, then the row is horizontally reduced per column. Loads per MAC are ~4–8× ggml's, and
   the float conversion happens per (row, column, block) instead of per tile. Q6_K (`v`,
   `down`, LM head) has the same shape.
7. **ISA.** Every Rust kernel is `#[target_feature(enable = "avx2,fma,f16c")]`; there is no
   AVX-512 or VNNI path. ggml built with `GGML_NATIVE` (the certified arm) on c7a (Zen 4:
   AVX-512 F/BW/DQ/VL, VNNI, BF16) takes the 512-bit tile body in the repacked GEMM and
   `_mm256_dpbusd_epi32` (VNNI) for the `maddubs`+`madd` pair in `vec_dot`. Zen 4 double-pumps
   512-bit ops, so the win is fewer instructions and half the loads, not 2× ALU throughput.

Prefill is compute-bound (Result 3), so 6 and 7 are the candidates for the 1.9× prefill gap
and, through the LM head and `gemv` layout, a slice of the decode gap. Decided by measurement:
llama-bench (pp512 / tg64) on the reference host across a build matrix — native+repack,
AVX2+repack, native without repack, AVX2 without repack — separates the tiling gain from the
ISA gain before any kernel is written (`/opt/ref-llamabench.sh`, results below when they land).

### Result 4 — chunk stealing × decode threads (reference host, interleaved, 3 rounds)

| config        | r1    | r2    | r3    |
|---------------|------:|------:|------:|
| chunk 0, dt 8 | 28.28 | 25.08 | 22.90 |
| chunk 16, dt 8| 24.67 | 21.43 | 23.28 |
| chunk 16, dt16| 25.39 | 33.58 | (pending) |
| chunk 64, dt16| 23.17 | 33.63 | (pending) |

Spread inside one configuration (21–34 tok/s) exceeds any difference between configurations:
the shared host cannot decide Difference 2. Chunk stealing stays behind
`JMODELS_KERNELS_CHUNK_ROWS` (off by default) until a c7a run says otherwise. Context switches
rise with stealing (≈250–370 k against ≈175 k), which is the atomic cursor contending across
SMT siblings — a mild reason to expect it not to win on 8 single-token workers.

### Result 5 — productised default (25 ms) against a 1 ms budget (reference host, interleaved)

| budget                       | r1    | r2    | r3    | context switches |
|------------------------------|------:|------:|------:|-----------------:|
| `pollMillis=25` (default)    | 22.35 | 24.88 | 27.46 | ≈175 k           |
| `pollMillis=1`               | 22.39 | 22.87 | 24.72 | ≈350 k           |

The productised path runs at the level the experiment reached (Result 1: 24–26 tok/s), so
the property, the ABI call and the diagnostics did not cost anything. A 1 ms budget is not the
old regime — it already outlasts the intra-token glue — which is why it sits close to the
default; the 4,000-round regime is the 15.6–16.6 tok/s row of Result 1. The remaining
context-switch halving comes from the inter-token gaps (sampling, tokenizer, harness) that
1 ms does not cover.

### Result 6 — pure-Java executor barrier (vectors `perf/gguf-executor-poll-budget`, same host)

Pure-Java backend (`--backend pure-java`, composite build against the vectors branch, jar
verified to carry `awaitAdvancePolling`), interleaved, 3 rounds, `vectors.gguf.pollMillis`:

| budget  | decode tok/s        | TTFT p50 ms           | context switches |
|---------|--------------------:|----------------------:|-----------------:|
| 25 ms   | 6.32 / 9.01 / 7.25  | 8,015 / 7,969 / 8,453 | 83 k – 243 k     |
| 0 (park)| 4.64 / 5.00 / 4.73  | 12,037 / 11,144 / 10,647 | 2.0 M – 2.2 M |

Same mechanism, larger effect: the Phaser parked at every one of the ~200 stage barriers per
token, so the pure-Java executor paid 2 M context switches per run. 3/3 rounds, both decode
(+36 % to +80 %) and prefill (TTFT −25 % to −34 %). The 25 ms default stands in vectors
(PR #72); confirmation on dedicated cores (c7a) is the remaining step before the number
changes anything certified.

### Result 7 — llama.cpp b10012 build matrix on the reference host (ISA × repack), llama-bench

Same GGUF, `-p 512 -n 64 -r 2`, two rounds; tok/s (round 1 / round 2). Reference host is a
shared 16-vCPU EPYC Genoa (AVX-512, VNNI, DDR5), so absolute numbers do not transfer to the
c7a; the ratios between builds are the measurement.

| build                          | pp512 t16     | pp512 t8      | tg64 t16      | tg64 t8       |
|--------------------------------|--------------:|--------------:|--------------:|--------------:|
| native (AVX-512+VNNI) + repack | 232.5 / 232.1 | 130.0 / 122.8 | 50.5 / 57.6   | 39.3 / 38.5   |
| AVX2 + repack                  | 180.2 / 183.1 | 99.5 / 100.8  | 56.7 / 47.9   | 42.8 / 36.1   |
| native, no repack              | 154.6 / 145.5 | 81.0 / 82.0   | 57.2 / 48.2   | 43.7 / 38.6   |
| AVX2, no repack                | 145.8 / 140.5 | 77.6 / 73.8   | 55.3 / 41.3   | 43.2 / 32.6   |

Prefill: the repacked register-tiled GEMM (Difference 6) is worth +26 % on AVX2 alone
(143 → 181), the ISA alone (Difference 7) only +5 % without tiling (143 → 150), and the
two together +62 % (143 → 232): the 512-bit tile body pays only once the data layout lets
it. Decode (tg64) is flat across all four builds — bandwidth-bound, as expected — so neither
difference is a decode lever for ggml.

**What this exposes about our decode.** On this host llama.cpp decodes at 50–57 tok/s with
16 threads and 39–43 with 8; our Rust arm sits at 22–27 (Results 1, 4, 5) — 2× — while on
the c7a the same comparison is 30.4 against 26.0. Our per-token time is ~40 ms on both hosts;
llama.cpp's follows the host's memory bandwidth (33 ms on c7a, 18 ms here). 2.1 GB of weights
per token at 40 ms is ~55 GB/s, which the c7a's 16-vCPU slice roughly caps but the Genoa
host does not. So our single-token projections are not bandwidth-bound: they are limited
per thread (8 decode workers at ~7 GB/s each), and Result 2 said 16 workers did not help
here either. That points at the kernel's per-thread throughput (nibble decode, per-block
scale handling, per-row float conversion) and the dispatch structure, not the pool. The
c7a confirmation host runs both harnesses on one box to settle whether this is
host-specific.

### Result 8 — symbolised whole-process profile of the Rust arm (reference host, unstripped library + JIT perf map)

`perf record -F 499` over the run (2,394 prefilled + 220 decoded tokens, decode 21.85 tok/s in
this run, composite build so the Java executor polls too). Share of all samples:

| where the CPUs are                                             | share  |
|----------------------------------------------------------------|-------:|
| Rust workers polling for the next job (`__rust_begin_short_backtrace` loop) | 38.2 % |
| Java executor workers polling at barriers (`awaitAdvancePolling`) | 33.8 % |
| `compute_q4_k_batched_row_range_avx2` (prefill Q4_K)           |  8.9 % |
| `dot_q4_k_q8_k_row_avx2` (decode Q4_K)                         |  4.1 % |
| `executePublishedPlan` (Java executor stages)                  |  4.3 % |
| Q6_K prefill + decode kernels                                  |  2.8 % |
| `WorkerPool::await_workers` (main thread waiting)              |  1.7 % |
| Java attention (`scoreGroup`/`accumulateGroup`), everything else | <1 % each |

Two conclusions, both measured here rather than believed:

1. **The decode kernels are not the bottleneck on this host.** Decode kernels are 5.2 % of
   881 CPU-seconds ≈ 46 CPU-s for 220 tokens ≈ 0.21 CPU-s per token: 2.1 GB per token at
   ~10 GB/s per worker, which is *above* ggml's per-thread rate here (117 GB/s over 16
   threads). Spread over 8 workers that is ~26 ms of a 45 ms token; the other ~19 ms is
   serial: main-thread glue (~5 ms from Result 3) and dispatch/barrier/straggler time across
   201 dispatches (~70 µs each). ggml's 18 ms token on this host is one parallel region with
   ~300 barriers and no main-thread hand-offs. That serial ~19 ms, not the kernel, is the
   decode lever — and it is exactly what a shared host inflates and a dedicated c7a shrinks
   (Result 9 below), which is why the c7a gap is 15 % and this host's is 2×.
2. **Two polling pools on one box fight.** With the vectors barrier polling (composite build)
   the Rust arm carries 16 Java executor threads spinning 34 % of all samples beside the 16
   Rust workers: 32 spinners on 16 vCPUs. The Rust arm still uses the Java executor for the
   batched attention and other Java-side parallel ops, so in production both defaults would
   be active at once. Measured next on the c7a (`/opt/c7a-interact.sh`): Rust arm with the
   Java executor polling 25 ms, parked (0), and the released vectors 0.1.21.

## Confirmation on dedicated cores (AWS c7a.4xlarge, EPYC 9R14, 16 vCPU, interleaved, 3 rounds)

### Result 9a — Rust poll budget: old regime (library at 51cfbd4, 4,000-round spin) vs 1 / 5 / 25 ms

| budget          | decode tok/s          | TTFT p50 ms       | prefill tok/s         | context switches |
|-----------------|----------------------:|------------------:|----------------------:|-----------------:|
| old (spin 4,000)| 25.44 / 25.39 / 25.81 | 868 / 897 / 848   | 133.8 / 131.2 / 137.8 | ≈490 k           |
| 1 ms            | 25.88 / 25.76 / 25.86 | 1007 / 1019 / 1148| 114.3 / 113.9 / 101.8 | ≈300 k           |
| 5 ms            | 26.15 / 25.90 / 25.74 | 944 / 976 / 970   | 122.8 / 118.0 / 119.0 | ≈214 k           |
| 25 ms           | 26.44 / 25.93 / 26.06 | 932 / 973 / 943   | 124.2 / 120.6 / 123.7 | ≈210 k           |

On dedicated cores the budget is worth about +2 % decode (25.5 → 26.1 mean) — the shared
host's third was the cost of *losing the core* when a worker parked, which a dedicated core
does not pay. Prefill is slightly slower with any budget than with the old library
(134 → 123 tok/s at 25 ms, 109 at 1 ms): the old library is the pre-productisation build, so
this row also carries whatever else changed between 51cfbd4 and 6e075d7 (the ABI setter, the
chunk cursor field); it is flagged, not explained, and is the first thing to isolate.

### Result 9b — chunk stealing at 25 ms (dedicated cores)

| config   | decode tok/s          | TTFT p50 ms        | prefill tok/s         |
|----------|----------------------:|-------------------:|----------------------:|
| chunk 0  | 25.94 / 25.72 / 25.98 | 955 / 992 / 917    | 122.5 / 117.6 / 122.5 |
| chunk 16 | 23.24 / 23.03 / 23.46 | 1318 / 1289 / 1015 | 88.0 / 91.1 / 115.5   |

3/3 rounds against: −10 % decode, −25 % prefill. The atomic cursor costs more than the
stragglers it rescues on 8–16 threads with these matrix sizes. Chunk stealing stays off and
should be removed rather than kept behind the env knob.

### Result 9c — pure-Java executor barrier budget (dedicated cores, composite build, 3 rounds)

| `vectors.gguf.pollMillis` | decode tok/s          | mean | TTFT p50 ms          | context switches |
|---------------------------|----------------------:|-----:|---------------------:|-----------------:|
| 0 (park, previous)        | 6.32 / 5.98 / 6.96    | 6.42 | 6754 / 6536 / 6332   | 1.4 – 1.7 M      |
| 1                         | 6.16 / 8.77 / 6.27    | 7.07 | 6346 / 6747 / 6879   | 0.83 – 0.98 M    |
| 5                         | 8.29 / 10.99 / 8.98   | 9.42 | 6722 / 6681 / 6756   | 0.14 – 0.55 M    |
| 25                        | 8.53 / 10.57 / 8.85   | 9.32 | 6717 / 6584 / 6663   | 0.17 – 0.71 M    |

Unlike the Rust pool (9a), the Java executor gains +45 % decode on dedicated cores: its
~200 Phaser barriers per token each parked and woke 16 threads. 1 ms is bimodal (it covers
some inter-stage gaps and misses others); 5 ms and 25 ms are indistinguishable. Prefill is
unchanged (17–18 tok/s: the pure-Java prefill is compute-bound in the kernels, not the
barriers).

**Defaults decided by 9a + 9c: 5 ms for both executors.** It captures everything 25 ms
does on every host measured and burns a fifth of the idle tail (workers × budget per call)
that a sporadic caller — an embedding server, one search a second — would otherwise pay.
Both remain settable (`models.native.kernels.pollMillis`, `vectors.gguf.pollMillis`).

### Result 9d — llama.cpp b10012 build matrix on the c7a (dedicated Zen 4), llama-bench, 2 rounds

| build                          | pp512 t16     | pp512 t8      | tg64 t16    | tg64 t8     |
|--------------------------------|--------------:|--------------:|------------:|------------:|
| native (AVX-512+VNNI) + repack | 269.5 / 270.9 | 146.1 / 143.7 | 32.2 / 32.4 | 31.9 / 31.9 |
| AVX2 + repack                  | 224.4 / 223.7 | 118.2 / 118.5 | 32.2 / 31.9 | 31.5 / 31.5 |
| native, no repack              | 164.5 / 165.5 | 87.2 / 87.3   | 31.1 / 33.1 | 32.3 / 32.2 |
| AVX2, no repack                | 161.7 / 161.9 | 85.1 / 85.1   | 32.8 / 33.2 | 31.9 / 32.1 |

Cleaner than the shared host and the same shape: on prefill the repacked register-tiled GEMM
is +38 % on AVX2 (162 → 224), the ISA alone +2 % (162 → 165), both together +67 %
(162 → 270). Decode is pinned at ~32 tok/s by the 16-vCPU slice's memory bandwidth
(2.1 GB per token ≈ 66 GB/s) in every build. Our Rust arm on the same box: decode 26
(81 % of that ceiling), prefill 123–134 tok/s — 76–83 % of ggml's *untiled* AVX2 rate and
under half of its tiled AVX-512 rate.

**Prefill plan, in order of measured value:** (1) a register-tiled Q4_K×Q8_K GEMM in the
Rust kernel — several activation rows × several weight rows per tile with accumulators
held across K — which does not need ggml's load-time repack if the tile decodes each
weight row's nibbles once per block and reuses them across the tile's activation rows;
(2) an AVX-512/VNNI body for that tile, worth ~+20 % on top of the tiling on Zen 4 and
nothing before it; (3) the same tile shape in the Java Vector API path for the pure-Java
backend, whose prefill (17 tok/s) is compute-bound in the kernel and gains nothing from the
barrier budget. Decode needs neither: it is bandwidth-bound at the kernel and serial at the
dispatch structure (Result 8).

### Result 9e — two pools on one box (c7a, Rust arm, interleaved, 3 rounds)

| Java executor (vectors)            | decode tok/s          | TTFT p50 ms          | prefill tok/s         | context switches |
|------------------------------------|----------------------:|---------------------:|----------------------:|-----------------:|
| polling 25 ms beside the Rust pool | 25.46 / 25.87 / 25.53 | 2,895 / 2,990 / 2,985| 40.1 / 38.5 / 38.9    | 24 – 45 M        |
| parked (budget 0)                  | 25.81 / 25.80 / 25.63 | 939 / 1,141 / 956    | 126.0 / 100.7 / 123.3 | 0.20 – 0.24 M    |
| released vectors 0.1.21 (parks)    | 25.04 / 26.13 / 25.57 | 943 / 1,202 / 940    | 123.0 / 97.9 / 119.9  | 0.21 – 0.24 M    |

The Rust arm drives the Java executor for its Java-side parallel ops; with that executor
polling, its 16 workers (yielding every 64 rounds) sit on the cores the 16 Rust workers need
during prefill: prefill −68 %, 100× the context switches. Decode is untouched because the
single-token path barely uses the Java executor. Parking it restores the released numbers
exactly. Consequence: the budget belongs to whoever owns the box's compute pool —
`VectorUtil.setGgufPollMillis(long)` (vectors #72) and the Models native backend sets 0 at
load. Two unbounded spinners never coexisted in ggml because ggml has one pool.

### Result 9f — the prefill flag from 9a, isolated (c7a, prefill tok/s, 3 rounds each)

| arm                                         | r1    | r2    | r3    |
|---------------------------------------------|------:|------:|------:|
| new library, 16 workers, 25 ms              | 123.4 | 122.0 | 120.7 |
| new library, 15 workers, 25 ms              | 120.7 | 120.9 | 109.9 |
| old library (51cfbd4), 16 workers           | 133.6 | 122.7 | 139.3 |
| old library, 15 workers                     | 129.3 | 120.1 | 129.5 |
| new, caller parks after 4,000 rounds, 5 ms  | 120.0 |  93.6 | 125.6 |
| new, caller spins 5 ms                      | 118.9 | 117.5 | 117.4 |
| new, caller spins 25 ms                     | 102.4 | 120.6 | 123.6 |
| old library (same session)                  | 123.9 | 122.8 | 120.4 |

Neither the worker count nor the caller's completion wait (the only behavioural change in the
library diff) moves prefill outside the host's own spread; the old library's lead is present
in some rounds (134, 139) and absent in others (123, 120). Prefill on this instance swings
±10 % within one configuration, and decode stays at 25.4–26.3 in every arm. Verdict: no
reproducible regression; the completion-wait knob was not kept. Prefill's real lever remains
the tiled GEMM (Result 9d).
