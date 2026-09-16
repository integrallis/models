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
