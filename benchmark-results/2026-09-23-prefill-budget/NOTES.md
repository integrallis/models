# Where the Qwen3.5-4B prefill actually goes

Measured 2026-09-23 on an idle Hetzner CCX33 (8 vCPU AMD EPYC-Milan, AVX2, no AVX-512,
32 GiB), load average 0.00, nothing else running.

## The claim this retracts

On 2026-09-23 I wrote, in `2026-09-23-prefill-gap/NOTES.md` and in conversation, that the
gated delta-net recurrence was the prefill bottleneck. The evidence offered was a `perf
report --sort dso` showing 92.97% of samples in `libjmodels_kernels.so` and a resolved hot
frame `jmodels_gated_delta_net_f32_with_context+0x217`.

**That was wrong, and the error was method, not arithmetic.** `--sort dso` attributes to a
shared object, not to a symbol. Two of our kernels live in that object. I read a named frame
out of the callee list and treated it as the share, without ever measuring the split.

## What the split actually is

`cargo test --release -- --ignored --nocapture prefill_budget`, 20 repeats, best-of, on the
box above. Shapes taken from the GGUF, not assumed:

| kernel | shape | time |
|---|---|---|
| gated delta-net, sequential (shipping) | 64 tokens, 16 heads, K=V=256 | 3.920 ms |
| gated delta-net, chunked scan (new) | same | 3.585 ms |
| Q4_K batched matmul | 12288x2560, batch 64 | 10.271 ms (392 GFLOP/s) |

The delta-net rewrite is worth **9%** of that kernel. Over a 1017-token prefill (24 GDN
layers, 16 batches of 64) the whole delta-net is **~1.5 s of a 28.4 s prefill — 5%.**

## The budget, from the GGUF's own tensor shapes

`qwen35`: 32 blocks (24 gated delta-net + 8 full attention), embedding 2560, FFN 9216,
`attn_qkv` 2560x8192, `attn_gate` 2560x4096, `ssm_out` 4096x2560.

Per token: **3,649 M MAC = 7.30 GFLOP.** For a 1017-token prefill: **7.30 TFLOP.**

- at our measured 392 GFLOP/s -> 18.6 s, i.e. **54 tok/s**
- at 1,910 GFLOP/s -> 3.8 s, i.e. **262 tok/s**

Measured `rust-ffm` prefill on this box: **36.4 tok/s** (matmul 18.6 s of 28.4 s = 65%,
remainder delta-net, attention, norms, conv1d and JVM-side work).

**Retracted 2026-09-23, same day.** I wrote here that llama.cpp does 261 tok/s on this box
and that the engine gap was therefore 392 versus ~1,910 GFLOP/s on the Q4_K matmul, 4.9x.
The 261 figure was a remembered number from an earlier run at a different prompt length; it
was never re-measured under this protocol. Re-run properly -- same host, same harness, same
60-token prompt, back to back -- llama.cpp gives **44.9 tok/s prefill and 16.23 tok/s
decode**, against our 40.9 and 15.72. There is no 4.9x matmul gap. We are at 0.91x llama.cpp
on prefill and 0.97x on decode. The 392 GFLOP/s measurement stands; the comparison built on
top of it does not.

## Decode: the actual finding

`p50DecodeTokensPerSecond = 2.46`, `p95TpotMillis = 408`. The run's own `backendDiagnostics`
say `native-quantized-decode = false`. Setting `models.native.quantizedDecode=true` and
changing nothing else:

| arm | decode | prefill | p95 ttft | p95 tpot | tier |
|---|---|---|---|---|---|
| defaults | 2.52 tok/s | - | 1639.4 ms | 401.2 ms | OFFLINE |
| `quantizedDecode=true` | 15.72 tok/s | 40.9 tok/s | 1633.9 ms | 65.1 ms | USABLE |
| llama.cpp b183d2a0 | 16.23 tok/s | 44.9 tok/s | 1442.9 ms | 67.0 ms | USABLE |

6.2x on decode. TTFT is identical across the two candidate arms, which is the control
showing the flag touched only decode. Comparator ratios against llama.cpp: decode **0.968**
(gate >= 0.80), ttft **1.132** and tpot **0.971** (gate <= 1.50).

The default is off *by design*, not by oversight: the production qualification runs a
correctness phase at library defaults and a performance phase that enables it. Flipping the
global default would destroy that control.

## What is *not* established here

- **The delta-net's share of a real prefill.** Every engine run above reports
  `native-gated-delta-net = false`: the Rust delta-net is opt-in via
  `models.native.gatedDeltaNet` and was off, so the engine ran the *Java* delta-net
  throughout. The 3.920 vs 3.585 ms numbers are a valid Rust-internal comparison measured
  through the FFI entry point, but they are not what any prefill above executed, and the
  Java path's cost was never measured. The "delta-net is 5% of prefill" attribution is
  withdrawn.
- That 392 GFLOP/s is a ceiling. It is what our kernel does today, nothing more.
