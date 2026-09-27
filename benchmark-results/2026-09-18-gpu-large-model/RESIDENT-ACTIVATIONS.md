# Device-resident activations — measured rejection of the dispatch hypothesis, and the design that follows

Written 2026-09-27. **Provenance: measured here**, on one NVIDIA L40S, both arms on the same host.

## The dispatch hypothesis is rejected

`backend-cuda/README.md` predicted roughly 3.4× as dispatched and roughly 9.8× "with grouped dispatch
and one sync per layer", attributing the shortfall to synchronisation and dispatch granularity. Three
fixes removed almost all of that overhead:

* the redundant per-projection `cuCtxSynchronize` (~200 full-context barriers per decode step);
* three activation uploads coalesced into one contiguous region and a single `cuMemcpyHtoD`;
* per-projection `Arena` allocation replaced by long-lived staging.

Measured A/B, same host, parity re-verified on each arm:

| | before `66714cdc` | after `ddaf818e` |
|---|---|---|
| G1 parity | PASS, 80 tokens identical | PASS, 80 tokens identical |
| accelerated | 12.23 tok/s | **12.01 tok/s** |
| CPU control | 7.12 tok/s | 6.60 tok/s |
| speedup | 1.717× | 1.820× |
| launches / step | 241 | 241 |
| transfers / step | 402 | **402** |
| bytes / step | 12,660,712 | **12,660,712** |

**Read the accelerated column, not the ratio.** Accelerated throughput went *down*; the ratio improved
only because the control fell further. One sample per arm on a shared host makes a 4–7% move in both
numbers noise. **The honest conclusion is no measurable improvement**, and 1.717 → 1.820 must not be
reported as a 6% win.

The prediction was therefore wrong in its *reasoning*, not merely its number. Latency was not the
bottleneck. **The bus is**: 12.7 MB per token, unchanged, with a download after every projection.

The three fixes are kept anyway — parity holds, they do strictly less work, and they remove ~200
native allocations and ~200 barriers per token. They simply do not move this gate.

## Why the traffic is structural

A decode layer alternates device-capable matmuls with host-side elementwise work:

```
RMSNorm(x) ──▶ Q,K,V projections ──▶ RoPE ──▶ GQA attention ──▶ Wo ──▶ residual add
   host              device           host        device        device      host
──▶ RMSNorm ──▶ gate,up projections ──▶ SwiGLU ──▶ Wdown ──▶ residual add
     host              device            host      device        host
```

Seven projections per layer, and **between almost every pair the tensor crosses PCIe**, because the
operation in between runs on the CPU. Grouping what already shares an input is done — `multiplyTriple`
stages one activation for Q/K/V, `multiplyDual` for gate/up — and it does not help, because the
*downloads* remain. The kernel SPI is the reason: `multiply(float[] output, float[] input, …)` names
Java arrays, so every result must land on the host by construction.

## The design

**One idea: results may stay on the device, and the host says when it needs them.**

Add an opaque device-resident handle to the SPI. A projection may write into a handle instead of a
`float[]`; a later call may read a handle as its input; `fetch` copies one to the host when the host
genuinely needs it. Nothing about the existing array-based methods changes, so every current caller and
every other backend keeps working — an accelerator that does not implement residency simply reports it
unsupported and the pure-Java path is unaffected.

Residency is only worth anything if the operations *between* projections also run on device. Four
kernels are missing, all small, all elementwise or one reduction:

| kernel | shape | notes |
|---|---|---|
| `rms_norm` | one reduction + one scale per row | the reduction order must match `TensorOps.rmsNorm` exactly |
| `rope` | elementwise over head pairs | NeoX split-half; the tables already exist host-side |
| `swiglu` | elementwise over two inputs | needs a `sigmoid`/`silu` transcribed from the CPU, same discipline as `expf` |
| `residual_add` | elementwise, scaled | `addScaledActiveInPlace`'s exact form |

With those, a whole layer runs device-side, and per token the bus carries one embedding row in and the
logits out. **A device argmax would reduce the download to a few bytes**, which matters because the
output projection is vocabulary-sized and would otherwise dominate what remains.

## Pre-registered gates, before any GPU hour is spent

1. **G1 parity is non-negotiable and applies to every new kernel individually.** Each of the four must
   be bit-exact against the exact CPU function it replaces, tested off-device against an oracle
   transcribed independently from the Java source — not from the Rust implementation. This campaign
   produced four defects and every one was a kernel validated against a plausible reference instead of
   the code path the gate compares it to. Assume the fifth is waiting in `silu`.
2. **A per-stage ablation switch per kernel**, counted, so a regression can be attributed. The
   attention ablation switch is what localised the token-7 divergence; without it that would still be
   open.
3. **Traffic is the primary metric, not speed.** `transfersPerDecodeStep` and
   `activationBytesPerDecodeStep` must fall by at least an order of magnitude before any speed claim is
   entertained. If traffic falls and speed does not, the new bound is kernel efficiency and that is a
   different investigation — say so rather than reframing.
4. **Decision rule, fixed now.** Decode ≥ 3.0× the same-host CPU control on the 26B MoE model. If
   traffic falls by 10× or more and decode still misses 3.0×, the conclusion is that this path cannot
   reach the gate on this hardware class, and the campaign stops rather than continuing to tune.
5. **One host, both arms, parity on each**, as in the A/B above. A ratio measured against a number
   from different hardware measures the hardware.

## Honest cost and risk

This touches the core inference path, adds four kernels and a device-resident KV cache, and changes an
SPI that other backends implement. It is substantially larger than everything in this campaign so far.
The risk is not that it fails to help — the traffic arithmetic is not in doubt — it is that four more
kernels means four more chances to repeat the mistake this campaign kept making, and each one is
invisible until a model-scale gate catches it.

**Not started.** This document is the proposal and the pre-registration; building it is a separate
decision.
