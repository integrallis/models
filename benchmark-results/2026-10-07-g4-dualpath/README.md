# G4 passes: 6.184x, once the FFN gate and up projections reach the device

**Measured 2026-10-07 on an RTX 4090 (cc 8.9, driver 12080), Runpod.** Models revision
`c107cd14f9c1ee8f7a446aa861eddbb77b42959e`. Same model, same prompts, same host class and the same
`--arm both` protocol as `benchmark-results/2026-10-07-g4-dispatch`, which is the baseline here.

The PTX module is **byte-identical** across the two runs -- kernel sha256
`4c534e622452df95ca7fbb3accc65246be54e2c74d53714d98a03be6d31e8e0e` in both. The change was Java
side only, so nothing in the device code moved and the comparison isolates the routing.

## Before and after

| | baseline | dual path routed | delta |
| --- | --- | --- | --- |
| **accelerated decode** | 9.02 tok/s | **31.26 tok/s** | **+246.6%** |
| control decode | 5.05 tok/s | 5.06 tok/s | +0.2% |
| **G4** | **1.788x — FAILED** | **6.184x — PASSED** | gate is 3.00x |
| `qualified` | false | **true** |
| launches / decode step | 241.0 | 321.0 | +33.2% |
| transfers / decode step | 402.0 | 522.0 | +29.9% |
| activation bytes / decode step | 12,660,712 | 15,398,952 | +21.6% |
| Q4_K decode projections per layer | 4.05 | **6.07** | predicted 6.00 |

The control arm moving 0.2% is what makes the rest readable: the denominator is stable, so the
accelerated arm's 3.47x is not an artefact of a quieter host.

## What it shows

**The dispatch cost rose and it did not matter.** 33% more launches and 30% more transfers bought a
3.47x faster accelerated arm, because the two projections that moved are 53.3% of a layer's
projection arithmetic -- `ffn_gate` and `ffn_up`, 8192x2560 each on Granite 4.1 3B. The earlier
reading of 1.788x as a dispatch ceiling was wrong: dispatch was never the binding constraint at that
point, an unimplemented code path was.

`Q4_K/DECODE_PROJECTION` per layer landing on **6.07** against the 6.00 predicted from the GGUF
header is the direct confirmation that the dual path is now taken, independent of any timing.

## Correctness is not the claim

The previous behaviour was correct. `TensorOps.ggufDualMatmul` computed the gate and up projections
on the CPU and produced right answers; it was slower, not wrong. So this change has no correctness
credit to claim and would have earned nothing had the speed not moved -- it is kept because
31.26 tok/s beats 9.02 tok/s, and for no other reason.

## What is still open

**Device-resident activations are no longer the next lever, and the number that says so is here.**
522 transfers per decode step moving 15.4 MB remains the standing overhead, but it now sits on top
of an arm that passes its gate at 2.06x the threshold. Collapsing those transfers to about 2 is
still arithmetic rather than a measurement, and it needs the elementwise kernels plus a seam that
hands off a layer; it should be prioritised against G1 rather than assumed next.

**G1 is untouched.** This run reports `tokenParity: null` -- it is a decode-speed measurement and
says nothing about the attention argmax. The path passes G4 and still cannot ship until G1 is
settled.

**The declined-projection counter did not reach this report.** `CudaRoutingCounters.declined` was
added and incremented, but `CudaKernelGateCli.Routing` never serialised it, so this run printed
`totalDeclinedProjections = None` -- the counter built to make a silent fallback visible was itself
invisible in the artifact. Fixed in the same change as this record, and verified off-device: the
report now carries `declinedProjections` and `totalDeclinedProjections`.
