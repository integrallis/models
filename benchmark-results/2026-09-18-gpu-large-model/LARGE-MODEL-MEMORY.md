# 27B-class Q4_K_M on the Java GPU backend: memory arithmetic and the capacity gate

Campaign work item 3 (`feat/tornado-large-model-plan`). **Nothing here is a GPU measurement.**
Every number is one of three kinds and is labelled as such:

- **read here** — a constant or behaviour read out of this repository or out of `tornado-api`;
- **measured** — a number from a prior run recorded in this repository, with its file named;
- **computed** — arithmetic over the two above.

No 27B-class model has been run on a GPU. Section 5 lists what needs a device.

## 1. The model

The catalog's only 27B-class Q4_K_M entry is **Gemma 4 26B-A4B IT Q4_K_M**, not a dense 27B.
It is a mixture of experts: 90% of its bytes are routed experts of which 8 of 128 are read per
token.

| Property | Value | Source (read here) |
| --- | ---: | --- |
| File bytes | 16,796,015,136 | `backend-java/src/test/resources/model-fixtures.properties:6` |
| Resident tensors | 1,650,027,640 | `Gemma4LargeModelFixtureSlowTest:77` |
| Routed expert tensors | 15,130,165,248 | `Gemma4LargeModelFixtureSlowTest:76` |
| Layers / heads / embedding | 30 / 16 / 2,816 | `Gemma4ConfigTest:37-40` |
| Experts, used per token | 128, 8 | `Gemma4ConfigTest:44-45` |
| Sliding layers (25) | kv heads 8 x head dim 256, keyDim 2,048 | `Gemma4ConfigTest:54-57` |
| Full layers (5: 5,11,17,23,29) | kv heads 2 x head dim 512, keyDim 1,024 | `Gemma4ConfigTest:62-69` |
| Sliding window / ring capacity | 1,024 / 1,024 + 1 | `Gemma4ConfigTest:46`, `Gemma4Decoder.java:76` |
| Model context length | 262,144 | `Gemma4ConfigTest:41` |

**KV cache (computed).** `LayeredKvCache` gives sliding layers a ring and full layers a linear
buffer, F32:

- sliding: 25 x 2 x 2,048 x 4 B x 1,025 slots = **400.4 MiB, constant in context length**
- full: 5 x 2 x 1,024 x 4 B = **40 KiB per token**, linear in context length

| Context | KV bytes |
| ---: | ---: |
| 4,096 | 0.55 GiB |
| 32,768 | 1.64 GiB |
| 131,072 | 5.39 GiB |
| 262,144 | 10.39 GiB |

## 2. What breaks at 27B today

**(a) There is no Q4_K device kernel.** `TornadoGgufBatchedMatrixKernel.supports` returns true only
for `Q4_0`. This GGUF holds F32/Q4_K/Q5_0/Q6_K/Q8_0 (`Gemma4LargeModelFixtureSlowTest:58-64`), so
`TornadoBackend.open` falls back with *"model has no eligible Q4_0 projections"*. Work item 1
(`feat/tornado-kquant-projections`) owns this.

**(b) Gemma 4 cannot use the batched path at all, even with a Q4_K kernel.**
`Gemma4ForwardPass.supportsBatchedPrefill` probes every projection with `batchSize = 2`
(`Gemma4ForwardPass.java:1519-1532`), and `TornadoGgufBatchedMatrixKernel.eligibleBatchSize`
requires `batchSize >= 4` unless it is exactly 1 (`MINIMUM_BATCH = 4`). The probe therefore fails
for every tensor regardless of format. **Coupling for work item 1: fixing the format without
lowering the probe or the minimum leaves Gemma 4 on the CPU.**

**(c) The plan cache retains a host *and* a device copy of every weight, once per batch shape.**
`ProjectionPlan`'s constructor calls `ByteArray.fromSegment(weights)`, which in
`tornado-api 5.2.0-jdk25` computes `(int) segment.byteSize()` and then `MemorySegment.copy`s the
whole tensor into a fresh off-heap array (read here, by decompiling the jar). The task graph pins it
with `DataTransferMode.FIRST_EXECUTION`, and plans are held in unbounded `LinkedHashMap`s released
only on `close()`. With `accelerateDecode` on, prefill and decode are separate keys, so every weight
is held **twice on the device and twice on the host**, on top of the mapped GGUF. That is what the
existing `retainedCopies = 2` in the gate was standing in for.

**(d) A tensor of 2 GiB or more cannot be uploaded at all.** `ByteArray` stores its element count in
an `int`. Nothing checked this before. Gemma 4's largest tensor is well under the limit, but the gate
now refuses rather than failing inside the plan constructor.

**(e) The plan count explodes on a MoE graph.** Expert weights are sliced per expert
(`Gemma4TensorLayout.TensorDescriptor.slice`), and the plan cache keys on `MemorySegment.address()`,
so each of the 128 experts in each of the 30 layers becomes its own plan: 30 x (128 x 2 + 4) x 2 =
**15,600 retained plans**, against 223 measured for Qwen3 0.6B
(`models-accelerator-bench/results/vultr-a40-4q-2026-08-29.md`).

**(f) What the old gate counted.** `modelSizeBytes x retainedCopies + 256 MiB`, against
`0.75 x globalMemory`, with `largestAllocation = min(modelSize, 512 MiB)`. It ignored the KV cache,
the per-plan scratch, the real largest tensor, the host copies, the plan count, and the readiness
time those plans cost. For this model it produced 31.54 GiB and, on a 48 GB L40S, **admitted it**.

## 3. The table (computed)

Device rows use driver-reported global memory, which sits below nameplate: the A40-4Q gate reported
3,917.7 MiB on a 4,096 MiB profile. Usable = global less the gate's 25% margin.

Terms: weights 15.63 GiB; per-plan device scratch across 15,600 plans 2.55 GiB; base 0.25 GiB.

| Context | KV | Shipped shape (2 device copies) | One shared device copy | Host copies |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0.55 | **34.60 GiB** | 18.97 GiB | 31.26 GiB |
| 32,768 | 1.64 | **35.69 GiB** | 20.06 GiB | 31.26 GiB |
| 131,072 | 5.39 | **39.44 GiB** | 23.81 GiB | 31.26 GiB |
| 262,144 | 10.39 | **44.44 GiB** | 28.81 GiB | 31.26 GiB |

| Device | Global | Usable | Shipped shape | One shared copy | Verdict |
| --- | ---: | ---: | --- | --- | --- |
| A40 24 GB | 23.00 | 17.25 | no, short by 17.4 - 27.2 GiB | no (19.0 - 28.8 > 17.25) | **cannot work as shaped** |
| L40S 48 GB | 44.00 | 33.00 | no, short by 1.6 - 11.4 GiB | **yes at every context** | needs shared-upload work |
| 80 GB | 79.00 | 59.25 | fits at every context | fits | capacity fine; **refused on readiness** |

Readiness for 15,600 plans at the measured 16.5 plans/s (223 plans in 13.554 s, A40-4Q) is
**about 948 s**, against the campaign's 120 s G5 ceiling. So even 80 GB is not deployable under the
shipped plan shape.

The 2.55 GiB scratch term is what 15,600 plans cost. If the per-expert plan explosion is solved as
well (240 resident-weight plans, 0.08 GiB of scratch, 14 s of readiness), the shared-upload budget
becomes 16.51 GiB at 4k context, 17.60 GiB at 32k and 26.35 GiB at full context. **A 24 GB device
then fits this model at short context only** — 4k yes, 32k already over the 17.25 GiB usable line —
which is too narrow a margin to qualify on. 48 GB fits every context with room to spare.

## 4. Streaming and partial offload

The design *already* supports host-resident weights on the CPU path: `Gemma4ExpertCache` is a
bounded, lease-safe per-layer expert cache and `Gemma4MappedExperts` serves zero-copy mapped slices.
The device path has no equivalent: plans are unbounded and never evicted.

Per-token weight traffic, computed from the measured layout: resident 1.54 GiB +
8/128 x 15,130,165,248 = 0.88 GiB = **2.42 GiB/token**.

| Arrangement | Per-token bytes over the link | Implied decode | vs measured CPU |
| --- | ---: | ---: | ---: |
| All weights streamed, PCIe Gen4 x16 @ 25 GB/s effective | 2.42 GiB | 9.6 tok/s | **0.75x** |
| All weights streamed, Gen5 x16 @ 55 GB/s | 2.42 GiB | 21.2 tok/s | 1.64x |
| Experts streamed, resident weights on device, Gen4 | 0.88 GiB | 24.7 tok/s | 1.91x |

The CPU control is **measured**: 12.91 tok/s median decode for this artifact on 8 AMD EPYC-Milan
vCPUs (`GEMMA4_QUALIFICATION.md`), implying about 33.5 GB/s of achieved host bandwidth.

**Per-token weight streaming over PCIe is a non-starter.** Streaming everything is slower than the
CPU we already ship; streaming only the routed experts is 1.9x, below the campaign's 3.0x G4 bar,
and an L40S is Gen4. Full device residency is the only arrangement whose arithmetic clears G4.

## 5. Decode reality check (arithmetic, not measurement)

The published 1.21x / 1.32x decode figures are what projections-only acceleration bought on a 0.6B
model. With K-quant projections and device attention landed, and **all 15.63 GiB of weights
resident** on an L40S (864 GB/s nameplate bandwidth — recalled from the vendor spec, not verified
here):

| Assumption | Decode | vs CPU control |
| --- | ---: | ---: |
| 100% of peak bandwidth, zero per-token overhead | 333 tok/s | 25.8x |
| 70% of peak, zero overhead | 233 tok/s | 18.0x |
| 70% of peak, +4 ms/token launch and transfer | 121 tok/s | 9.3x |
| 70% of peak, +8 ms/token launch and transfer | 81 tok/s | 6.3x |

For that to be a real win, **all** of the following must be true, and none is true today:

1. weights are uploaded **once** and shared across batch shapes (halves 31.26 GiB to 15.63 GiB);
2. the retained plan count drops from 15,600 to the low hundreds, or readiness blows G5;
3. attention and the KV cache stay on device, or the per-token host round trip dominates — the
   current plans transfer activations host to device and results back on **every** execution, so at
   30 layers x ~4 projections that latency is the binding term, not bandwidth;
4. host memory holds the off-heap copies: 15.63 GiB of copies **plus** the mapped GGUF, so a
   32 GB host (the one that qualified this model on CPU) cannot host the accelerated path.

The +4 ms and +8 ms overhead rows are guesses at kernel-launch and transfer latency. They are the
term most likely to decide the outcome and the term we have no measurement for.

## 6. What this branch implemented

- `DeviceMemoryRequest` — coarse (file-size-only) or detailed (weights, retained shapes, plan count,
  plan scratch, largest allocation, device KV).
- `PlanShapeStrategy` — `PER_SHAPE_WHOLE_MODEL` (shipped), `SHARED_WEIGHT_UPLOAD` and
  `BOUNDED_RESIDENT_WORKING_SET` (not implemented, each carrying the reason).
- `DeviceBudget` — itemised device and host budget plus a readiness estimate from the measured plan
  compile rate.
- `AcceleratorEligibility` — gates on capacity, the 2 GiB `ByteArray` cap, the device allocation
  limit, the readiness ceiling, and **refuses a file-size-only budget above 8 GiB of weights**
  instead of gating on a number known to be incomplete. Messages name what was needed, what was
  available, and which unimplemented plan shape would have fit.
- `LargeModelBudgetExperiment` — the table above, from the command line, no GPU and no recompile.

The shipped plan shape is **not** made to host a 27B model on this branch. Doing that needs edits
inside `TornadoGgufBatchedMatrixKernel`, which work item 1 is rewriting for K-quants; this branch
deliberately does not touch it.

## REPRODUCE

```
./gradlew :backend-tornado:test --tests '*LargeModelEligibilityTest*'
./gradlew :models-accelerator-bench:compileJava
java -cp <models-accelerator-bench and its deps> \
  com.integrallis.models.accelerator.LargeModelBudgetExperiment \
  --preset gemma4-26b-a4b --context 4096,32768,131072,262144 \
  --device "A40 24 GB:23:8" --device "L40S 48 GB:44:8" --device "H100 80 GB:79:32"
```

## What still needs a GPU host

1. **Calibrate the base overhead.** `BASE_PLAN_OVERHEAD_BYTES` (256 MiB) is unmeasured.
   `TornadoExecutionPlan.getCurrentDeviceMemoryUsage()` reports real usage after readiness; run it
   for 1, 10, and 223 plans and replace the constant with a measured base plus per-plan term.
2. **Measure the plan compile rate on the target device.** The 16.5 plans/s rate is from A40-4Q on
   Qwen3 0.6B; K-quant kernels are larger and may compile slower.
3. **Measure per-token launch and transfer latency**, which decides section 5 outright.
4. **Recommended device for the qualification run: an L40S 48 GB**, not an A40 24 GB. A 24 GB
   device holds this model only if the weights are uploaded once *and* the per-expert plan
   explosion is fixed, and then only up to about 4k context — too narrow to qualify on, and an A40
   run under the shipped shape can only reproduce the refusal. A 48 GB L40S holds weights, KV at
   full 262k context, and plan scratch once the weights are uploaded once.

   Vendor list prices, read from Vultr's published pages via search on 2026-09-18 and **not**
   verified against a live quote: L40S from **$0.848 per GPU-hour** (36-month prepaid; on-demand is
   higher), A40 from **$1.712 per GPU-hour**. On these numbers the L40S is both the right size and
   the cheaper device, so there is no cost argument for the A40.
