# Inference Benchmarks

`models-bench` contains the controlled comparison harness and a decode-only JFR profiler for the
pure-Java backend.

## Retained grouped K-quant prefill gate

The mixed Q4_K/Q4_K/Q6_K grouped batched path was retained on the controlled eight-vCPU AMD
EPYC-Milan host with Java 25. Using `prompts/completion.txt`, MiniCPM5 1B Q4_K_M, prefill batch 32,
one warmup, and 12 counterbalanced measurements per mode produced:

| Mode | Median TTFT | Median prefill | Output hash |
| --- | ---: | ---: | --- |
| Independent batched projections | 8330.16 ms | 17.95 tok/s | `01ba4719...546b` |
| Grouped batched projections | 7948.31 ms | 18.78 tok/s | `01ba4719...546b` |

This is a 4.58% TTFT reduction and a 4.61% prefill-throughput increase with bit-identical output.
Use `-Dmodels.purejava.groupedProjections=false` for the controlled baseline or rollback. This gate
does not justify a global batch-size increase: Q4_0 and Q8_0 models plateaued near batches 32-64,
while MiniCPM's ungrouped mixed K-quant path was fastest at batch one.

## Retained grouped Q4_0 prefill gate

Qwen3 0.6B Q4_0 was measured independently on the same Java 25 EPYC-Milan host, with prefill batch
32, the committed completion prompt, and six counterbalanced process pairs. Across 18 trials per
mode, grouped Q/K/V and gate/up projection dispatch improved median TTFT from 3127.92 to 3096.45 ms
and median prefill throughput from 49.59 to 50.19 tok/s. Every paired trial had identical input and
output token counts and an identical output SHA-256.

An allocation-enabled JFR follow-up used a fixed 2 GiB heap, five warmups, and five measurements.
Grouped median TTFT was 3076.4 ms versus 3124.0 ms, while total recorded TLAB plus outside-TLAB
allocation differed by 9.99 MB over the complete ten-prompt process. Peak RSS differed by 8.28 MB.
The much larger delta seen with only one warmup was transient JIT scalar-replacement behavior, not
steady per-prefill allocation. The same `models.purejava.groupedProjections` property controls this
path and provides its rollback gate.

## Retained grouped Q8_0 gate/up prefill gate

SmolLM2 360M Q8_0 was measured with the same Java 25 host, batch 32, completion prompt, and six
counterbalanced process pairs. Across 18 trials per mode, grouping only the batched gate/up
projection improved median TTFT from 2612.09 to 2581.73 ms and median prefill throughput from 60.22
to 61.01 tok/s. Mean process CPU time fell from 18,875 to 18,698 ms, median RSS did not increase,
and every pair had identical input/output token counts and output SHA-256 values.

Q8_0 Q/K/V remains independent. A direct follow-up against the retained dual-only baseline measured
2584.27 versus 2580.77 ms median TTFT and 61.01 versus 61.04 tok/s for dual-only versus dual plus
triple Q/K/V. Three of six pairs were faster and three were slower, while median RSS increased by
6.68 MB. The route was rejected as noise despite exact outputs; no runtime flag exposes it.

## Retained Q5_0 batched prefill gate

DeepSeek-Coder 1.3B Q4_K_M contains Q5_0 tensors, so the missing Q5_0 batched operation forced its
entire prefill plan to batch one. The retained route was measured on the same Java 25 EPYC-Milan
host with the pinned 873,582,624-byte GGUF, a fixed 2 GiB heap, eight inference threads, one warmup,
three trials per process, and six counterbalanced batch-one/batch-32 process pairs. The committed
`prompts/code-completion-medium.txt` produced 193-196 tokens after the per-trial nonce.

The first process, batch one, had a 33.40-second median while every later batch-one process measured
16.81-16.90 seconds. That global-order cold-start outlier cannot be separated from mode, so the
steady aggregate uses the remaining five pairs (15 trials per mode):

| Metric | Batch 1 | Batch 32 | Change |
| --- | ---: | ---: | ---: |
| Median TTFT | 16,837.21 ms | 16,181.43 ms | -3.90% |
| Median prefill | 11.524 tok/s | 12.110 tok/s | +5.09% |
| Mean trial CPU | 127,165 ms | 123,253 ms | -3.08% |
| Median peak RSS, all six processes | 1,493,338,112 B | 1,590,452,224 B | +97,114,112 B |

All six paired medians favored batch 32. Corresponding input-token sequences and output hashes were
identical; the sole output hash was
`1a0f564ddc6039457b2fb26b3d6a316c15eba20a886449847c3210c35821a693`. A separate six-pair
23-24-token gate measured a smaller 1.80% TTFT reduction, while a single 421-token corroborating
trial measured 7.13%; those results show the expected prompt-length scaling but do not replace the
steady medium-prompt gate. Set `-Dmodels.purejava.prefillBatchSize=1` when memory capacity is more
important than prompt latency.

## Retained final-layer prefill FFN pruning gate

Ordinary prefill returns logits only for the final prompt token. After the final layer has written
every K/V cache row and completed attention, FFN results for earlier prompt rows cannot affect
future decoding. The retained plan skips those FFN rows for the measured homogeneous Q4_0 and Q8_0
layouts, while speculative verification and observer-backed diagnostics continue to compute every
requested output row. Other tensor layouts remain on their exact full-row path pending a separate
gate.

Two quantization families were measured on the same Java 25 EPYC-Milan host with prefill batch 32,
the committed completion prompt, fixed heaps, one warmup, and six counterbalanced process pairs.
Each mode contains 18 measured trials:

| Model | Metric | Full final FFN | Pruned final FFN | Change |
| --- | --- | ---: | ---: | ---: |
| Qwen3 0.6B Q4_0 | Median TTFT | 3103.87 ms | 3037.63 ms | -2.13% |
| Qwen3 0.6B Q4_0 | Median prefill | 50.021 tok/s | 51.167 tok/s | +2.29% |
| Qwen3 0.6B Q4_0 | Mean trial CPU | 22,455 ms | 21,942 ms | -2.29% |
| SmolLM2 360M Q8_0 | Median TTFT | 2580.03 ms | 2528.12 ms | -2.01% |
| SmolLM2 360M Q8_0 | Median prefill | 61.097 tok/s | 62.381 tok/s | +2.10% |
| SmolLM2 360M Q8_0 | Mean trial CPU | 18,696 ms | 18,289 ms | -2.17% |

All six pair medians improved for both models. Corresponding input counts, output counts, and output
SHA-256 values matched in all 36 paired trials. The default-on Qwen path also passed its pinned
llama.cpp greedy-token reference. Median process peak RSS changed from 1,192,462,336 to
1,221,462,016 bytes for Qwen and from 688,959,488 to 705,396,736 bytes for SmolLM2. The
implementation adds no retained buffers; these overlapping process-level observations are recorded
without claiming a memory improvement. Use
`-Dmodels.purejava.finalLayerPrefillPruning=false` for rollback or reproduction.

## Retained final-layer K/V-only prefill gate

The next graph stage moves the final-layer output-row boundary ahead of Q projection. Every prompt
row still runs attention normalization and K/V projection, normalization, RoPE, and cache storage.
Only the requested final row runs Q projection, attention, output projection, and FFN. The stage is
enabled only when all final-layer attention and FFN projections use the same validated Q4_0 or Q8_0
format; the retained FFN-only route remains the fallback.

The incremental baseline kept final-layer FFN pruning enabled and disabled only K/V-only prefill.
The candidate used the same Java 25 EPYC-Milan host, batch 32, committed completion prompt, fixed
heaps, one warmup, three measured trials per process, and six counterbalanced process pairs:

| Model | Metric | FFN-only baseline | K/V-only prompt rows | Change |
| --- | --- | ---: | ---: | ---: |
| Qwen3 0.6B Q4_0 | Mean trial TTFT | 3039.72 ms | 3023.80 ms | -0.52% |
| Qwen3 0.6B Q4_0 | Mean prefill | 51.024 tok/s | 51.341 tok/s | +0.62% |
| Qwen3 0.6B Q4_0 | Mean trial CPU | 21,989 ms | 21,858 ms | -0.60% |
| SmolLM2 360M Q8_0 | Mean trial TTFT | 2532.43 ms | 2515.50 ms | -0.67% |
| SmolLM2 360M Q8_0 | Mean prefill | 62.203 tok/s | 62.623 tok/s | +0.68% |
| SmolLM2 360M Q8_0 | Mean trial CPU | 18,298 ms | 18,207 ms | -0.50% |

All six paired medians improved for both models, and all 36 corresponding trials matched input
count, output count, and output SHA-256. Median process peak RSS changed from 1,200,603,136 to
1,203,666,944 bytes for Qwen and from 708,460,544 to 694,071,296 bytes for SmolLM2. Synthetic Q4_0
and Q8_0 tests additionally require bit-identical complete K/V buffers and next-token state for
batch one and batch 32. The default-on Qwen path passed its pinned llama.cpp greedy reference and
every-layer state test. Use `-Dmodels.purejava.finalLayerKvOnlyPrefill=false` to return to the
retained FFN-only baseline.

## Retained exact attention row gates

Qwen3 0.6B Q4_0 was used to gate two independent attention optimizations on the same Java 25
EPYC-Milan host. The value baseline used ordinary row-at-a-time accumulation; the candidate used
the generic Vectors four-row weighted accumulation primitive. Six counterbalanced process pairs,
with three trials per process, produced:

| Metric | Independent values | Four-row values | Change |
| --- | ---: | ---: | ---: |
| Median decode | 35.389 tok/s | 36.367 tok/s | +2.76% |
| Median TPOT | 28.259 ms | 27.497 ms | -2.70% |
| Median TTFT | 1787.03 ms | 1761.08 ms | -1.45% |
| Median prefill | 87.640 tok/s | 89.112 tok/s | +1.68% |
| Mean trial CPU | 23,290 ms | 23,159 ms | -0.56% |

The score gate then compared independent key-cache dots with exact two-row batching while retaining
the value candidate in both modes. A three-fork Vectors JMH gate at 128 columns improved 64, 192,
and 512 rows by 12.5%, 14.4%, and 12.6%. The whole-model Graal run again used six counterbalanced
process pairs and 18 trials per mode. Raw aggregate medians moved from 37.747 to 37.834 decode tok/s,
26.502 to 26.431 ms TPOT, and 1761.98 to 1740.81 ms TTFT. Paired analysis is more sensitive to the
small kernel effect: median paired decode improved 1.89%, 16 of 18 decode trials won, and five of
six process-pair medians won.

A fixed-position follow-up used 64 warmup and 256 measured decode tokens per process. All three
counterbalanced pairs improved: 35.30 to 36.56, 35.03 to 36.38, and 35.74 to 36.48 tok/s. Every
recording reported zero GC and the identical checksum `514.657357`. The full benchmark additionally
matched input count, output count, and output SHA-256 in all 18 corresponding trial pairs. Synthetic
tests require bit-identical logits, complete K/V buffers, and the next autoregressive step for both
batch one and batch 32. The model-scoped controls are
`models.purejava.batchedAttentionValues` and `models.purejava.batchedAttentionScores`.

## Retained Q4_0 attention-and-FFN stage plan

The first retained widening kept output projection, residual/FFN-input preparation, gate/up plus
exact SwiGLU, and down projection inside one four-stage Vectors plan. Its controlled gate used
Qwen3 0.6B Q4_0, GraalVM Java 25.0.3, batch 24, fixed 2 GiB heap, eight processors and workers, the
required `MaximumInliningSize=10000` argument, unsigned-pairwise Q4, and every previously retained
Qwen prefill option. Ten warmups and five trials per fresh process across six counterbalanced pairs
produced 30 trials per mode:

| Metric | Staged FFN baseline | Four-stage layer plan | Change |
| --- | ---: | ---: | ---: |
| p50 TTFT | 1,283.533 ms | 1,208.101 ms | -5.88%; -3.05% paired median |
| p95 TTFT | 1,354.849 ms | 1,291.279 ms | -4.69% |
| p50 prefill | 121.449 tok/s | 129.200 tok/s | +6.38%; +3.53% paired median |
| p50 CPU | 8,755 ms | 8,165 ms | -6.74%; -3.88% paired median |

That candidate won 24 of 30 paired TTFT/prefill trials and five of six process-pair medians. Every
corresponding input count, output count, and output SHA-256 matched. Median process RSS was
1,145,901,056 bytes for the baseline and 1,214,373,888 bytes for the candidate; compiler lifetime
and process variance prevent a causal memory claim. Reports remain on the controlled host under
`/opt/modeljars-bench/q4-layer-candidate/reports/full/`.

The current Models-owned schedule extends the same publication across Q/K/V bias, normalization,
RoPE, and cache writes; exact grouped-query attention; and Q8_0 attention preparation. Those three
stages precede the retained output-and-FFN stages, producing one seven-stage plan. Cache storage is
reserved before worker publication, the first stage writes disjoint positions, and each batch row
has independent score scratch. The QKV matrix projection remains the preceding boundary. No
Vectors source or API change was needed for the original Q4_0 widening, and the established
two-stage FFN route remains the fallback when a layer's output projection is not supported.

The incremental gate used Models candidate `18552bc` and otherwise repeated the controls above.
It differed only by `models.purejava.stagedQuantizedLayer=false/true` within one candidate
distribution:

| Metric | Staged FFN baseline | Seven-stage layer plan | Change |
| --- | ---: | ---: | ---: |
| p50 TTFT | 1,274.479 ms | 1,139.435 ms | -10.60%; -12.43% paired median |
| p95 TTFT | 1,372.869 ms | 1,166.341 ms | -15.04% |
| p50 prefill | 122.122 tok/s | 136.974 tok/s | +12.16%; +15.02% paired median |
| p50 CPU | 8,490 ms | 8,625 ms | +1.59%; +0.65% paired median |

The seven-stage candidate won all 30 paired TTFT and prefill trials and all six process-pair
medians. Every corresponding input count, output count, and output SHA-256 matched. Median process
RSS was 1,140,885,504 bytes for the baseline and 1,179,170,816 bytes for the candidate; neither RSS
nor the small CPU change supports an efficiency claim. Reports remain on the controlled host under
`/opt/modeljars-bench/q4-layer-candidate/reports/attention-full/` and were copied to
`/private/tmp/q4-attention-full/`. Select the route with
`-Dmodels.purejava.stagedQuantizedLayer=true`; exact ModelJars profiles supply it only for measured
tuples.

## Retained Q8_0 attention-and-FFN stage plan

The format-neutral schedule now dispatches Q4_0 and Q8_0 row ranges through their corresponding
Vectors kernels. Q8_0 support added reusable generic activation block-range quantization and a
prequantized row-range matrix primitive to Vectors; Models continues to own transformer staging.
Synthetic tests require bit-identical final logits, complete K/V buffers, and the next
autoregressive step.

The controlled gate used SmolLM2 360M Instruct Q8_0, Models `b2eb75b`, Vectors `fa24b91`, GraalVM
Java 25.0.3, batch 32, a fixed 1 GiB heap, eight processors and workers, and
`MaximumInliningSize=10000`. Five warmups and five measured one-token requests per fresh process
across six counterbalanced pairs produced 30 trials per mode:

| Metric | Established Q8_0 prefill | Seven-stage Q8_0 layer | Change |
| --- | ---: | ---: | ---: |
| p50 TTFT | 1,183.920 ms | 995.194 ms | -15.94%; -15.76% paired median |
| p95 TTFT | 1,254.584 ms | 1,015.008 ms | -19.10% |
| p50 prefill | 133.829 tok/s | 159.522 tok/s | +19.20%; +19.19% paired median |
| p50 CPU | 7,525 ms | 7,490 ms | -0.47%; -0.40% paired median |
| median process RSS | 993,314,816 bytes | 988,583,936 bytes | no regression observed |

All 30 paired TTFT and prefill trials and all six process-pair medians favored the staged route.
Every corresponding input count, output count, and output SHA-256 matched. A separate prefill-only
gate with ten warmups improved median throughput from 130.08 to 150.71 tok/s (+15.86%) with zero GC
and checksum `-114227.445` in every process. Candidate JFR samples moved activation quantization
from 15.82% of the established profile to 3.08%; the Q8_0 row-range kernel then represented 74.79%
of samples. Reports remain under `/opt/modeljars-bench/q8-staged-layer/reports/` and the JSON reports
were copied to `/private/tmp/q8-staged-layer/`. Select the route with
`-Dmodels.purejava.stagedQuantizedLayer=true`; exact ModelJars profiles supply both staged settings
only for measured runtime tuples.

## Retained Q8_0 batch weight reuse gate

The staged Q8_0 profile made row-range projection dominant. Vectors now widens each packed weight
block once and reuses those integer lanes across the activation batch instead of repeating the
same weight conversion for every prompt row. The batch-one path retains the established integer
dot. A prequantized 1024x2048 JMH gate on AVX2 reduced batch-32 latency from 21.044 to 13.781 ms
(-34.5%) while exact scalar-versus-Panama row-range tests remained bit identical.

The full-model gate held Models `6c306f0`, the SmolLM2 bytes, GraalVM Java 25.0.3, batch 32, fixed
1 GiB heap, eight processors/workers, staged settings, and `MaximumInliningSize=10000` constant.
It compared Vectors `7fb6fa5` and `25aa094` over five warmups and five one-token measurements in
each of six counterbalanced process pairs:

| Metric | Repeated weight conversion | Reused weight conversion | Change |
| --- | ---: | ---: | ---: |
| p50 TTFT | 1,003.739 ms | 925.665 ms | -7.78%; -7.63% paired median |
| p95 TTFT | 1,031.209 ms | 935.595 ms | -9.27% |
| p50 prefill | 158.580 tok/s | 170.463 tok/s | +7.49%; +7.47% paired median |
| p50 process CPU | 7,545 ms | 6,960 ms | -7.75% |
| median process RSS | 992,069,632 B | 1,038,714,880 B | +4.70%; no memory claim |

All 30 corresponding input counts, output counts, and output SHA-256 values matched, and every pair
favored the candidate. A fixed-position decode control measured 42.46 versus 42.23 tok/s median
throughput (-0.54%, treated as neutral variance); every process reported checksum `763.224787`,
zero collections, and zero GC pause. Reports remain under
`/opt/modeljars-bench/q8-weight-reuse-reports/`.

## Retained Q8_0 block-major activation gate

After weight conversion was hoisted, activation-batch locality became the next Q8_0 row-kernel
target. A pre-widened `int[]` layout was implemented test first and rejected: at 1024x2048,
batch 32, it improved local AVX2 from 19.013 to 15.432 ms/op (-18.8%) but was neutral on
controlled EPYC/GraalVM (21.812 versus 21.828 ms/op). The authoritative host did not validate the
extra representation; its integer copy quadrupled the activation payload and did not remove the
observed cache-locality boundary after roughly 16 rows.

The retained layout instead preserves the canonical batch-major Q8 bytes and creates one additional
block-major byte copy during activation quantization. The explicit Vectors primitive reuses the
existing exact batched accumulator with a 32-byte activation stride. Exact tests cover whole and
split quantization, scalar and Panama row ranges, provider ownership, final logits, complete K/V
buffers, and the next autoregressive state. The default packed API and batch-one path are unchanged.

On local Temurin 25 AVX2, the 1024x2048 prequantized JMH gate was neutral at batch one, improved
4.8% at batch four, 2.9% at batch eight, and 11.5% at batch 32. On controlled EPYC/GraalVM it was
neutral through batch eight and improved batch 32 from 21.318 to 20.347 ms/op (-4.6%). The
authoritative full-model gate held Vectors `d295f32`, Models `29b5169`, SmolLM2 model bytes,
GraalVM Java 25, fixed 1 GiB heap, eight processors/workers, batch 32, all staged settings, and
`MaximumInliningSize=10000` constant. Only
`models.purejava.blockMajorQ8Activations=false/true` changed across six counterbalanced process
pairs, each with five warmups and five measured one-token requests:

| Metric | Packed activation bytes | Block-major activation bytes | Change |
| --- | ---: | ---: | ---: |
| p50 TTFT | 939.347 ms | 925.622 ms | -1.46%; -1.69% paired-process median |
| p95 TTFT | 965.011 ms | 943.771 ms | -2.20% |
| p50 prefill | 167.890 tok/s | 170.006 tok/s | +1.26% |
| p50 process CPU | 7,040 ms | 6,920 ms | -1.70% |
| median process RSS | 1,020,530,688 B | 1,018,851,328 B | no regression observed |

Five of six process medians favored the candidate, all 60 trials succeeded, and every corresponding
input count, output count, and output SHA-256 matched. Reports remain under
`/opt/modeljars-bench/q8-block-major-reports/`. The planner keeps the feature default-off and reports
it through plan schema `pure-java-v13`; exact ModelJars profiles opt in with
`-Dmodels.purejava.blockMajorQ8Activations=true` only for measured Q8_0 tuples.

## Retained parallel Q8_0 FFN preparation gate

The post-layout JFR profile showed stage-completion skew while one participant performed residual
addition, RMS normalization, and Q8 activation preparation for the whole prompt batch. The retained
candidate partitions those independent operations by active batch row while preserving the same
seven-stage plan. Vectors exposes exact disjoint Q8 batch-range quantization; Models owns the
all-Q8 topology gate and transformer scheduling. Mixed gate/up formats, packed activations,
single-participant runtimes, and batch-one decoding remain on the serial path.

The authoritative full-model gate held Vectors candidate `4dcf935` (merged as `523f3aa`), Models
`4887cde`, ModelJars `6db5163`,
SmolLM2 SHA-256 `48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201`,
prompt SHA-256 `2db2d875631cc7e3af3f6e4471ae4c9b2b7dfdb31ab561a41ef78182a31532e6`,
GraalVM Java 25, a fixed 1 GiB heap, eight processors/workers, batch 32, block-major activations,
all staged settings, and `MaximumInliningSize=10000` constant. Only
`models.purejava.parallelQ8FfnPreparation=false/true` changed. Five warmups and five measured
one-token requests in each of six counterbalanced fresh-process pairs produced 30 trials per mode:

| Metric | Serial FFN preparation | Batch-row preparation | Change |
| --- | ---: | ---: | ---: |
| p50 TTFT | 913.102 ms | 899.064 ms | -1.54%; -1.68% paired-process median |
| p95 TTFT | 930.788 ms | 914.340 ms | -1.77% |
| p50 prefill | 172.543 tok/s | 175.592 tok/s | +1.77% |
| p50 process CPU | 6,860 ms | 6,840 ms | -0.29% |
| median process RSS | 1,015,816,192 B | 1,013,358,592 B | -0.24% |

All six process-pair medians and 28 of 30 corresponding trials favored the candidate for TTFT and
prefill. All 60 trials succeeded, and every corresponding input count, output count, and output
SHA-256 matched. Maximum observed RSS changed by +0.76%, with no median-memory regression. Reports
remain under `/opt/modeljars-bench/q8-ffn-preparation-reports/` and local copies under
`/private/tmp/q8-ffn-preparation-reports/`. The option remains default-off in plan schema
`pure-java-v14`; the exact ModelJars profile
`smollm2_360m_q8_0_epyc_milan_jdk25_parallel_ffn` opts in for the measured tuple.

## Retained row-accumulated Q8_0 block-major gate

The established block-major kernel updated each widely separated batch output after every weight
block. The retained Vectors strategy instead keeps one batch-sized partial-sum array local to an
output row, preserves the exact block arithmetic order, and scatters the completed row once. It is
an explicit `GgufQ8BlockMajorKernel` choice; the existing scattered strategy remains the default.

Local Java 25 HotSpot/C2 was neutral at 17.713 versus 17.480 ms/op. Controlled EPYC/GraalVM Java
25 improved the 1,024x2,048 batch-32 JMH kernel from 20.571 to 4.335 ms/op (-78.9%). Models plan
schema `pure-java-v16` therefore selects the row accumulator only when an exact profile recommends
`models.purejava.q8BlockMajorKernel=row-accumulated`, the compiler is Graal JVMCI, and staged
block-major Q8 topology is active. Other runtimes and models retain the scattered kernel.

The full-model gate held the exact SmolLM2 and prompt bytes, GraalVM Java 25, a fixed 1 GiB heap,
eight processors/workers, batch 32, all prior staged Q8 settings, parallel FFN preparation, and
`MaximumInliningSize=10000` constant. Six counterbalanced process pairs with five warmups and five
measured requests per mode produced:

| Metric | Scattered output updates | Row-local accumulator | Change |
| --- | ---: | ---: | ---: |
| p50 TTFT | 898.794 ms | 878.801 ms | -2.22% |
| p95 TTFT | 919.981 ms | 886.209 ms | -3.67% |
| p50 prefill | 175.219 tok/s | 179.479 tok/s | +2.43% |
| p50 process CPU | 6,840 ms | 6,680 ms | -2.34% |
| median process RSS | 1,009,147,904 B | 1,015,336,960 B | +0.61% |

The candidate won TTFT, prefill, and CPU in all 30 corresponding trials and all six process-pair
medians. Every corresponding input count, output count, and output SHA-256 matched. Maximum RSS
moved by +0.41%. Reports remain under
`/opt/q8-signed-pairwise-20260723/reports/row-accumulator-full-*.json` with local copies under
`/private/tmp/q8-row-accumulator-reports/`.

## Retained float-lane Q8_0 block-major gate

The row-local kernel still horizontally reduced eight integer lanes after every Q8 block. The
float-lane strategy keeps those eight strided lanes live across the complete output and reduces
once. It is available only with an exact 256-bit vector species and remains profile-selected with
`models.purejava.q8BlockMajorKernel=float-lane-accumulated`. Its deterministic arithmetic order is
not bit-identical to row/scattered accumulation, so a profile must not imply cross-kernel token
identity.

Controlled 1,024x2,048 batch-32 EPYC/GraalVM Java 25 JMH improved from
`4.320 +/- 0.012 ms/op` for the row accumulator to `2.829 +/- 0.019 ms/op` (-34.5%). Reusable
thread-local lane scratch reduced normalized candidate allocation to `19.613 +/- 0.137 B/op`.
Models performs one bounded prewarm before staged worker fan-out. The exact launch profile also
uses:

```text
-XX:CompileCommand=option,com.integrallis.vectors.core.PanamaVectorUtilSupport::ggufQ8_0Q8_0BlockMajorFloatLaneRow,double,CompileThresholdScaling,0.001
-XX:CompileCommand=option,com.integrallis.vectors.core.PanamaVectorUtilSupport::ggufQ8_0Q8_0BlockMajorFloatLaneRow,bool,BackgroundCompilation,false
```

The authoritative full-model gate held the row-accumulator setup above constant and used six
counterbalanced process pairs, each with five warmups and five measured requests:

| Metric | Row-local accumulator | Eight retained float lanes | Change |
| --- | ---: | ---: | ---: |
| p50 TTFT | 876.832 ms | 746.621 ms | -14.85% |
| p95 TTFT | 886.504 ms | 756.694 ms | -14.64% |
| p50 prefill | 180.000 tok/s | 211.547 tok/s | +17.53% |
| p50 process CPU | 6,660 ms | 5,600 ms | -15.92% |
| median process RSS | 1,019,949,056 B | 1,038,696,448 B | +1.84% |
| maximum process RSS | 1,044,836,352 B | 1,058,574,336 B | +1.31% |

The candidate won all 30 corresponding TTFT, prefill, and CPU trials and all six process-pair
medians. Every corresponding one-token output matched. A three-prompt, 64-token, three-repeat audit
proved exact internal repeatability while explicitly confirming expected trajectory divergence from
the row/scattered arithmetic order. Reports remain under `/opt/q8-float-lane-model-gate/reports/`
with local copies under `/private/tmp/q8-float-lane-reports/`.

## Exact determinism audit

Audit the raw float bits of every generated logit vector across repeated greedy inference trials:

```shell
./gradlew :models-bench:installDist
models-bench/build/install/models-bench/bin/models-bench determinism \
  --modeljar qwen3_0_6b_q4_0 \
  --prompt-file models-bench/prompts/completion.txt \
  --tokens 4 \
  --iterations 3 \
  --context 128 \
  --output build/reports/determinism/qwen3_0_6b_q4_0.json
```

The schema-2 report records the artifact hash, prompt tokens, winning and runner-up logits, winner
margins, raw-bit logit hashes, JVM and host details, the complete vectors execution configuration,
and the selected backend plan and ModelJar profile decisions. The command exits non-zero when any
trial differs.

Use exactly one model source. `--modeljar <alias>` resolves the descriptor from the aggregate
catalog on the application runtime classpath and preserves its model-scoped performance profile
when loading the backend. `--model /path/to/model.gguf` loads unregistered bytes without a profile.

Use `scripts/run-purejava-determinism-matrix.sh` for fresh-JVM coverage of the default Panama path,
serial execution, FMA-disabled execution, 128-bit vectors, and the scalar reference. Provider
selection and vector constants are fixed at class initialization, so these modes must not share a
JVM.

## Prefill-only profile

Warm up complete prompt passes, then reset the backend and record exactly one measured prefill. The
recording excludes model loading, warmup, reset, checksum calculation, and autoregressive decode:

```shell
./gradlew :models-bench:run --args='profile-prefill \
  --modeljar qwen3_0_6b_q4_0 \
  --prompt-file models-bench/prompts/completion.txt \
  --context 2048 \
  --warmups 2 \
  --output build/reports/inference/prefill-profile.jfr'
```

The command reports prompt tokens, warmup passes, prefill throughput, a deterministic final-logit
checksum, and GC deltas. Use a representative prompt long enough to produce useful execution
samples. `--model /path/to/model.gguf` remains the explicit unregistered alternative.

## Decode-only profile

Run prompt prefill and warmup, reset and replay the prompt outside JFR, then record a fixed-token
autoregressive decode loop at the same positions as the production generation window:

```shell
./gradlew :models-bench:run --args='profile-decode \
  --modeljar qwen3_0_6b_q4_0 \
  --prompt "Explain vectorized inference." \
  --context 2048 \
  --warmup-tokens 64 \
  --measure-tokens 256 \
  --output build/reports/inference/decode-profile.jfr'
```

The command reports prompt, warmup, and measured token counts, decode throughput, and a deterministic
logit checksum. The recording excludes model loading, both prompt prefill passes, and decode warmup
so CPU attribution describes steady autoregressive decode without measuring a later context window.
Use `--prompt-file` for a file-backed prompt and `--token-id` to select the repeated token explicitly.

The requested context must hold the prompt plus the larger of the warmup and measured token counts.
The resolved artifact is validated before loading, and the command rejects token IDs outside the
model vocabulary. As with the determinism command, `--model /path/to/model.gguf` is the explicit
unregistered alternative.

## ModelJar-backed comparison run

Use a catalog alias for pure-Java measurements that must apply and record an exact model-scoped
execution plan:

```shell
./gradlew :models-bench:installDist
models-bench/build/install/models-bench/bin/models-bench \
  --backend pure-java \
  --modeljar qwen3_0_6b_q4_0 \
  --prompt-file models-bench/prompts/completion.txt \
  --max-tokens 64 \
  --warmups 2 \
  --iterations 10 \
  --context 2048 \
  --output build/reports/inference/qwen3_0_6b_q4_0-pure-java.json
```

Benchmark schema 5 records `backendDiagnostics`, including the pure-Java plan version, runtime
fingerprint, optimization decisions, and any enabled ModelJar performance profile. External
Ollama and llama.cpp reports record explicit `unavailable` diagnostics.
