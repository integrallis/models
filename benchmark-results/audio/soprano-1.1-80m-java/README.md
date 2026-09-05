# Soprano 1.1 80M Java-native qualification

Date: 2026-09-05

Status: **qualified** for the checksum-bound Q8_0 and BF16 artifacts.

## Question

Can Models synthesize speech from the standalone Soprano GGUF entirely inside the JVM, and which
Java-native changes materially close the performance gap to audio.cpp?

External code is used only as an independent oracle and benchmark peer. The product path parses,
tokenizes, runs the language model, samples acoustic tokens, executes the vocoder, performs ISTFT,
and emits PCM in Java.

## Final qualification inputs

| Input | Q8_0 | BF16 |
| --- | --- | --- |
| Artifact bytes | 123,162,336 | 221,809,792 |
| Artifact SHA-256 | `4758ad908395dc73a1b973d9a29ce96941f4328594d1c6c1223e7b7710a6a131` | `46c60cdf5b8e5a3b26bfd185c0870826c05b533979ea7c1d399a9ffb76a50a54` |
| Product backend | Models-owned Rust/FFM Q8 projection kernel | pure Java |
| Evidence | [q8_0-qualification.json](q8_0-qualification.json) | [bf16-qualification.json](bf16-qualification.json) |

Both artifacts come from `WalkingCat/Soprano-1.1-80M-GGUF` at
`36c6f47cf91421b7f0cf3d862d28ae2e41aab3f2`. The independent oracle is the official
Soprano PyTorch/Transformers implementation at
`12fac06eb8fa53bad8b3941d3cb11e9c869477c4`. Production inference does not load that code.

The controlled workload uses `The JVM can speak for itself.`, greedy sampling, one complete warmup,
then five true-streaming trials. It ran on an AWS `c7i.4xlarge` with 16 logical processors, an Intel
Xeon Platinum 8488C, Amazon Corretto 25.0.4.1, and a 2 GiB test heap. The gate requires at least
0.995 PCM cosine, 20 dB SDR, p95 RTF at most 2.0, p95 time to first audio at most 2,000 ms, and audio
before synthesis completes.

## Correctness and final performance

| Variant | PCM cosine | SDR | p95 TTFA | p95 RTF | Peak RSS upper bound | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Q8_0 / Rust-FFM | 0.998389315 | 24.923 dB | 676.696 ms | 1.225912 | 1.90 GiB | Qualified |
| BF16 / pure Java | 0.999331179 | 28.698 dB | 1,670.465 ms | 1.886551 | 1.21 GiB | Qualified |

The language-model probe also clears its independent oracle gate. Q8_0 reaches 0.999974035 logits
cosine and 0.999926520 hidden-state cosine; BF16 rounds to 1.0 for both. Every streaming trial emits
32 kHz mono PCM before completion and reconstructs the blocking result at 1.0 cosine.

The initial one-token streaming implementation decoded an overlapping vocoder window for every
acoustic token. On the controlled host that made an otherwise fast Q8 kernel report RTF 4.13-4.21.
A test-first change batches 12 stable acoustic frames per vocoder call while retaining the exact
PCM timeline. The Q8 kernel also recommends at most six workers for these small projections;
explicit deployment configuration still overrides the recommendation. Those changes moved Q8 to
RTF 1.18-1.23 and BF16 from RTF 2.00-2.41 to 1.84-1.89.

The reported RSS is deliberately conservative: GNU `time` wrapped the Gradle qualification
process, so the Q8 figure also includes first-invocation compilation. It is an upper bound, not a
claim that the mapped 117 MiB artifact becomes a 1.90 GiB resident model.

## Earlier profile

The first warmed Java profile identified Q8 matrix work, not FFT, as the dominant cost:
`q8_0Q8_0AccumulateBatchedBlock` represented 60.39% of sampled hot time, while FFT work was about
0.32%. That evidence led to the prepared-F32 vocoder and the narrow Q8 projection shim rather than
moving the graph, tokenizer, sampling, vocoder, or streaming lifecycle out of Java.

## Rejected experiments

| Experiment | Result | Finding |
| --- | ---: | --- |
| Signed-short Vector API reduction | Failed an all-127 overflow test before correction; corrected form was 61.628 ms versus 61.394 ms for the widened kernel. | `ShortVector.reduceLanesToLong(ADD)` reduces in 16-bit lanes before widening. Correct emulation does not improve this HotSpot profile. |
| Block-major activations and staged row kernel | 10.973 s total, 5.056 s vocoder, RTF 5.358. | Rearranging activations without a matching 2-D weight tile regresses the complete graph. |
| Row-major output scratch plus transpose | 12.204 s total, 5.430 s vocoder, RTF 5.959. | A small isolated-kernel signal did not survive the complete-model gate. |
| Physically packed Q/K/V rows | Bit-identical wave, but 13.252 s total and 8.747 s language-model time, RTF 6.471. | One larger projection loses more to the current Q8 scheduler/kernel than it saves in quantization and dispatch. |
| Prepared Q8 bytes and predecoded scales | 56.249 ms versus 58.374 ms interleaved at 253x2304x768, with overlapping confidence intervals. | A possible 3.6% gain does not justify 5.9% extra weight storage or a new production layout by itself. |
| Two-dimensional row/batch Q8 tiling | Best tile (4 rows x 8 batches) took 688.0 ms versus 322.0 ms for the production prequantized single-thread control: 2.14x slower. | Reusing each weight tile across batches cannot repay repeated weight loads per batch tile plus workspace and vector-array overhead. All 12 tile and tail shapes remained bit-exact. |

These rejected results remain part of the evidence so later work does not repeat plausible but
incorrect optimizations.

## Current interpretation

audio.cpp uses x86 VNNI or packed multiply-add instructions for signed byte dot products. Java
25's Vector API has no signed-byte dot-accumulate operation, no public VNNI/SDOT capability query,
and must widen byte lanes manually. The current 256-bit Java Q8 block performs four byte-to-int
conversions per operand. A useful JVM improvement would expose portable signed and unsigned INT8
dot-accumulate operations with INT32 accumulation and documented overflow semantics, lowering to
x86 VNNI and AArch64 SDOT/SMMLA.

A prepared-F32 control supplied the successful Java path. At 253x2304x768, decoding one Q8 matrix
costs 4.560 ms and expands it from 1.793 MiB to 6.750 MiB. The row-parallel Java F32 kernel runs in
15.769 ms versus 63.260 ms for Q8, a 4.01x kernel gain. The serial F32 control takes 86.739 ms, so
parallel row scheduling remains essential.

For Q8_0, the end-to-end runtime expands only the vocoder's 17 Q8 projections: 31,753,824 serialized
bytes become 119,543,808 execution bytes, an 87,789,984-byte (83.72 MiB) increase. The remaining
autoregressive Q8 projections use the narrow Models-owned kernel. BF16 needs no shim and proves the
same graph can meet policy entirely in Java when the artifact provides execution-friendly weights.

Both exact artifacts are catalog-qualified. Removing the Q8 shim still depends on a JVM-level INT8
dot-accumulate primitive or a Java kernel that clears the same complete-model gate.
