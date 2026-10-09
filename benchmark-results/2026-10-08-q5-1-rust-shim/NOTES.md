# A Rust shim for Q5_1, and what it is and is not worth

Date: 2026-10-08. Box: i9-14900HX, AVX2, **no AVX-512**, 12 threads reported to the shim.
Shim: `backend-native` `jmodels-kernels` at ABI 6, capability bit 23.

Every number here is reproduced by the commands in **Reproduce**; the raw output of each is in
`raw/`.

## Why Q5_1 needed a kernel at all

Q5_1 was unloadable before this work, so it had no shim and no Java kernel either. Once the Java
side could load it, the question became how much of the matmul it is. `count_types.py` reads the
tensor table of the published artifacts:

`./shares.py` derives this from those dumps rather than it being typed in -- the first hand-typed
version of this table had the wrong denominator:

| artifact | Q5_1 tensors | Q5_1 weights | matmul weights | share |
| --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 Q5_K_S | 30 of 37 | 7,077,888 | 10,813,440 | **65.5%** |
| all-MiniLM-L6-v2 Q4_K_S | 4 of 37 | 589,824 | 10,813,440 | **5.5%** |

`token_embd.weight` is excluded from the denominator: it is the widest tensor in the file and it is
read by lookup, not multiplied. F32 norms are excluded as vectors rather than matrices.

The first measurement contradicted a guess made from the `Q4_K_S` file alone: in the `Q5_K_S` and
`Q5_K_M` builds Q5_1 is the dominant matmul type, not a rounding error. That is what justified the
kernel, and reading one file would have killed it.

## The constraint that set the design

Every other quantized shim here dots in Q8_0 or Q8_K integers, which is why a shim and the Java
kernel can agree bit for bit: integer accumulation is exact and associative, so partitioning rows
across workers cannot move a result. Q5_1 carries a per-block **minimum** instead of centring its
quants, and the reference pairs it with Q8_1 for the activation block sums that minimum needs.
Nothing on the Java side carries a Q8_1 block sum.

Inventing a second arithmetic for one format would have made a published vector depend on whether
a host loaded the shim. So the shim reproduces the Java kernel instead: it dequantizes the row and
folds in F32 along `PinnedReduction.dot`'s pinned order. That order is specified, not incidental --
four eight-lane accumulators stepping thirty-two elements, `(acc1+acc2) + (acc3+acc4)`, then
`((l4+l0) + (l6+l2)) + ((l5+l1) + (l7+l3))` -- and the species is pinned to 256 bits, which is
what makes an AVX2 register the right width rather than a coincidence. One Q5_1 block is exactly
thirty-two elements, so one block is one iteration of that loop and no row buffer is needed.

Two consequences, both deliberate:

- **`q * d + m` is a multiply and a separate add, never fused.** Java writes it unfused; fusing it
  in the shim would change the last bits.
- **There is no grouped Q5_1 kernel.** Grouping exists to share one quantized activation between
  matrices, and this format has none to share. The shim refuses such a group with
  `STATUS_INVALID_SHAPE` rather than reading an empty activation slice and returning zeros, and
  the Java side excludes Q5_1 from `groupable`.

## Parity: measured, not argued

- Rust: the scalar and AVX2 row kernels agree on `to_bits()`, not within a tolerance, over
  pseudo-random blocks whose scale, minimum and fifth-bit plane all vary per block.
- Java to shim: `NativeKernelLibraryTest` asserts `isEqualTo`, not `isCloseTo`, against
  `TensorOps.ggufExactBatchedMatmul`, batched and single-token. Every other quantized type in that
  file is compared within a tolerance; Q5_1 is the exception on purpose.
- End to end, on a real artifact: all-MiniLM-L6-v2 Q5_K_S through the embedding gate, with and
  without `backend-native` on the classpath, gives
  `min 0.9756330  mean 0.9967763  max component delta 0.111217` in **both** arms, identical to
  every digit printed. See `raw/end-to-end-ab.txt`.

## Speed: a kernel win and an end-to-end null

Kernel level, at the shapes these artifacts actually use -- `attn_q/k/v/output` are 384x384 and
`ffn_up` is 1536x384 -- 500 calls per trial, 9 trials, outputs compared element by element inside
the harness so a wrong-but-fast arm cannot score:

| shape | batch | java min | java mean | shim min | shim mean | min ratio | mean ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 384x384 | 8 | 0.2786 | 0.2834 | 0.2018 | 0.3648 | 1.38x | **0.78x** |
| 384x384 | 64 | 1.1418 | 1.1577 | 0.6260 | 0.9888 | 1.82x | 1.17x |
| 384x384 | 256 | 4.3150 | 4.3594 | 2.3147 | 2.6656 | 1.86x | 1.64x |
| 384x384 | 512 | 8.1026 | 8.1860 | 4.3023 | 4.7112 | 1.88x | 1.74x |
| 1536x384 | 8 | 1.1341 | 1.1824 | 0.3708 | 0.5652 | 3.06x | 2.09x |
| 1536x384 | 64 | 4.6105 | 4.6817 | 2.6000 | 2.8189 | 1.77x | 1.66x |
| 1536x384 | 256 | 17.8797 | 18.3382 | 8.0628 | 8.8527 | 2.22x | 2.07x |
| 1536x384 | 512 | 34.7836 | 35.1697 | 15.4775 | 16.8316 | 2.25x | 2.09x |

Milliseconds per call. The mean column is reported because it disagrees with the minimum in one
regime: on the **narrow** 384x384 projection at batch 8 the shim's mean is worse than Java's
(0.3648 against 0.2834) even though its best case is better, which is worker-pool overhead on a
job too small to amortize a barrier across twelve threads. Part of the win at the larger shapes is
threading rather than SIMD; that is the shipped configuration, but it is not all vector width.

**The end-to-end result is a null.** The same artifact through the embedding gate takes 1.6/1.6/1.7s
with the shim and 1.4/1.4/1.5s without it -- if anything slightly worse, and within run-to-run
noise either way. Eight short probes on a six-layer 384-wide model is not enough matmul to see a
2x kernel; model load and tokenization dominate. A faster kernel nobody experiences is worth
nothing, so this is recorded as a null rather than as a speedup.

Where it should be visible, by the table above rather than by assertion: sequences in the hundreds
of tokens, which is document embedding rather than query embedding. **That has not been measured
end to end here**, and until it is, the honest claim is "1.4-2.2x at the kernel, no observable
end-to-end effect on short-probe embedding".

## Reproduce

    # kernel microbenchmark (writes raw/kernel-microbenchmark.txt)
    javac --add-modules jdk.incubator.vector \
      -cp "models-bench/build/install/models-bench/lib/*:backend-native/build/libs/backend-native-0.3.54.jar" \
      -d . Q51Bench.java
    java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
      -cp "models-bench/build/install/models-bench/lib/*:backend-native/build/libs/backend-native-0.3.54.jar:." \
      Q51Bench backend-native/build/rust-target/release/libjmodels_kernels.dylib

    # weight shares
    ./count_types.py <artifact>.gguf > raw/weight-share-<name>.txt
    ./shares.py                        # the table above, from raw/

    # parity
    cargo test --release                       # in backend-native/src/main/rust/model-kernels
    ./gradlew :backend-native:test --tests "*NativeKernelLibraryTest*"
