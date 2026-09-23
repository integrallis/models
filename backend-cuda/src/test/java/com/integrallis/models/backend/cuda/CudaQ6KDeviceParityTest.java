/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.integrallis.models.backend.cuda;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * G1 parity: a Q6_K projection must be bit-exact against the CPU control on a real device, at every
 * row width, with no attention and no MoE routing anywhere in the picture.
 *
 * <p>Written to reproduce the G1 parity failure on Gemma 4 26B-A4B IT Q4_K_M (an MoE model), where
 * the leading hypothesis was a missing per-expert weight offset. That was falsified by two
 * independent pieces of device evidence, both gathered on an NVIDIA A40 (compute capability 8.6):
 *
 * <ul>
 *   <li>A bisection over Q4_K row widths from 1 to 48 super-blocks &mdash; spanning the 32-lane
 *       warp boundary where a lane starts owning more than one super-block &mdash; is bit-exact at
 *       every width. There is no multi-block-per-lane defect in Q4_K.
 *   <li>The G1 parity gate was re-run against a fully <b>dense</b> Q4_K_M model (Granite 4.1 3B, no
 *       MoE code touched at all) and it <b>also</b> failed, at prompt 0 token 0, with the same
 *       "both attention and projections routed" signature as the original Gemma 4 report.
 * </ul>
 *
 * <p><b>The root cause.</b> {@code models} carries two Q6_K row reductions on the CPU and they are
 * not bit-identical to each other. {@code PanamaVectorUtilSupport.ggufQ6_KQ8_KMatVecDot} reduces
 * each super-block to one exact {@code int} and folds it with one {@code fma} into one {@code
 * float} accumulator. {@code VectorUtilSupport.ggufQ6_KQ8_KScalarRowDot} &mdash; the {@code
 * VECTOR_BITSIZE < 256} fallback, mirrored by {@code dot_q6_k_q8_k_row_scalar} in {@code
 * backend-native} &mdash; keeps eight {@code float} lane accumulators instead. The device kernel
 * was transcribed from the scalar one; the CPU control it is measured against is the Panama one,
 * because {@code TensorOps.ggufMatmul} takes that path on any host with 256-bit vectors, which is
 * every host that can also run CUDA. The two folds coincide for a row of one super-block and differ
 * by one ULP from two super-blocks on.
 *
 * <p>That is why a one-super-block test passed for weeks while the gate failed: <b>one super-block
 * proves almost nothing about a fold</b>. Hence the bisection below, at 1, 2, 3, 32, 33 and 48
 * super-blocks &mdash; both sides of the 32-lane warp boundary, and the first widths at which a
 * lane owns more than one super-block.
 *
 * <p>The fixture is two real, consecutive Q6_K super-blocks lifted verbatim from Granite's {@code
 * token_embd.weight}; wider rows tile them with a deterministic per-block rotation of the 192
 * quantised-value bytes, so every row is built from genuine trained scale and {@code d} fields
 * rather than hand-rolled ones, and no two super-blocks in a row are identical.
 *
 * <p><b>This is a device-only defect.</b> The Rust host tests in {@code models-cuda-kernels} now
 * cover it too (they were comparing against the wrong CPU reduction, which is exactly why they
 * passed), but only a test that actually launches the kernel can prove what the arithmetic does
 * when compiled for {@code nvptx64-nvidia-cuda} and executed on real hardware. That is this test.
 */
class CudaQ6KDeviceParityTest {

  // Two real, consecutive Q6_K super-blocks (210 bytes each) read from Granite 4.1 3B Q4_K_M's
  // token_embd.weight tensor (sha256
  // 662b0626cd58f443baea23559b469df6576a81d349649c59413b36a9fb32eb29).
  // Genuine trained-weight bytes, not hand-rolled, so every bit pattern this test dequantises is
  // one
  // the kernel has to handle in production. Which tensor they came from is otherwise irrelevant: a
  // Q6_K super-block stands on its own, and the defect reproduces with any two real super-blocks.
  private static final int BLOCK_BYTES = 210;
  private static final int QK_K = 256;

  /** Bytes of a Q6_K super-block holding quantised values; the 16 scales and {@code d} follow. */
  private static final int QUANT_BYTES = 192;

  private static final byte[] TWO_REAL_Q6K_BLOCKS = {
    (byte) 0x7b,
    (byte) 0x0f,
    (byte) 0xf2,
    (byte) 0x3d,
    (byte) 0xad,
    (byte) 0x34,
    (byte) 0x9c,
    (byte) 0x82,
    (byte) 0x63,
    (byte) 0x5d,
    (byte) 0x45,
    (byte) 0x25,
    (byte) 0xd5,
    (byte) 0x31,
    (byte) 0x4a,
    (byte) 0x06,
    (byte) 0xb6,
    (byte) 0x20,
    (byte) 0xc6,
    (byte) 0x5f,
    (byte) 0x1e,
    (byte) 0x5f,
    (byte) 0xed,
    (byte) 0x99,
    (byte) 0x3c,
    (byte) 0x2b,
    (byte) 0x3e,
    (byte) 0x49,
    (byte) 0x07,
    (byte) 0xab,
    (byte) 0x97,
    (byte) 0xc7,
    (byte) 0x71,
    (byte) 0xc7,
    (byte) 0x2d,
    (byte) 0x19,
    (byte) 0xec,
    (byte) 0x79,
    (byte) 0x60,
    (byte) 0x07,
    (byte) 0xee,
    (byte) 0x2c,
    (byte) 0x12,
    (byte) 0x23,
    (byte) 0x4a,
    (byte) 0x58,
    (byte) 0x18,
    (byte) 0x5f,
    (byte) 0x02,
    (byte) 0xe7,
    (byte) 0x8e,
    (byte) 0x45,
    (byte) 0xb0,
    (byte) 0x3b,
    (byte) 0xa9,
    (byte) 0xc3,
    (byte) 0x76,
    (byte) 0xcc,
    (byte) 0xd0,
    (byte) 0x58,
    (byte) 0x60,
    (byte) 0x52,
    (byte) 0x1d,
    (byte) 0xce,
    (byte) 0xe5,
    (byte) 0xc1,
    (byte) 0xde,
    (byte) 0xf4,
    (byte) 0x42,
    (byte) 0x87,
    (byte) 0x8e,
    (byte) 0x45,
    (byte) 0xc6,
    (byte) 0x79,
    (byte) 0xce,
    (byte) 0x14,
    (byte) 0xb8,
    (byte) 0xd6,
    (byte) 0x32,
    (byte) 0x1d,
    (byte) 0x4b,
    (byte) 0x2d,
    (byte) 0x8a,
    (byte) 0x18,
    (byte) 0xa5,
    (byte) 0x9a,
    (byte) 0xe5,
    (byte) 0x56,
    (byte) 0x07,
    (byte) 0xc1,
    (byte) 0x9a,
    (byte) 0xad,
    (byte) 0x2e,
    (byte) 0xe8,
    (byte) 0xef,
    (byte) 0x60,
    (byte) 0x2c,
    (byte) 0xe5,
    (byte) 0x42,
    (byte) 0x17,
    (byte) 0xd0,
    (byte) 0x02,
    (byte) 0x56,
    (byte) 0x0a,
    (byte) 0xea,
    (byte) 0x60,
    (byte) 0x32,
    (byte) 0x76,
    (byte) 0x8a,
    (byte) 0x67,
    (byte) 0x01,
    (byte) 0xed,
    (byte) 0xab,
    (byte) 0xf0,
    (byte) 0x0f,
    (byte) 0xf8,
    (byte) 0x3c,
    (byte) 0xa5,
    (byte) 0x2e,
    (byte) 0x49,
    (byte) 0x78,
    (byte) 0x18,
    (byte) 0xb1,
    (byte) 0xd7,
    (byte) 0xa8,
    (byte) 0xe3,
    (byte) 0x00,
    (byte) 0x16,
    (byte) 0xa1,
    (byte) 0xaa,
    (byte) 0x2b,
    (byte) 0x26,
    (byte) 0x64,
    (byte) 0x7b,
    (byte) 0x9a,
    (byte) 0x19,
    (byte) 0x65,
    (byte) 0x55,
    (byte) 0xba,
    (byte) 0xbe,
    (byte) 0xca,
    (byte) 0x54,
    (byte) 0xa4,
    (byte) 0x89,
    (byte) 0x29,
    (byte) 0x58,
    (byte) 0x67,
    (byte) 0x9a,
    (byte) 0x65,
    (byte) 0x6b,
    (byte) 0x52,
    (byte) 0xab,
    (byte) 0x54,
    (byte) 0x65,
    (byte) 0x70,
    (byte) 0xad,
    (byte) 0x4a,
    (byte) 0x99,
    (byte) 0x9d,
    (byte) 0xdb,
    (byte) 0xc2,
    (byte) 0x50,
    (byte) 0x56,
    (byte) 0x9b,
    (byte) 0x5d,
    (byte) 0xa5,
    (byte) 0x52,
    (byte) 0x23,
    (byte) 0xd1,
    (byte) 0x61,
    (byte) 0x46,
    (byte) 0x8c,
    (byte) 0x66,
    (byte) 0x95,
    (byte) 0xea,
    (byte) 0x60,
    (byte) 0xe1,
    (byte) 0x91,
    (byte) 0x5f,
    (byte) 0x4a,
    (byte) 0xc6,
    (byte) 0xa2,
    (byte) 0xa9,
    (byte) 0xbd,
    (byte) 0xa6,
    (byte) 0x88,
    (byte) 0x31,
    (byte) 0x0a,
    (byte) 0xba,
    (byte) 0x64,
    (byte) 0xd7,
    (byte) 0x22,
    (byte) 0xbb,
    (byte) 0x4d,
    (byte) 0x96,
    (byte) 0x62,
    (byte) 0x5a,
    (byte) 0x84,
    (byte) 0xbe,
    (byte) 0xb4,
    (byte) 0x5a,
    (byte) 0x48,
    (byte) 0x4a,
    (byte) 0x5e,
    (byte) 0x80,
    (byte) 0xa7,
    (byte) 0x61,
    (byte) 0x49,
    (byte) 0xb9,
    (byte) 0x01,
    (byte) 0xa5,
    (byte) 0x20,
    (byte) 0xe5,
    (byte) 0x5f,
    (byte) 0xd1,
    (byte) 0xed,
    (byte) 0x2d,
    (byte) 0x8b,
    (byte) 0x48,
    (byte) 0xaa,
    (byte) 0xde,
    (byte) 0x3d,
    (byte) 0x6a,
    (byte) 0x0f,
    (byte) 0x65,
    (byte) 0xd3,
    (byte) 0xa2,
    (byte) 0x8d,
    (byte) 0x5e,
    (byte) 0xd1,
    (byte) 0xe1,
    (byte) 0x30,
    (byte) 0xfe,
    (byte) 0xeb,
    // --- block 1 ---
    (byte) 0x85,
    (byte) 0x71,
    (byte) 0xca,
    (byte) 0xab,
    (byte) 0x06,
    (byte) 0x83,
    (byte) 0xb6,
    (byte) 0x7d,
    (byte) 0xe1,
    (byte) 0x2b,
    (byte) 0x16,
    (byte) 0x11,
    (byte) 0xa1,
    (byte) 0x43,
    (byte) 0x72,
    (byte) 0x87,
    (byte) 0x5a,
    (byte) 0x81,
    (byte) 0x52,
    (byte) 0xbe,
    (byte) 0x51,
    (byte) 0xb6,
    (byte) 0xbd,
    (byte) 0x4d,
    (byte) 0x6f,
    (byte) 0xd1,
    (byte) 0x82,
    (byte) 0xcb,
    (byte) 0x87,
    (byte) 0x3f,
    (byte) 0x1c,
    (byte) 0x9d,
    (byte) 0x3c,
    (byte) 0x66,
    (byte) 0x01,
    (byte) 0xdf,
    (byte) 0xdd,
    (byte) 0x13,
    (byte) 0x6a,
    (byte) 0x13,
    (byte) 0x03,
    (byte) 0xce,
    (byte) 0xc5,
    (byte) 0x0e,
    (byte) 0xe4,
    (byte) 0x38,
    (byte) 0x0b,
    (byte) 0x80,
    (byte) 0x21,
    (byte) 0x0b,
    (byte) 0xc9,
    (byte) 0xf7,
    (byte) 0x68,
    (byte) 0xbd,
    (byte) 0x25,
    (byte) 0xed,
    (byte) 0x2b,
    (byte) 0xb1,
    (byte) 0x9b,
    (byte) 0x69,
    (byte) 0x59,
    (byte) 0x54,
    (byte) 0xee,
    (byte) 0xb0,
    (byte) 0xe9,
    (byte) 0x69,
    (byte) 0xef,
    (byte) 0xa5,
    (byte) 0x76,
    (byte) 0xb5,
    (byte) 0x19,
    (byte) 0x50,
    (byte) 0x10,
    (byte) 0xa2,
    (byte) 0x09,
    (byte) 0x9a,
    (byte) 0x71,
    (byte) 0xf9,
    (byte) 0x3d,
    (byte) 0x54,
    (byte) 0x00,
    (byte) 0x1f,
    (byte) 0x7f,
    (byte) 0xb5,
    (byte) 0xb2,
    (byte) 0x51,
    (byte) 0x0f,
    (byte) 0xa7,
    (byte) 0x1e,
    (byte) 0x44,
    (byte) 0xb3,
    (byte) 0x34,
    (byte) 0x2d,
    (byte) 0x23,
    (byte) 0xeb,
    (byte) 0x95,
    (byte) 0xcb,
    (byte) 0x30,
    (byte) 0x63,
    (byte) 0xe5,
    (byte) 0x16,
    (byte) 0x56,
    (byte) 0x1b,
    (byte) 0xf8,
    (byte) 0x26,
    (byte) 0xe1,
    (byte) 0x13,
    (byte) 0xa2,
    (byte) 0xa4,
    (byte) 0x9a,
    (byte) 0x3e,
    (byte) 0x5a,
    (byte) 0x15,
    (byte) 0x9a,
    (byte) 0x58,
    (byte) 0xab,
    (byte) 0xa9,
    (byte) 0x45,
    (byte) 0xae,
    (byte) 0x67,
    (byte) 0xe2,
    (byte) 0x51,
    (byte) 0x65,
    (byte) 0x95,
    (byte) 0x9c,
    (byte) 0x69,
    (byte) 0x15,
    (byte) 0x84,
    (byte) 0x66,
    (byte) 0xaf,
    (byte) 0xfa,
    (byte) 0xb4,
    (byte) 0x8e,
    (byte) 0x6a,
    (byte) 0x59,
    (byte) 0xca,
    (byte) 0xc1,
    (byte) 0xa9,
    (byte) 0xd7,
    (byte) 0xa0,
    (byte) 0x9f,
    (byte) 0x29,
    (byte) 0x45,
    (byte) 0xb4,
    (byte) 0xaf,
    (byte) 0x21,
    (byte) 0x65,
    (byte) 0x59,
    (byte) 0x59,
    (byte) 0xa4,
    (byte) 0xb7,
    (byte) 0x5c,
    (byte) 0x55,
    (byte) 0xd4,
    (byte) 0xd6,
    (byte) 0xe8,
    (byte) 0x61,
    (byte) 0x68,
    (byte) 0x51,
    (byte) 0x46,
    (byte) 0x08,
    (byte) 0x51,
    (byte) 0x74,
    (byte) 0x8c,
    (byte) 0x2a,
    (byte) 0x0b,
    (byte) 0x8b,
    (byte) 0x1a,
    (byte) 0x7d,
    (byte) 0x75,
    (byte) 0x6b,
    (byte) 0x99,
    (byte) 0x83,
    (byte) 0x6a,
    (byte) 0x7b,
    (byte) 0xac,
    (byte) 0x4c,
    (byte) 0x80,
    (byte) 0x9c,
    (byte) 0xb2,
    (byte) 0xb5,
    (byte) 0x54,
    (byte) 0x7e,
    (byte) 0x49,
    (byte) 0x6a,
    (byte) 0x01,
  };

  static {
    if (TWO_REAL_Q6K_BLOCKS.length != 2 * BLOCK_BYTES) {
      throw new ExceptionInInitializerError(
          "fixture must be exactly two Q6_K super-blocks: "
              + TWO_REAL_Q6K_BLOCKS.length
              + " != "
              + 2 * BLOCK_BYTES);
    }
  }

  /**
   * The bisection. One super-block is the width at which the two CPU folds happen to agree, so it
   * is the baseline rather than the evidence; 2 and 3 are the first widths that separate them; 32,
   * 33 and 48 straddle the warp boundary, where a lane starts owning more than one super-block.
   */
  @ParameterizedTest(name = "{0} super-block(s)")
  @ValueSource(ints = {1, 2, 3, 32, 33, 48})
  @DisplayName("a Q6_K row projection is bit-exact against the CPU control at every width")
  void q6kRowProjectionIsBitExactAtEveryWidth(int superBlocks) {
    CudaGgufBatchedMatrixKernel.Status status = CudaGgufBatchedMatrixKernel.open();
    assumeTrue(
        status.accelerated(),
        "requires a CUDA device; not evidence of anything off-device: " + status.reason());
    CudaGgufBatchedMatrixKernel kernel = status.kernel().orElseThrow();
    try (Arena arena = Arena.ofShared()) {
      int cols = superBlocks * QK_K;
      MemorySegment weights = weightsFor(arena, superBlocks);
      float[] input = activations(42, cols);

      float[] cpu = new float[1];
      TensorOps.ggufMatmul(cpu, input, weights, GgufTensorType.Q6_K, 1, cols);
      float[] gpu = new float[1];
      kernel.multiply(gpu, input, weights, GgufTensorType.Q6_K, 1, 1, cols);

      assertEquals(
          Float.floatToRawIntBits(cpu[0]),
          Float.floatToRawIntBits(gpu[0]),
          () ->
              "Q6_K row projection is not bit-exact at "
                  + superBlocks
                  + " super-block(s): cpu="
                  + cpu[0]
                  + " (bits "
                  + Float.floatToRawIntBits(cpu[0])
                  + ") gpu="
                  + gpu[0]
                  + " (bits "
                  + Float.floatToRawIntBits(gpu[0])
                  + ", relative delta "
                  + Math.abs((cpu[0] - gpu[0]) / cpu[0])
                  + "); G1 admits no token-level tolerance -- see class Javadoc");
    } finally {
      kernel.close();
    }
  }

  /**
   * The exact case that failed, pinned with the numbers it failed with.
   *
   * <p>Kept separate from the bisection so the regression is unmistakable in a failure report:
   * measured on an NVIDIA A40 (cc 8.6) on 2026-09-19 as cpu bits {@code -1098673107} against gpu
   * bits {@code -1098673106}, one ULP apart, which compounded over 30 transformer layers into a
   * flipped argmax at prompt 0, token 0.
   */
  @Test
  @DisplayName("the two-super-block row that failed G1 is bit-exact again")
  void theTwoSuperBlockRowThatFailedG1IsBitExact() {
    CudaGgufBatchedMatrixKernel.Status status = CudaGgufBatchedMatrixKernel.open();
    assumeTrue(
        status.accelerated(),
        "requires a CUDA device; not evidence of anything off-device: " + status.reason());
    CudaGgufBatchedMatrixKernel kernel = status.kernel().orElseThrow();
    try (Arena arena = Arena.ofShared()) {
      int cols = 2 * QK_K;
      MemorySegment weights = arena.allocate(2L * BLOCK_BYTES);
      MemorySegment.copy(
          TWO_REAL_Q6K_BLOCKS, 0, weights, ValueLayout.JAVA_BYTE, 0, 2 * BLOCK_BYTES);
      float[] input = activations(42, cols);

      float[] cpu = new float[1];
      TensorOps.ggufMatmul(cpu, input, weights, GgufTensorType.Q6_K, 1, cols);
      float[] gpu = new float[1];
      kernel.multiply(gpu, input, weights, GgufTensorType.Q6_K, 1, 1, cols);

      assertEquals(
          -1098673107,
          Float.floatToRawIntBits(cpu[0]),
          () ->
              "the CPU control moved: this test pins the exact G1 repro, so a changed control "
                  + "means the fixture or the CPU path changed and the pinned gpu comparison below "
                  + "no longer reproduces anything. cpu="
                  + cpu[0]);
      assertEquals(
          Float.floatToRawIntBits(cpu[0]),
          Float.floatToRawIntBits(gpu[0]),
          () ->
              "the G1 root-cause row diverged again: cpu="
                  + cpu[0]
                  + " gpu="
                  + gpu[0]
                  + " (the original failure was gpu bits -1098673106)");
    } finally {
      kernel.close();
    }
  }

  /**
   * A row of {@code superBlocks} Q6_K super-blocks built from the two real ones.
   *
   * <p>Blocks 0 and 1 are the real bytes verbatim. Block {@code i >= 2} is real block {@code i % 2}
   * with its 192 quantised-value bytes rotated by {@code i}, which keeps the genuine trained scales
   * and {@code d} while making every super-block in the row distinct, so a fold that silently
   * reused one block's partials could not pass.
   */
  private static MemorySegment weightsFor(Arena arena, int superBlocks) {
    byte[] row = new byte[superBlocks * BLOCK_BYTES];
    for (int block = 0; block < superBlocks; block++) {
      int source = (block % 2) * BLOCK_BYTES;
      int target = block * BLOCK_BYTES;
      System.arraycopy(TWO_REAL_Q6K_BLOCKS, source, row, target, BLOCK_BYTES);
      if (block < 2) {
        continue;
      }
      for (int index = 0; index < QUANT_BYTES; index++) {
        row[target + index] = TWO_REAL_Q6K_BLOCKS[source + (index + block) % QUANT_BYTES];
      }
    }
    MemorySegment weights = arena.allocate(row.length);
    MemorySegment.copy(row, 0, weights, ValueLayout.JAVA_BYTE, 0, row.length);
    return weights;
  }

  private static float[] activations(long seed, int count) {
    Random random = new Random(seed);
    float[] values = new float[count];
    for (int i = 0; i < count; i++) {
      values[i] = (random.nextFloat() - 0.5f) * 2.0f;
    }
    return values;
  }
}
