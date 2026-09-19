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

/**
 * G1 root cause: the Q6_K projection kernel is not bit-exact against the CPU control, even with no
 * attention and no MoE routing anywhere in the picture.
 *
 * <p>Found chasing down the G1 parity failure on Gemma 4 26B-A4B IT Q4_K_M (an MoE model): the
 * leading hypothesis was a missing per-expert weight offset. It is falsified by two independent
 * pieces of device evidence, both gathered on an NVIDIA A40 (compute capability 8.6):
 *
 * <ul>
 *   <li>A bisection over Q4_K row widths from 1 to 48 super-blocks — spanning the 32-lane warp
 *       boundary where a lane starts owning more than one super-block — is bit-exact at every
 *       width. There is no multi-block-per-lane defect in Q4_K.
 *   <li>The G1 parity gate was re-run against a fully <b>dense</b> Q4_K_M model (Granite 4.1 3B,
 *       no MoE code touched at all) and it <b>also</b> fails, at prompt 0 token 0, with the same
 *       "both attention and projections routed" signature as the original Gemma 4 report.
 * </ul>
 *
 * <p>Bisecting Q6_K the same way isolates it further: a single super-block is exact, but as soon as
 * a second super-block joins the row — the point where the fused kernel's one-lane, ascending-order
 * float fold actually has more than one term to fold — the result stops matching the CPU control
 * bit-for-bit. This is reproduced here with two real Q6_K super-blocks lifted verbatim from
 * Granite's {@code token_embd.weight} (any two real super-blocks would do; the fault is in the
 * fold, not the data) and a fixed-seed activation, with no model, no forward pass, no attention, and
 * no MoE routing anywhere in the call path.
 *
 * <p><b>This is a device-only defect.</b> The 21 Rust host tests in {@code models-cuda-kernels}
 * (bit-exact against a scalar oracle, run on every host with {@code cargo test --release}) cover
 * {@code kquant.rs}'s shared arithmetic compiled for the <em>host</em> target. They cannot see this:
 * the divergence is between what that arithmetic produces when it is compiled for {@code
 * nvptx64-nvidia-cuda} and executed on a real device, and what the same row projection produces on
 * the CPU control path. Nothing off-device can catch that class of bug; only a test that actually
 * launches the kernel, like this one, can.
 */
class CudaQ6KDeviceParityTest {

  // Two real, consecutive Q6_K super-blocks (210 bytes each) read from Granite 4.1 3B Q4_K_M's
  // token_embd.weight tensor (sha256 662b0626cd58f443baea23559b469df6576a81d349649c59413b36a9fb32eb29).
  // Genuine trained-weight bytes, not hand-rolled, so every bit pattern this test dequantises is one
  // the kernel has to handle in production. Which tensor they came from is otherwise irrelevant: a
  // Q6_K super-block stands on its own, and the defect reproduces with any two real super-blocks.
  private static final int BLOCK_BYTES = 210;
  private static final int QK_K = 256;

  private static final byte[] TWO_REAL_Q6K_BLOCKS = {
    (byte) 0x7b, (byte) 0x0f, (byte) 0xf2, (byte) 0x3d, (byte) 0xad, (byte) 0x34, (byte) 0x9c,
    (byte) 0x82, (byte) 0x63, (byte) 0x5d, (byte) 0x45, (byte) 0x25, (byte) 0xd5, (byte) 0x31,
    (byte) 0x4a, (byte) 0x06, (byte) 0xb6, (byte) 0x20, (byte) 0xc6, (byte) 0x5f, (byte) 0x1e,
    (byte) 0x5f, (byte) 0xed, (byte) 0x99, (byte) 0x3c, (byte) 0x2b, (byte) 0x3e, (byte) 0x49,
    (byte) 0x07, (byte) 0xab, (byte) 0x97, (byte) 0xc7, (byte) 0x71, (byte) 0xc7, (byte) 0x2d,
    (byte) 0x19, (byte) 0xec, (byte) 0x79, (byte) 0x60, (byte) 0x07, (byte) 0xee, (byte) 0x2c,
    (byte) 0x12, (byte) 0x23, (byte) 0x4a, (byte) 0x58, (byte) 0x18, (byte) 0x5f, (byte) 0x02,
    (byte) 0xe7, (byte) 0x8e, (byte) 0x45, (byte) 0xb0, (byte) 0x3b, (byte) 0xa9, (byte) 0xc3,
    (byte) 0x76, (byte) 0xcc, (byte) 0xd0, (byte) 0x58, (byte) 0x60, (byte) 0x52, (byte) 0x1d,
    (byte) 0xce, (byte) 0xe5, (byte) 0xc1, (byte) 0xde, (byte) 0xf4, (byte) 0x42, (byte) 0x87,
    (byte) 0x8e, (byte) 0x45, (byte) 0xc6, (byte) 0x79, (byte) 0xce, (byte) 0x14, (byte) 0xb8,
    (byte) 0xd6, (byte) 0x32, (byte) 0x1d, (byte) 0x4b, (byte) 0x2d, (byte) 0x8a, (byte) 0x18,
    (byte) 0xa5, (byte) 0x9a, (byte) 0xe5, (byte) 0x56, (byte) 0x07, (byte) 0xc1, (byte) 0x9a,
    (byte) 0xad, (byte) 0x2e, (byte) 0xe8, (byte) 0xef, (byte) 0x60, (byte) 0x2c, (byte) 0xe5,
    (byte) 0x42, (byte) 0x17, (byte) 0xd0, (byte) 0x02, (byte) 0x56, (byte) 0x0a, (byte) 0xea,
    (byte) 0x60, (byte) 0x32, (byte) 0x76, (byte) 0x8a, (byte) 0x67, (byte) 0x01, (byte) 0xed,
    (byte) 0xab, (byte) 0xf0, (byte) 0x0f, (byte) 0xf8, (byte) 0x3c, (byte) 0xa5, (byte) 0x2e,
    (byte) 0x49, (byte) 0x78, (byte) 0x18, (byte) 0xb1, (byte) 0xd7, (byte) 0xa8, (byte) 0xe3,
    (byte) 0x00, (byte) 0x16, (byte) 0xa1, (byte) 0xaa, (byte) 0x2b, (byte) 0x26, (byte) 0x64,
    (byte) 0x7b, (byte) 0x9a, (byte) 0x19, (byte) 0x65, (byte) 0x55, (byte) 0xba, (byte) 0xbe,
    (byte) 0xca, (byte) 0x54, (byte) 0xa4, (byte) 0x89, (byte) 0x29, (byte) 0x58, (byte) 0x67,
    (byte) 0x9a, (byte) 0x65, (byte) 0x6b, (byte) 0x52, (byte) 0xab, (byte) 0x54, (byte) 0x65,
    (byte) 0x70, (byte) 0xad, (byte) 0x4a, (byte) 0x99, (byte) 0x9d, (byte) 0xdb, (byte) 0xc2,
    (byte) 0x50, (byte) 0x56, (byte) 0x9b, (byte) 0x5d, (byte) 0xa5, (byte) 0x52, (byte) 0x23,
    (byte) 0xd1, (byte) 0x61, (byte) 0x46, (byte) 0x8c, (byte) 0x66, (byte) 0x95, (byte) 0xea,
    (byte) 0x60, (byte) 0xe1, (byte) 0x91, (byte) 0x5f, (byte) 0x4a, (byte) 0xc6, (byte) 0xa2,
    (byte) 0xa9, (byte) 0xbd, (byte) 0xa6, (byte) 0x88, (byte) 0x31, (byte) 0x0a, (byte) 0xba,
    (byte) 0x64, (byte) 0xd7, (byte) 0x22, (byte) 0xbb, (byte) 0x4d, (byte) 0x96, (byte) 0x62,
    (byte) 0x5a, (byte) 0x84, (byte) 0xbe, (byte) 0xb4, (byte) 0x5a, (byte) 0x48, (byte) 0x4a,
    (byte) 0x5e, (byte) 0x80, (byte) 0xa7, (byte) 0x61, (byte) 0x49, (byte) 0xb9, (byte) 0x01,
    (byte) 0xa5, (byte) 0x20, (byte) 0xe5, (byte) 0x5f, (byte) 0xd1, (byte) 0xed, (byte) 0x2d,
    (byte) 0x8b, (byte) 0x48, (byte) 0xaa, (byte) 0xde, (byte) 0x3d, (byte) 0x6a, (byte) 0x0f,
    (byte) 0x65, (byte) 0xd3, (byte) 0xa2, (byte) 0x8d, (byte) 0x5e, (byte) 0xd1, (byte) 0xe1,
    (byte) 0x30, (byte) 0xfe, (byte) 0xeb,
    // --- block 1 ---
    (byte) 0x85, (byte) 0x71, (byte) 0xca, (byte) 0xab, (byte) 0x06, (byte) 0x83, (byte) 0xb6,
    (byte) 0x7d, (byte) 0xe1, (byte) 0x2b, (byte) 0x16, (byte) 0x11, (byte) 0xa1, (byte) 0x43,
    (byte) 0x72, (byte) 0x87, (byte) 0x5a, (byte) 0x81, (byte) 0x52, (byte) 0xbe, (byte) 0x51,
    (byte) 0xb6, (byte) 0xbd, (byte) 0x4d, (byte) 0x6f, (byte) 0xd1, (byte) 0x82, (byte) 0xcb,
    (byte) 0x87, (byte) 0x3f, (byte) 0x1c, (byte) 0x9d, (byte) 0x3c, (byte) 0x66, (byte) 0x01,
    (byte) 0xdf, (byte) 0xdd, (byte) 0x13, (byte) 0x6a, (byte) 0x13, (byte) 0x03, (byte) 0xce,
    (byte) 0xc5, (byte) 0x0e, (byte) 0xe4, (byte) 0x38, (byte) 0x0b, (byte) 0x80, (byte) 0x21,
    (byte) 0x0b, (byte) 0xc9, (byte) 0xf7, (byte) 0x68, (byte) 0xbd, (byte) 0x25, (byte) 0xed,
    (byte) 0x2b, (byte) 0xb1, (byte) 0x9b, (byte) 0x69, (byte) 0x59, (byte) 0x54, (byte) 0xee,
    (byte) 0xb0, (byte) 0xe9, (byte) 0x69, (byte) 0xef, (byte) 0xa5, (byte) 0x76, (byte) 0xb5,
    (byte) 0x19, (byte) 0x50, (byte) 0x10, (byte) 0xa2, (byte) 0x09, (byte) 0x9a, (byte) 0x71,
    (byte) 0xf9, (byte) 0x3d, (byte) 0x54, (byte) 0x00, (byte) 0x1f, (byte) 0x7f, (byte) 0xb5,
    (byte) 0xb2, (byte) 0x51, (byte) 0x0f, (byte) 0xa7, (byte) 0x1e, (byte) 0x44, (byte) 0xb3,
    (byte) 0x34, (byte) 0x2d, (byte) 0x23, (byte) 0xeb, (byte) 0x95, (byte) 0xcb, (byte) 0x30,
    (byte) 0x63, (byte) 0xe5, (byte) 0x16, (byte) 0x56, (byte) 0x1b, (byte) 0xf8, (byte) 0x26,
    (byte) 0xe1, (byte) 0x13, (byte) 0xa2, (byte) 0xa4, (byte) 0x9a, (byte) 0x3e, (byte) 0x5a,
    (byte) 0x15, (byte) 0x9a, (byte) 0x58, (byte) 0xab, (byte) 0xa9, (byte) 0x45, (byte) 0xae,
    (byte) 0x67, (byte) 0xe2, (byte) 0x51, (byte) 0x65, (byte) 0x95, (byte) 0x9c, (byte) 0x69,
    (byte) 0x15, (byte) 0x84, (byte) 0x66, (byte) 0xaf, (byte) 0xfa, (byte) 0xb4, (byte) 0x8e,
    (byte) 0x6a, (byte) 0x59, (byte) 0xca, (byte) 0xc1, (byte) 0xa9, (byte) 0xd7, (byte) 0xa0,
    (byte) 0x9f, (byte) 0x29, (byte) 0x45, (byte) 0xb4, (byte) 0xaf, (byte) 0x21, (byte) 0x65,
    (byte) 0x59, (byte) 0x59, (byte) 0xa4, (byte) 0xb7, (byte) 0x5c, (byte) 0x55, (byte) 0xd4,
    (byte) 0xd6, (byte) 0xe8, (byte) 0x61, (byte) 0x68, (byte) 0x51, (byte) 0x46, (byte) 0x08,
    (byte) 0x51, (byte) 0x74, (byte) 0x8c, (byte) 0x2a, (byte) 0x0b, (byte) 0x8b, (byte) 0x1a,
    (byte) 0x7d, (byte) 0x75, (byte) 0x6b, (byte) 0x99, (byte) 0x83, (byte) 0x6a, (byte) 0x7b,
    (byte) 0xac, (byte) 0x4c, (byte) 0x80, (byte) 0x9c, (byte) 0xb2, (byte) 0xb5, (byte) 0x54,
    (byte) 0x7e, (byte) 0x49, (byte) 0x6a, (byte) 0x01,
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

  @Test
  @DisplayName("a single Q6_K super-block matches the CPU control bit-for-bit (baseline)")
  void oneSuperBlockIsExact() {
    CudaGgufBatchedMatrixKernel.Status status = CudaGgufBatchedMatrixKernel.open();
    assumeTrue(
        status.accelerated(), "requires a CUDA device; not evidence of anything off-device: " + status.reason());
    CudaGgufBatchedMatrixKernel kernel = status.kernel().orElseThrow();
    try (Arena arena = Arena.ofShared()) {
      MemorySegment weights = arena.allocate(BLOCK_BYTES);
      MemorySegment.copy(TWO_REAL_Q6K_BLOCKS, 0, weights, ValueLayout.JAVA_BYTE, 0, BLOCK_BYTES);
      float[] input = activations(42, QK_K);

      float[] cpu = new float[1];
      TensorOps.ggufMatmul(cpu, input, weights, GgufTensorType.Q6_K, 1, QK_K);
      float[] gpu = new float[1];
      kernel.multiply(gpu, input, weights, GgufTensorType.Q6_K, 1, 1, QK_K);

      assertEquals(
          Float.floatToRawIntBits(cpu[0]),
          Float.floatToRawIntBits(gpu[0]),
          () -> "one super-block should be exact by construction: cpu=" + cpu[0] + " gpu=" + gpu[0]);
    } finally {
      kernel.close();
    }
  }

  @Test
  @DisplayName(
      "G1 root cause: a two-super-block Q6_K row does not match the CPU control on the device")
  void twoSuperBlocksDivergeFromTheCpuControl() {
    CudaGgufBatchedMatrixKernel.Status status = CudaGgufBatchedMatrixKernel.open();
    assumeTrue(
        status.accelerated(), "requires a CUDA device; not evidence of anything off-device: " + status.reason());
    CudaGgufBatchedMatrixKernel kernel = status.kernel().orElseThrow();
    try (Arena arena = Arena.ofShared()) {
      int cols = 2 * QK_K;
      MemorySegment weights = arena.allocate(2L * BLOCK_BYTES);
      MemorySegment.copy(TWO_REAL_Q6K_BLOCKS, 0, weights, ValueLayout.JAVA_BYTE, 0, 2 * BLOCK_BYTES);
      float[] input = activations(42, cols);

      float[] cpu = new float[1];
      TensorOps.ggufMatmul(cpu, input, weights, GgufTensorType.Q6_K, 1, cols);
      float[] gpu = new float[1];
      kernel.multiply(gpu, input, weights, GgufTensorType.Q6_K, 1, 1, cols);

      // This is the actual G1 defect: the moment the one-lane ascending-order float fold has to
      // fold a second super-block's contribution, the device result stops being the CPU control's
      // bit pattern. Measured (2026-09-19, NVIDIA A40, cc 8.6): cpu=-0.256989866, gpu=-0.256989896.
      // That is what should be failing here until the kernel is fixed.
      assertEquals(
          Float.floatToRawIntBits(cpu[0]),
          Float.floatToRawIntBits(gpu[0]),
          () ->
              "Q6_K row projection is not bit-exact past one super-block: cpu="
                  + cpu[0]
                  + " gpu="
                  + gpu[0]
                  + " (relative delta "
                  + Math.abs((cpu[0] - gpu[0]) / cpu[0])
                  + "); this is a real kernel/CPU-reference disagreement, not test noise -- see "
                  + "class Javadoc");
    } finally {
      kernel.close();
    }
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
