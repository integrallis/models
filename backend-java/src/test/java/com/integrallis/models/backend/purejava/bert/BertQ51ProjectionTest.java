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
package com.integrallis.models.backend.purejava.bert;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.gguf.*;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.lang.foreign.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class BertQ51ProjectionTest {
  @Test
  void injectedExactProjectionIsExecutedAndUnsupportedShapesRetainJava(@TempDir Path dir)
      throws Exception {
    var builder = new SyntheticGgufBuilder();
    addFloat(builder, "token_embd.weight", 32, 8, false);
    addFloat(builder, "position_embd.weight", 32, 16, false);
    addFloat(builder, "token_types.weight", 32, 2, false);
    addFloat(builder, "token_embd_norm.weight", 32, 1, true);
    addFloat(builder, "token_embd_norm.bias", 32, 1, false);
    for (String projection :
        List.of("attn_q", "attn_k", "attn_v", "attn_output", "ffn_up", "ffn_down")) {
      ByteBuffer bytes = ByteBuffer.allocate(32 * 24).order(ByteOrder.LITTLE_ENDIAN);
      for (int row = 0; row < 32; row++) {
        bytes.putShort((short) (0x2800 + row)).putShort((short) 0xa400).putInt(row * 0x1234567);
        for (int n = 0; n < 16; n++) bytes.put((byte) (row * 7 + n * 11));
      }
      builder.addTensor(
          "blk.0." + projection + ".weight",
          GgufTensorType.Q5_1,
          new long[] {32, 32},
          bytes.array());
      addFloat(builder, "blk.0." + projection + ".bias", 32, 1, false);
    }
    for (String norm : List.of("attn_output_norm", "layer_output_norm")) {
      addFloat(builder, "blk.0." + norm + ".weight", 32, 1, true);
      addFloat(builder, "blk.0." + norm + ".bias", 32, 1, false);
    }
    Path file = dir.resolve("bert.gguf");
    Files.write(file, builder.build());
    var config =
        new BertConfig(
            "bert",
            BertConfig.EncoderStyle.BERT,
            32,
            1,
            4,
            8,
            16,
            32,
            1e-5f,
            10000f,
            BertConfig.Pooling.MEAN);
    try (Arena arena = Arena.ofShared()) {
      var parsed = GgufParser.parse(file, arena);
      var spy = new ExactProvider();
      var baseline = BertForwardPass.fromGgufFile(parsed, config);
      var candidate = BertForwardPass.fromGgufFile(parsed, config, spy);
      assertThat(candidate.encode(new int[] {1, 2, 3}))
          .containsExactly(baseline.encode(new int[] {1, 2, 3}));
      assertThat(spy.calls).isEqualTo(6);
      spy.eligible = false;
      assertThat(candidate.encode(new int[] {3, 2}))
          .containsExactly(baseline.encode(new int[] {3, 2}));
      assertThat(spy.calls).isEqualTo(6);
    }
  }

  private static void addFloat(
      SyntheticGgufBuilder b, String name, int cols, int rows, boolean ones) {
    ByteBuffer data = ByteBuffer.allocate(cols * rows * 4).order(ByteOrder.LITTLE_ENDIAN);
    for (int n = 0; n < cols * rows; n++)
      data.putFloat(ones ? 1f : (float) Math.sin(n * 0.17 + name.length()) * .03f);
    b.addTensor(
        name,
        GgufTensorType.F32,
        rows == 1 ? new long[] {cols} : new long[] {cols, rows},
        data.array());
  }

  private static final class ExactProvider implements GgufBatchedMatrixKernel {
    int calls;
    boolean eligible = true;

    public boolean supports(GgufTensorType type) {
      return type == GgufTensorType.Q5_1;
    }

    public boolean isEligible(GgufTensorType type, int batch, int rows, int cols) {
      return eligible && supports(type);
    }

    public void multiply(
        float[] out,
        float[] input,
        MemorySegment weights,
        GgufTensorType type,
        int batch,
        int rows,
        int cols) {
      calls++;
      TensorOps.ggufExactBatchedMatmul(
          out, input, weights, type, batch, rows, cols, new float[cols]);
    }

    public void close() {}
  }
}
