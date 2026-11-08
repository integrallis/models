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
package com.integrallis.models.backend.purejava;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import com.integrallis.vectors.core.GgufQ4Kernel;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Checks the public loader, kernel eligibility, arithmetic and ownership together. */
@Tag("unit")
class EmbeddingGemma2BackendRouteTest {
  @Test
  void injectedKernelReceivesEligibleProjectionsAndKeepsJavaResults(@TempDir Path directory)
      throws Exception {
    Path model = write(directory);
    CountingKernel kernel = new CountingKernel();
    try (var control = PureJavaBackend.load(model);
        var candidate = PureJavaBackend.load(model, kernel)) {
      int[] tokens = {2, 4, 3};
      float[] expected = control.embedSequence(tokens);
      assertThat(candidate.embedSequence(tokens)).containsExactly(expected);
      // Nine projections in each layer plus the input projection; the F16 output stays in Java.
      assertThat(kernel.calls).isEqualTo(19);
      assertThat(kernel.closed).isFalse();

      // This provider declines one-token shapes. Its supports(type) alone is insufficient.
      assertThat(candidate.embedSequence(new int[] {5}))
          .containsExactly(control.embedSequence(new int[] {5}));
      assertThat(kernel.calls).isEqualTo(19);
      assertThat(candidate.embedSequence(tokens)).containsExactly(expected);
      assertThat(kernel.calls).isEqualTo(38);
    }
    assertThat(kernel.closed).isTrue();
    assertThat(kernel.closeCalls).isEqualTo(1);
  }

  private static final class CountingKernel implements GgufBatchedMatrixKernel {
    int calls;
    int closeCalls;
    boolean closed;

    @Override
    public boolean supports(GgufTensorType type) {
      return type == GgufTensorType.Q8_0;
    }

    @Override
    public boolean isEligible(GgufTensorType type, int count, int rows, int columns) {
      return supports(type) && count > 1;
    }

    @Override
    public void multiply(
        float[] out,
        float[] in,
        MemorySegment weights,
        GgufTensorType type,
        int count,
        int rows,
        int columns) {
      assertThat(closed).isFalse();
      assertThat(isEligible(type, count, rows, columns)).isTrue();
      calls++;
      int elements = count * columns;
      TensorOps.ggufBatchedMatmul(
          out,
          in,
          weights,
          type,
          count,
          rows,
          columns,
          new byte[elements],
          new float[elements / 32],
          new int[elements / 32],
          new short[elements / 16],
          new float[0],
          GgufQ4Kernel.WIDENED);
    }

    @Override
    public void close() {
      closed = true;
      closeCalls++;
    }
  }

  private static Path write(Path directory) throws Exception {
    String p = "gemma-embedding2.";
    var b =
        new SyntheticGgufBuilder()
            .addString("general.architecture", "gemma-embedding2")
            .addUint32(p + "block_count", 2)
            .addUint32(p + "embedding_length", 32)
            .addUint32(p + "embedding_length_out", 16)
            .addUint32(p + "context_length", 16)
            .addUint32(p + "feed_forward_length", 64)
            .addUint32(p + "attention.head_count", 2)
            .addInt32Array(p + "attention.head_count_kv", List.of(1, 1))
            .addUint32(p + "attention.key_length", 32)
            .addUint32(p + "attention.value_length", 32)
            .addUint32(p + "rope.dimension_count", 32)
            .addUint32(p + "attention.key_length_swa", 16)
            .addUint32(p + "attention.value_length_swa", 16)
            .addUint32(p + "rope.dimension_count_swa", 16)
            .addUint32(p + "attention.sliding_window", 2)
            .addBoolArray(p + "attention.sliding_window_pattern", List.of(true, false))
            .addUint32(p + "embedding_length_per_layer_input", 32)
            .addFloat32(p + "rope.freq_base", 10000)
            .addFloat32(p + "rope.freq_base_swa", 1000)
            .addFloat32(p + "attention.layer_norm_rms_epsilon", 1e-6f)
            .addBool(p + "attention.causal", false)
            .addUint32(p + "pooling_type", 1)
            .addStringArray("tokenizer.ggml.tokens", List.of("t0", "t1", "t2", "t3", "t4", "t5"))
            .addFloat32Array("tokenizer.ggml.scores", List.of(0f, 0f, 0f, 0f, 0f, 0f))
            .addUint32("tokenizer.ggml.bos_token_id", 0)
            .addUint32("tokenizer.ggml.eos_token_id", 1);
    matrix(b, "token_embd.weight", 6, 32);
    matrix(b, "per_layer_model_proj.weight", 64, 32);
    vector(b, "per_layer_proj_norm.weight", 32);
    vector(b, "output_norm.weight", 32);
    var output = ByteBuffer.allocate(16 * 32 * 2).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < 16 * 32; i++)
      output.putShort(Float.floatToFloat16(((i * 7) % 23 - 11) * .01f));
    b.addTensor("output.weight", GgufTensorType.F16, new long[] {32, 16}, output.array());
    for (int layer = 0; layer < 2; layer++) {
      String n = "blk." + layer + ".";
      int head = layer == 0 ? 16 : 32;
      matrix(b, n + "attn_q.weight", head * 2, 32);
      matrix(b, n + "attn_k.weight", head, 32);
      matrix(b, n + "attn_v.weight", head, 32);
      matrix(b, n + "attn_output.weight", 32, head * 2);
      matrix(b, n + "ffn_gate.weight", 64, 32);
      matrix(b, n + "ffn_up.weight", 64, 32);
      matrix(b, n + "ffn_down.weight", 32, 64);
      matrix(b, n + "inp_gate.weight", 32, 32);
      matrix(b, n + "proj.weight", 32, 32);
      for (String norm :
          List.of("attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm", "post_norm"))
        vector(b, n + norm + ".weight", 32);
      vector(b, n + "attn_q_norm.weight", head);
      vector(b, n + "attn_k_norm.weight", head);
      vector(b, n + "layer_output_scale.weight", 1);
    }
    Path path = directory.resolve("gemma2-route.gguf");
    Files.write(path, b.build());
    return path;
  }

  private static void matrix(SyntheticGgufBuilder b, String name, int rows, int columns) {
    var data = ByteBuffer.allocate(rows * columns / 32 * 34).order(ByteOrder.LITTLE_ENDIAN);
    for (int block = 0; block < rows * columns / 32; block++) {
      data.putShort(Float.floatToFloat16(.0078125f));
      for (int i = 0; i < 32; i++)
        data.put((byte) ((i * 13 + block * 7 + name.length()) % 31 - 15));
    }
    b.addTensor(name, GgufTensorType.Q8_0, new long[] {columns, rows}, data.array());
  }

  private static void vector(SyntheticGgufBuilder b, String name, int size) {
    var data = ByteBuffer.allocate(size * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < size; i++) data.putFloat(1f + (i % 3) * .01f);
    b.addTensor(name, GgufTensorType.F32, new long[] {size}, data.array());
  }
}
