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
package com.integrallis.models.backend.purejava.gemma4;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufHeader;
import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import com.integrallis.models.backend.purejava.gguf.GgufTensorInfo;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import java.io.ByteArrayOutputStream;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class Gemma4WeightsTest {

  @Test
  void loadsResidentWeightsAndLeavesRoutedExpertsInTheSourceLayout() {
    GgufFile file = fixture(true);

    Gemma4Weights weights = Gemma4Weights.fromGgufFile(file, config());

    float[] embedding = new float[4];
    weights.embedToken(1, embedding);
    assertThat(embedding).containsExactly(5.0f, 6.0f, 7.0f, 8.0f);
    assertThat(weights.tokenEmbeddingType()).isEqualTo(GgufTensorType.F32);
    assertThat(weights.outputNorm()).containsExactly(1.1f, 1.2f, 1.3f, 1.4f);
    assertThat(weights.ropeFrequencyFactors()).containsExactly(1.0f);

    Gemma4Weights.LayerWeights sliding = weights.layer(0);
    assertThat(sliding.queryProjection().rows()).isEqualTo(4);
    assertThat(sliding.queryProjection().columns()).isEqualTo(4);
    assertThat(sliding.valueProjection()).isNotNull();
    assertThat(sliding.routerProjection().type()).isEqualTo(GgufTensorType.F32);
    assertThat(sliding.expertScales()).containsExactly(0.5f, 1.5f);
    assertThat(sliding.layerOutputScale()).isEqualTo(0.75f);

    Gemma4Weights.LayerWeights full = weights.layer(1);
    assertThat(full.valueProjection()).isNull();
    assertThat(full.keyProjection().rows()).isEqualTo(2);
    assertThat(full.queryNorm()).containsExactly(1.0f, 1.25f);
    assertThat(weights.expertLayout().layer(1).expert(1).gateUp().type())
        .isEqualTo(GgufTensorType.Q4_K);
  }

  @Test
  void requiresTheExactFrequencyFactorsUsedByFullAttention() {
    GgufFile file = fixture(false);

    assertThatThrownBy(() -> Gemma4Weights.fromGgufFile(file, config()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("rope_freqs.weight");
  }

  @Test
  void rejectsMatrixTypesThatTheExecutionKernelCannotRun() {
    GgufFile source = fixture(true);
    List<GgufTensorInfo> tensors = new ArrayList<>(source.tensorInfos());
    int queryIndex =
        tensors.indexOf(
            tensors.stream()
                .filter(tensor -> tensor.name().equals("blk.0.attn_q.weight"))
                .findFirst()
                .orElseThrow());
    GgufTensorInfo query = tensors.get(queryIndex);
    tensors.set(
        queryIndex,
        new GgufTensorInfo(
            query.name(), query.nDimensions(), query.shape(), GgufTensorType.F16, query.offset()));
    GgufFile unsupported =
        new GgufFile(
            source.header(),
            source.metadata(),
            tensors,
            source.tensorDataOffset(),
            source.fileSegment());

    assertThatThrownBy(() -> Gemma4Weights.fromGgufFile(unsupported, config()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("blk.0.attn_q.weight")
        .hasMessageContaining("F16");
  }

  private static Gemma4Config config() {
    return new Gemma4Config(
        4,
        2,
        2,
        List.of(1, 1),
        2,
        2,
        2,
        2,
        8,
        16,
        4,
        32,
        2,
        1,
        1_000_000.0f,
        10_000.0f,
        2,
        2,
        1.0e-6f,
        4,
        List.of(true, false),
        30.0f);
  }

  private static GgufFile fixture(boolean includeRopeFactors) {
    FixtureBuilder fixture = new FixtureBuilder();
    fixture.f32("token_embd.weight", new long[] {4, 8}, sequence(1, 32));
    fixture.f32("output_norm.weight", new long[] {4}, 1.1f, 1.2f, 1.3f, 1.4f);
    if (includeRopeFactors) {
      fixture.f32("rope_freqs.weight", new long[] {1}, 1.0f);
    }
    addLayer(fixture, 0, true);
    addLayer(fixture, 1, false);
    return fixture.build();
  }

  private static void addLayer(FixtureBuilder fixture, int layer, boolean sliding) {
    String prefix = "blk." + layer + ".";
    fixture.f32(prefix + "attn_norm.weight", new long[] {4}, 1, 1, 1, 1);
    fixture.f32(prefix + "attn_q.weight", new long[] {4, 4}, sequence(1, 16));
    fixture.f32(prefix + "attn_k.weight", new long[] {4, 2}, sequence(1, 8));
    if (sliding) {
      fixture.f32(prefix + "attn_v.weight", new long[] {4, 2}, sequence(9, 8));
    }
    fixture.f32(prefix + "attn_output.weight", new long[] {4, 4}, sequence(1, 16));
    fixture.f32(prefix + "attn_q_norm.weight", new long[] {2}, 1.0f, 1.25f);
    fixture.f32(prefix + "attn_k_norm.weight", new long[] {2}, 0.75f, 1.0f);
    fixture.f32(prefix + "post_attention_norm.weight", new long[] {4}, 1, 1, 1, 1);

    fixture.f32(prefix + "ffn_norm.weight", new long[] {4}, 1, 1, 1, 1);
    fixture.f32(prefix + "ffn_gate.weight", new long[] {4, 4}, sequence(1, 16));
    fixture.f32(prefix + "ffn_up.weight", new long[] {4, 4}, sequence(17, 16));
    fixture.f32(prefix + "ffn_down.weight", new long[] {4, 4}, sequence(33, 16));
    fixture.f32(prefix + "pre_ffw_norm_2.weight", new long[] {4}, 1, 1, 1, 1);
    fixture.f32(prefix + "post_ffw_norm_1.weight", new long[] {4}, 1, 1, 1, 1);
    fixture.f32(prefix + "post_ffw_norm_2.weight", new long[] {4}, 1, 1, 1, 1);
    fixture.f32(prefix + "post_ffw_norm.weight", new long[] {4}, 1, 1, 1, 1);

    fixture.f32(prefix + "ffn_gate_inp.scale", new long[] {4}, 1, 2, 3, 4);
    fixture.f32(prefix + "ffn_gate_inp.weight", new long[] {4, 2}, sequence(1, 8));
    fixture.f32(prefix + "ffn_down_exps.scale", new long[] {2}, 0.5f, 1.5f);
    fixture.f32(prefix + "layer_output_scale.weight", new long[] {1}, 0.75f);
    fixture.raw(prefix + "ffn_gate_up_exps.weight", new long[] {4, 64, 2}, GgufTensorType.Q4_K);
    fixture.raw(prefix + "ffn_down_exps.weight", new long[] {32, 4, 2}, GgufTensorType.Q8_0);
  }

  private static float[] sequence(int first, int count) {
    float[] values = new float[count];
    for (int index = 0; index < count; index++) {
      values[index] = first + index;
    }
    return values;
  }

  private static final class FixtureBuilder {
    private final List<GgufTensorInfo> tensors = new ArrayList<>();
    private final ByteArrayOutputStream data = new ByteArrayOutputStream();

    private void f32(String name, long[] shape, float... values) {
      long expected = 1;
      for (long dimension : shape) {
        expected = Math.multiplyExact(expected, dimension);
      }
      if (values.length != expected) {
        throw new IllegalArgumentException(name + " requires " + expected + " values");
      }
      ByteBuffer bytes =
          ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      for (float value : values) {
        bytes.putFloat(value);
      }
      add(name, shape, GgufTensorType.F32, bytes.array());
    }

    private void raw(String name, long[] shape, GgufTensorType type) {
      GgufTensorInfo info = new GgufTensorInfo(name, shape.length, shape, type, data.size());
      add(name, shape, type, new byte[Math.toIntExact(info.byteSize())]);
    }

    private void add(String name, long[] shape, GgufTensorType type, byte[] bytes) {
      GgufTensorInfo info = new GgufTensorInfo(name, shape.length, shape, type, data.size());
      if (bytes.length != info.byteSize()) {
        throw new IllegalArgumentException(name + " has the wrong byte count");
      }
      tensors.add(info);
      data.writeBytes(bytes);
    }

    private GgufFile build() {
      return new GgufFile(
          new GgufHeader(3, tensors.size(), 0),
          new GgufMetadata(Map.of()),
          tensors,
          0,
          MemorySegment.ofArray(data.toByteArray()));
    }
  }
}
