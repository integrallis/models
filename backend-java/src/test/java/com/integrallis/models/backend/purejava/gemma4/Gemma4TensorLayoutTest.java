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

import com.integrallis.models.backend.purejava.gguf.GgufTensorInfo;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class Gemma4TensorLayoutTest {

  private static final long DATA_OFFSET = 15_821_408L;
  private static final long GATE_UP_OFFSET = 907_262_464L;
  private static final long DOWN_OFFSET = 632_848_896L;

  @Test
  void resolvesTwoContiguousSourceRangesForEachExpert() {
    Gemma4TensorLayout layout =
        Gemma4TensorLayout.fromTensorInfos(
            DATA_OFFSET,
            config(),
            List.of(
                tensor(
                    "token_embd.weight", new long[] {2_816, 262_144}, GgufTensorType.Q6_K, 12_288),
                tensor(
                    "blk.0.ffn_gate_up_exps.weight",
                    new long[] {2_816, 1_408, 128},
                    GgufTensorType.Q4_K,
                    GATE_UP_OFFSET),
                tensor(
                    "blk.0.ffn_down_exps.weight",
                    new long[] {704, 2_816, 128},
                    GgufTensorType.Q8_0,
                    DOWN_OFFSET),
                tensor(
                    "blk.0.ffn_down_exps.scale",
                    new long[] {128},
                    GgufTensorType.F32,
                    632_848_384L)));

    Gemma4TensorLayout.ExpertWeights expert = layout.layer(0).expert(7);

    assertThat(expert.gateUp().type()).isEqualTo(GgufTensorType.Q4_K);
    assertThat(expert.gateUp().shape()).containsExactly(2_816, 1_408);
    assertThat(expert.gateUp().byteSize()).isEqualTo(2_230_272L);
    assertThat(expert.gateUp().fileOffset())
        .isEqualTo(DATA_OFFSET + GATE_UP_OFFSET + 7L * 2_230_272L);

    assertThat(expert.down().type()).isEqualTo(GgufTensorType.Q8_0);
    assertThat(expert.down().shape()).containsExactly(704, 2_816);
    assertThat(expert.down().byteSize()).isEqualTo(2_106_368L);
    assertThat(expert.down().fileOffset()).isEqualTo(DATA_OFFSET + DOWN_OFFSET + 7L * 2_106_368L);
    assertThat(expert.totalBytes()).isEqualTo(4_336_640L);

    assertThat(layout.routedExpertBytes()).isEqualTo(555_089_920L);
    assertThat(layout.residentBytes()).isEqualTo(605_553_152L);
    assertThat(layout.residentTensors())
        .extracting(GgufTensorInfo::name)
        .containsExactly("token_embd.weight", "blk.0.ffn_down_exps.scale");
  }

  @Test
  void rejectsAnExpertTensorWhoseExpertAxisDoesNotMatchTheConfiguration() {
    List<GgufTensorInfo> tensors =
        List.of(
            tensor(
                "blk.0.ffn_gate_up_exps.weight",
                new long[] {2_816, 1_408, 64},
                GgufTensorType.Q4_K,
                GATE_UP_OFFSET),
            tensor(
                "blk.0.ffn_down_exps.weight",
                new long[] {704, 2_816, 128},
                GgufTensorType.Q8_0,
                DOWN_OFFSET),
            tensor(
                "blk.0.ffn_down_exps.scale", new long[] {128}, GgufTensorType.F32, 632_848_384L));

    assertThatThrownBy(() -> Gemma4TensorLayout.fromTensorInfos(DATA_OFFSET, config(), tensors))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("blk.0.ffn_gate_up_exps.weight")
        .hasMessageContaining("[2816, 1408, 128]");
  }

  @Test
  void rejectsUnsupportedRoutedExpertTensorTypes() {
    List<GgufTensorInfo> tensors =
        List.of(
            tensor(
                "blk.0.ffn_gate_up_exps.weight",
                new long[] {2_816, 1_408, 128},
                GgufTensorType.Q6_K,
                GATE_UP_OFFSET),
            tensor(
                "blk.0.ffn_down_exps.weight",
                new long[] {704, 2_816, 128},
                GgufTensorType.Q8_0,
                DOWN_OFFSET),
            tensor(
                "blk.0.ffn_down_exps.scale", new long[] {128}, GgufTensorType.F32, 632_848_384L));

    assertThatThrownBy(() -> Gemma4TensorLayout.fromTensorInfos(DATA_OFFSET, config(), tensors))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("blk.0.ffn_gate_up_exps.weight")
        .hasMessageContaining("Q4_K")
        .hasMessageContaining("Q6_K");
  }

  private static Gemma4Config config() {
    return new Gemma4Config(
        2_816,
        1,
        16,
        List.of(2),
        512,
        256,
        512,
        256,
        262_144,
        262_144,
        2_112,
        704,
        128,
        8,
        1_000_000.0f,
        10_000.0f,
        512,
        256,
        1.0e-6f,
        1_024,
        List.of(false),
        30.0f);
  }

  private static GgufTensorInfo tensor(
      String name, long[] shape, GgufTensorType type, long offset) {
    return new GgufTensorInfo(name, shape.length, shape, type, offset);
  }
}
