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
package com.integrallis.models.backend.purejava.gptoss;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.safetensors.SafetensorsBundle;
import com.integrallis.models.backend.purejava.safetensors.SyntheticSafetensorsBuilder;
import com.integrallis.models.backend.purejava.tensor.SafetensorsTensorSource;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class GptOssMxfp4ExpertWeightsTest {

  private static final int EXPERTS = 2;
  private static final int HIDDEN = 64;
  private static final int INTERMEDIATE = 32;
  private static final String PREFIX = "model.layers.0.mlp.experts.";
  private static final int[] OFFICIAL_FIRST_BLOCK = {
    0x00, 0xc0, 0x80, 0xa9, 0x10, 0x1b, 0x81, 0x22,
    0x93, 0xe4, 0xa0, 0xb2, 0x3b, 0x19, 0xb2, 0x31
  };
  private static final float[] OFFICIAL_FIRST_VALUES = {
    0.0f,
    0.0f,
    0.0f,
    -0.0625f,
    0.0f,
    -0.0f,
    -0.015625f,
    -0.03125f,
    0.0f,
    0.015625f,
    -0.046875f,
    0.015625f,
    0.015625f,
    -0.0f,
    0.03125f,
    0.03125f,
    0.046875f,
    -0.015625f,
    0.0625f,
    -0.125f,
    0.0f,
    -0.03125f,
    0.03125f,
    -0.046875f,
    -0.046875f,
    0.046875f,
    -0.015625f,
    0.015625f,
    0.03125f,
    -0.046875f,
    0.015625f,
    0.046875f
  };

  @Test
  void mapsOfficialExpertTensorGeometryWithoutTransposingOrExpanding(@TempDir Path directory)
      throws IOException {
    SafetensorsTensorSource source = source(directory, false);

    GptOssMxfp4ExpertWeights weights =
        GptOssMxfp4ExpertWeights.load(source, 0, EXPERTS, HIDDEN, INTERMEDIATE);

    assertThat(weights.expertCount()).isEqualTo(EXPERTS);
    GptOssMxfp4ExpertWeights.Expert first = weights.expert(0);
    GptOssMxfp4ExpertWeights.Expert second = weights.expert(1);
    assertThat(first.gateUp().rows()).isEqualTo(2 * INTERMEDIATE);
    assertThat(first.gateUp().columns()).isEqualTo(HIDDEN);
    assertThat(first.down().rows()).isEqualTo(HIDDEN);
    assertThat(first.down().columns()).isEqualTo(INTERMEDIATE);
    // openai/gpt-oss-20b at 6cee5e81ee83917806bbde320786a8fb61efebee,
    // layer 0, expert 0, gate/up row 0,
    // bytes 703133288..703133303 and E8M0 scale byte 968554088 in shard 0.
    for (int column = 0; column < OFFICIAL_FIRST_VALUES.length; column++) {
      assertThat(first.gateUp().value(0, column)).isEqualTo(OFFICIAL_FIRST_VALUES[column]);
    }
    assertThat(second.gateUp().value(0, 0)).isEqualTo(1.0f);
    assertThat(first.down().value(0, 0)).isEqualTo(1.5f);
    assertThat(second.down().value(0, 0)).isEqualTo(2.0f);
    assertThat(first.gateUpBias()).startsWith(1.0f, 2.0f).hasSize(2 * INTERMEDIATE);
    assertThat(second.gateUpBias()).startsWith(65.0f, 66.0f).hasSize(2 * INTERMEDIATE);
    assertThat(first.downBias()).startsWith(129.0f, 130.0f).hasSize(HIDDEN);
    assertThat(second.downBias()).startsWith(193.0f, 194.0f).hasSize(HIDDEN);
  }

  @Test
  void rejectsCheckpointGeometryThatDoesNotMatchTheConfiguration(@TempDir Path directory)
      throws IOException {
    SafetensorsTensorSource source = source(directory, true);

    assertThatThrownBy(
            () -> GptOssMxfp4ExpertWeights.load(source, 0, EXPERTS, HIDDEN, INTERMEDIATE))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining(PREFIX + "gate_up_proj_blocks")
        .hasMessageContaining("[2, 64, 2, 16]");
  }

  private static SafetensorsTensorSource source(Path directory, boolean invalidGateUpShape)
      throws IOException {
    int gateUpBlockBytes = EXPERTS * 2 * INTERMEDIATE * HIDDEN / 2;
    int gateUpScaleBytes = EXPERTS * 2 * INTERMEDIATE * HIDDEN / 32;
    int downBlockBytes = EXPERTS * HIDDEN * INTERMEDIATE / 2;
    int downScaleBytes = EXPERTS * HIDDEN * INTERMEDIATE / 32;
    int[] gateUpBlocks = new int[gateUpBlockBytes];
    int[] downBlocks = new int[downBlockBytes];
    java.util.Arrays.fill(gateUpBlocks, 0, gateUpBlockBytes / 2, 0x11);
    java.util.Arrays.fill(gateUpBlocks, gateUpBlockBytes / 2, gateUpBlockBytes, 0x22);
    System.arraycopy(OFFICIAL_FIRST_BLOCK, 0, gateUpBlocks, 0, OFFICIAL_FIRST_BLOCK.length);
    java.util.Arrays.fill(downBlocks, 0, downBlockBytes / 2, 0x33);
    java.util.Arrays.fill(downBlocks, downBlockBytes / 2, downBlockBytes, 0x44);
    int[] gateUpScales = filled(gateUpScaleBytes, 127);
    gateUpScales[0] = 122;
    SyntheticSafetensorsBuilder builder =
        new SyntheticSafetensorsBuilder()
            .add(
                PREFIX + "gate_up_proj_blocks",
                "U8",
                invalidGateUpShape
                    ? new long[] {EXPERTS, 2L * INTERMEDIATE, 16, HIDDEN / 32L}
                    : new long[] {EXPERTS, 2L * INTERMEDIATE, HIDDEN / 32, 16},
                gateUpBlocks)
            .add(
                PREFIX + "gate_up_proj_scales",
                "U8",
                new long[] {EXPERTS, 2L * INTERMEDIATE, HIDDEN / 32},
                gateUpScales)
            .add(
                PREFIX + "gate_up_proj_bias",
                "BF16",
                new long[] {EXPERTS, 2L * INTERMEDIATE},
                bfloat16Sequence(EXPERTS * 2 * INTERMEDIATE, 1))
            .add(
                PREFIX + "down_proj_blocks",
                "U8",
                new long[] {EXPERTS, HIDDEN, INTERMEDIATE / 32, 16},
                downBlocks)
            .add(
                PREFIX + "down_proj_scales",
                "U8",
                new long[] {EXPERTS, HIDDEN, INTERMEDIATE / 32},
                filled(downScaleBytes, 127))
            .add(
                PREFIX + "down_proj_bias",
                "BF16",
                new long[] {EXPERTS, HIDDEN},
                bfloat16Sequence(EXPERTS * HIDDEN, 129));
    Path artifact = Files.write(directory.resolve("model.safetensors"), builder.build());
    return new SafetensorsTensorSource(SafetensorsBundle.open(artifact, Arena.global()));
  }

  private static int[] filled(int count, int value) {
    int[] values = new int[count];
    java.util.Arrays.fill(values, value);
    return values;
  }

  private static int[] bfloat16Sequence(int count, int start) {
    int[] bytes = new int[count * Short.BYTES];
    for (int index = 0; index < count; index++) {
      int bits = Float.floatToRawIntBits(start + index) >>> Short.SIZE;
      bytes[index * Short.BYTES] = bits & 0xff;
      bytes[index * Short.BYTES + 1] = bits >>> Byte.SIZE;
    }
    return bytes;
  }
}
