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
package com.integrallis.models.backend.purejava.cact;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class CactNeedle2LayoutTest {

  @Test
  void namesAndValidatesTheCanonicalTensorOrder() {
    CactNeedle2Layout layout = CactNeedle2Layout.from(layoutWithoutHeads());

    assertThat(layout.tensorNames()).hasSize(30);
    assertThat(layout.tensorNames().getFirst()).isEqualTo("embedding");
    assertThat(layout.tensorNames().getLast()).isEqualTo("tokenizer");
    assertThat(layout.tensor("layer00.q_proj").info().shape()).containsExactly(8, 8);
    assertThat(layout.tensor("layer00.k_proj").info().shape()).containsExactly(4, 8);
    assertThat(layout.tensor("mhc_phi_res").info().shape()).containsExactly(1, 8);
    assertThat(layout.tensor("engram0.tables").info().shape()).containsExactly(16, 4);
    assertThat(layout.tensor("engram0.taps").info().shape()).containsExactly(4, 8);
  }

  @Test
  void rejectsATypeOrShapeThatDoesNotMatchItsPosition() {
    CactFile wrongEmbedding = layoutWithoutHeads(false);

    assertThatThrownBy(() -> CactNeedle2Layout.from(wrongEmbedding))
        .isInstanceOf(MalformedCactException.class)
        .hasMessageContaining("embedding");
  }

  @Test
  void readsProbeHeadsFromTheSerializedManifest() {
    CactNeedle2Layout layout = CactNeedle2Layout.from(layoutWithHeads(1, 2));

    assertThat(layout.tensorNames())
        .containsSubsequence(
            "heads.manifest",
            "contrastive_head.probes",
            "contrastive_head.proj",
            "contrastive_head.bias",
            "confidence_head.probes",
            "confidence_head.proj",
            "confidence_head.bias",
            "tokenizer");
  }

  @Test
  void rejectsProbeHeadsOutsideCanonicalManifestOrder() {
    assertThatThrownBy(() -> CactNeedle2Layout.from(layoutWithHeads(2, 1)))
        .isInstanceOf(MalformedCactException.class)
        .hasMessageContaining("canonical codes");
  }

  @Test
  void matchesPinnedOfficialNeedle2TensorLayoutWhenProvided() throws IOException {
    String configured = System.getProperty("models.fixtures.needle2Cact", "");
    assumeTrue(!configured.isBlank(), "set -Dmodels.fixtures.needle2Cact=<needle2.cact>");
    Path path = Path.of(configured);
    assumeTrue(Files.isRegularFile(path), "Needle 2 fixture is not installed");

    try (Arena arena = Arena.ofConfined()) {
      CactNeedle2Layout layout = CactNeedle2Layout.from(CactParser.parse(path, arena));

      assertThat(layout.tensorNames()).hasSize(405);
      assertThat(layout.tensorNames().getFirst()).isEqualTo("embedding");
      assertThat(layout.tensorNames().getLast()).isEqualTo("tokenizer");
      assertThat(layout.tensor("layer26.out_proj").info().shape()).containsExactly(512, 512);
      assertThat(layout.tensor("mhc_phi_res").info().shape()).containsExactly(432, 2048);
      assertThat(layout.tensor("engram1.tables").info().shape()).containsExactly(32_768, 128);
      assertThat(layout.tensor("heads.manifest").info().shape()).containsExactly(2);
      assertThat(layout.tensor("contrastive_head.probes").info().shape()).containsExactly(4, 512);
      assertThat(layout.tensor("contrastive_head.proj").info().shape()).containsExactly(128, 2048);
      assertThat(layout.tensor("confidence_head.probes").info().shape()).containsExactly(8, 512);
      assertThat(layout.tensor("confidence_head.proj").info().shape()).containsExactly(1, 4096);
    }
  }

  private static CactFile layoutWithoutHeads() {
    return layoutWithoutHeads(true);
  }

  private static CactFile layoutWithoutHeads(boolean quantizedEmbedding) {
    SyntheticCactBuilder builder = baseLayoutBuilder(quantizedEmbedding);
    builder.addRaw((byte) 1);
    return CactParser.parseSegment(MemorySegment.ofArray(builder.build()));
  }

  private static CactFile layoutWithHeads(int... codes) {
    SyntheticCactBuilder builder = baseLayoutBuilder(true).addFp16Values(toFloats(codes));
    for (int code : codes) {
      if (code == 1) {
        builder.addFp16(4, 8).addFp16(3, 32).addFp16(3);
      } else {
        builder.addFp16(8, 8).addFp16(1, 64).addFp16(1);
      }
    }
    builder.addRaw((byte) 1);
    return CactParser.parseSegment(MemorySegment.ofArray(builder.build()));
  }

  private static SyntheticCactBuilder baseLayoutBuilder(boolean quantizedEmbedding) {
    SyntheticCactBuilder builder = new SyntheticCactBuilder();
    if (quantizedEmbedding) {
      builder.addCq(32, 8, 8, 2);
    } else {
      builder.addFp16(32, 8);
    }
    builder.addFp16(8);
    builder.addCq(8, 8, 8, 2);
    builder.addCq(4, 8, 8, 2);
    builder.addCq(4, 8, 8, 2);
    builder.addFp16(4);
    builder.addFp16(4);
    builder.addCq(8, 8, 8, 2);
    builder.addCq(8, 8, 8, 2);
    builder.addFp16(8);
    builder.addFp16(1);
    builder.addFp16(8);
    builder.addFp16(8);
    builder.addFp16(8);
    builder.addFp16(8);
    builder.addFp16(1);
    builder.addFp16(1);
    builder.addFp16(1);
    builder.addFp16(1, 1);
    builder.addFp16(1, 1);
    builder.addFp16(1, 1, 1);
    builder.addCq(1, 8, 8, 2);
    builder.addCq(1, 8, 8, 2);
    builder.addCq(1, 8, 8, 2);
    builder.addCq(16, 4, 8, 2);
    builder.addCq(8, 8, 8, 2);
    builder.addCq(8, 8, 8, 2);
    builder.addFp16(4, 8);
    builder.addFp16(8);
    return builder;
  }

  private static float[] toFloats(int[] values) {
    float[] result = new float[values.length];
    for (int index = 0; index < values.length; index++) {
      result[index] = values[index];
    }
    return result;
  }
}
