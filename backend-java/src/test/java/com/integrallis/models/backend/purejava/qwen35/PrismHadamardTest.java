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
package com.integrallis.models.backend.purejava.qwen35;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.data.Offset.offset;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufHeader;
import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import com.integrallis.models.backend.purejava.gguf.GgufMetadataValue;
import com.integrallis.models.backend.purejava.gguf.GgufTensorInfo;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.GgufValueType;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class PrismHadamardTest {
  @Test
  void metadataRequiresEveryDeclaredProjectionToUseTheTransform() {
    var hadamard = PrismHadamard.load(file(metadata()), config());
    assertThat(hadamard.projection("output.weight")).isNotNull();
    assertThatThrownBy(() -> hadamard.requireRouted(Set.of()))
        .isInstanceOf(IllegalArgumentException.class);
    hadamard.requireRouted(Set.of("output.weight"));
    float[] embedding = {1, 2, 3, 4};
    hadamard.restoreEmbedding(embedding);
    assertThat(embedding).isNotEqualTo(new float[] {1, 2, 3, 4});
  }

  @Test
  void rejectsUnsupportedMetadataInsteadOfSilentlyIgnoringTheRotation() {
    for (String field :
        List.of("version", "block_size", "sign_values", "weight_names", "inverse_weight_names")) {
      var metadata = metadata();
      metadata.put(
          "prism.hadamard." + field,
          switch (field) {
            case "version" -> new GgufMetadataValue.Uint32Value(2);
            case "block_size" -> new GgufMetadataValue.Uint32Value(3);
            case "sign_values" -> integers(1, 1, 1, 0);
            case "weight_names" -> strings("output.weight", "output.weight");
            default -> strings("output.weight");
          });
      assertThatThrownBy(() -> PrismHadamard.load(file(metadata), config()))
          .isInstanceOf(IllegalArgumentException.class);
    }
  }

  private static Qwen35Config config() {
    return new Qwen35Config(
        "qwen35", 4, 2, 2, 1, 2, 8, 32, 8, 10000, 2, 1e-6f, 4, 2, 2, 4, 8, 2, 0, 0, 0, 0);
  }

  private static Map<String, GgufMetadataValue> metadata() {
    var metadata = new HashMap<String, GgufMetadataValue>();
    metadata.put("prism.hadamard.version", new GgufMetadataValue.Uint32Value(1));
    metadata.put("prism.hadamard.block_size", new GgufMetadataValue.Uint32Value(4));
    metadata.put(
        "prism.hadamard.transform",
        new GgufMetadataValue.StringValue("normalized-sylvester-walsh-hadamard"));
    metadata.put("prism.hadamard.axis", new GgufMetadataValue.StringValue("input-last-dimension"));
    metadata.put("prism.hadamard.sign_mode", new GgufMetadataValue.StringValue("explicit"));
    metadata.put("prism.hadamard.sign_widths", integers(4));
    metadata.put("prism.hadamard.sign_values", integers(1, -1, 1, -1));
    metadata.put("prism.hadamard.weight_names", strings("output.weight"));
    metadata.put("prism.hadamard.inverse_weight_names", strings("token_embd.weight"));
    return metadata;
  }

  private static GgufMetadataValue integers(Integer... values) {
    return new GgufMetadataValue.ArrayValue(
        GgufValueType.INT32,
        Arrays.stream(values)
            .map(x -> (GgufMetadataValue) new GgufMetadataValue.Int32Value(x))
            .toList());
  }

  private static GgufMetadataValue strings(String... values) {
    return new GgufMetadataValue.ArrayValue(
        GgufValueType.STRING,
        Arrays.stream(values)
            .map(x -> (GgufMetadataValue) new GgufMetadataValue.StringValue(x))
            .toList());
  }

  private static GgufFile file(Map<String, GgufMetadataValue> metadata) {
    return new GgufFile(
        new GgufHeader(3, 2, metadata.size()),
        new GgufMetadata(metadata),
        List.of(
            new GgufTensorInfo("output.weight", 2, new long[] {4, 8}, GgufTensorType.F32, 0),
            new GgufTensorInfo("token_embd.weight", 2, new long[] {4, 8}, GgufTensorType.F32, 128)),
        0,
        MemorySegment.ofArray(new float[64]));
  }

  @Test
  void forwardMatchesDenseSylvesterMatrixAndInverseRestoresSignedInputs() {
    float[] input = {1, -2, 3, 4, 5, -6, 7, 8};
    int[] signs = {-1, 1, -1, 1, 1, -1, -1, 1};
    var transform = new PrismHadamard.Transform(4, 8, signs, 1, 1);
    float[] output = new float[8];
    transform.forward(input, output);
    for (int i = 0; i < 8; i++) {
      double expected = 0;
      for (int j = 0; j < 4; j++) {
        int index = (i / 4) * 4 + j;
        expected +=
            input[index] * signs[index] * ((Integer.bitCount((i % 4) & j) & 1) == 0 ? .5 : -.5);
      }
      assertThat(output[i]).isCloseTo((float) expected, offset(1e-6f));
    }
    transform.inverse(output);
    assertThat(output).containsExactly(input);
  }

  @Test
  void recurrentProjectionGroupsTiledValueHeadsBeforeRotating() {
    // Two key heads, three values per key. Undo just the rotation to expose the permutation.
    float[] input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float[] output = new float[12];
    var grouped = new PrismHadamard.Transform(4, 12, null, 2, 6);
    grouped.forward(input, output);
    new PrismHadamard.Transform(4, 12, null, 1, 1).inverse(output);
    assertThat(output).containsExactly(1, 2, 5, 6, 9, 10, 3, 4, 7, 8, 11, 12);
  }

  @Test
  void rejectsInvalidGeometryAndSigns() {
    assertThatThrownBy(() -> new PrismHadamard.Transform(3, 12, null, 1, 1))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new PrismHadamard.Transform(4, 10, null, 1, 1))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new PrismHadamard.Transform(4, 12, null, 4, 6))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new PrismHadamard.Transform(4, 4, new int[] {1, 1, 0, 1}, 1, 1))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
