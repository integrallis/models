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
package com.integrallis.models.backend.purejava.safetensors;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class SafetensorsParserTest {

  @Test
  void parsesMetadataDenseTypesScalarsAndMappedTensorViews(@TempDir Path directory)
      throws IOException {
    byte[] artifact =
        new SyntheticSafetensorsBuilder()
            .metadata("format", "pt")
            .add("model.embed_tokens.weight", "BF16", new long[] {2, 2}, 1, 2, 3, 4, 5, 6, 7, 8)
            .add("scale", "F32", new long[0], 0, 0, (byte) 0x80, 0x3f)
            .build();
    Path path = directory.resolve("model.safetensors");
    Files.write(path, artifact);

    try (Arena arena = Arena.ofConfined()) {
      SafetensorsFile file = SafetensorsParser.parse(path, arena);

      assertThat(file.metadata()).containsEntry("format", "pt");
      assertThat(file.tensorNames()).containsExactly("model.embed_tokens.weight", "scale");
      SafetensorsTensor embedding = file.tensor("model.embed_tokens.weight");
      assertThat(embedding.info().dtype()).isEqualTo(SafetensorsDtype.BF16);
      assertThat(embedding.info().shape()).containsExactly(2, 2);
      assertThat(embedding.data().toArray(ValueLayout.JAVA_BYTE))
          .containsExactly(1, 2, 3, 4, 5, 6, 7, 8);
      assertThat(file.tensor("scale").info().elementCount()).isEqualTo(1);
    }
  }

  @Test
  void acceptsEmptyTensorWithoutAllocatingData() {
    byte[] artifact =
        new SyntheticSafetensorsBuilder().add("empty", "F32", new long[] {4, 0, 3}).build();

    SafetensorsFile file = SafetensorsParser.parseSegment(MemorySegment.ofArray(artifact));

    assertThat(file.tensor("empty").info().elementCount()).isZero();
    assertThat(file.tensor("empty").data().byteSize()).isZero();
    assertThat(file.tensor("empty").data().isReadOnly()).isTrue();
  }

  @Test
  void recognizesEveryDtypeInTheCurrentOfficialFormat() {
    assertThat(SafetensorsDtype.values())
        .extracting(SafetensorsDtype::code)
        .containsExactlyInAnyOrder(
            "BOOL",
            "F4",
            "F6_E2M3",
            "F6_E3M2",
            "U8",
            "I8",
            "F8_E5M2",
            "F8_E4M3",
            "F8_E8M0",
            "F8_E4M3FNUZ",
            "F8_E5M2FNUZ",
            "I16",
            "U16",
            "F16",
            "BF16",
            "I32",
            "U32",
            "F32",
            "C64",
            "F64",
            "I64",
            "U64");
  }

  @Test
  void parsesAnArtifactWrittenByTheOfficialPythonImplementationWhenProvided() throws IOException {
    String configured = System.getProperty("models.fixtures.safetensorsReference", "");
    assumeTrue(!configured.isBlank(), "set -Dmodels.fixtures.safetensorsReference=<file>");
    Path path = Path.of(configured);
    assumeTrue(Files.isRegularFile(path), "official Safetensors fixture is not installed");

    try (Arena arena = Arena.ofConfined()) {
      SafetensorsFile file = SafetensorsParser.parse(path, arena);

      assertThat(file.metadata())
          .containsEntry("format", "numpy")
          .containsEntry("oracle", "huggingface-safetensors");
      assertThat(file.tensor(GLOBAL).info().shape()).containsExactly(1);
      assertThat(
              file.tensor(GLOBAL)
                  .data()
                  .get(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), 0))
          .isEqualTo(4.0f);
      assertThat(file.tensor(PACKED).info().shape()).containsExactly(2, 2);
      assertThat(file.tensor(PACKED).data().toArray(ValueLayout.JAVA_BYTE))
          .containsExactly(0x01, 0x23, 0x45, 0x67);
    }
  }

  @Test
  void rejectsMalformedHeadersTensorLayoutsAndIncompleteBuffers() {
    byte[] badPrefix = SyntheticSafetensorsBuilder.file("[]", new byte[0]);
    byte[] duplicate =
        SyntheticSafetensorsBuilder.file(
            "{\"dup\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},"
                + "\"dup\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]}}",
            new byte[] {1, 2});
    byte[] hole =
        SyntheticSafetensorsBuilder.file(
            "{\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]}}", new byte[] {0, 1});
    byte[] wrongSize =
        SyntheticSafetensorsBuilder.file(
            "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}",
            new byte[] {0, 0, 0, 0});
    byte[] trailingData =
        SyntheticSafetensorsBuilder.file(
            "{\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}", new byte[] {1, 2});
    byte[] oversizedHeader =
        ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(100_000_001).array();
    byte[] fractionalShape =
        SyntheticSafetensorsBuilder.file(
            "{\"a\":{\"dtype\":\"U8\",\"shape\":[1.5],\"data_offsets\":[0,1]}}", new byte[] {1});
    byte[] widerThanOfficialSizeT =
        SyntheticSafetensorsBuilder.file(
            "{\"a\":{\"dtype\":\"U8\",\"shape\":[281474976710656]," + "\"data_offsets\":[0,0]}}",
            new byte[0]);

    assertMalformed(badPrefix, "begin");
    assertMalformed(duplicate, "duplicate");
    assertMalformed(hole, "offset");
    assertMalformed(wrongSize, "byte count");
    assertMalformed(trailingData, "entire data buffer");
    assertMalformed(oversizedHeader, "100000000");
    assertMalformed(fractionalShape, "integers");
    assertMalformed(widerThanOfficialSizeT, "48-bit");
  }

  private static final String BASE = "model.language_model.layers.49.mlp.down_proj";
  private static final String PACKED = BASE + ".weight_packed";
  private static final String GLOBAL = BASE + ".weight_global_scale";

  private static void assertMalformed(byte[] artifact, String message) {
    assertThatThrownBy(() -> SafetensorsParser.parseSegment(MemorySegment.ofArray(artifact)))
        .isInstanceOf(MalformedSafetensorsException.class)
        .hasMessageContaining(message);
  }
}
