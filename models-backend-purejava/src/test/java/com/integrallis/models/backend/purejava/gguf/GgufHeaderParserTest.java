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
package com.integrallis.models.backend.purejava.gguf;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class GgufHeaderParserTest {

  private MemorySegment toSegment(byte[] data) {
    var segment = Arena.ofConfined().allocate(data.length);
    MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);
    return segment;
  }

  @Nested
  static class MinimalHeader {

    @Test
    void parsesMinimalV3Header() {
      byte[] data = new SyntheticGgufBuilder().version(3).build();
      var segment = Arena.ofConfined().allocate(data.length);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);

      var result = GgufHeaderParser.parse(segment);

      assertThat(result.header().version()).isEqualTo(3);
      assertThat(result.header().tensorCount()).isZero();
      assertThat(result.header().metadataKvCount()).isZero();
      assertThat(result.metadata().entries()).isEmpty();
    }

    @Test
    void parsesMinimalV2Header() {
      byte[] data = new SyntheticGgufBuilder().version(2).build();
      var segment = Arena.ofConfined().allocate(data.length);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);

      var result = GgufHeaderParser.parse(segment);

      assertThat(result.header().version()).isEqualTo(2);
    }
  }

  @Nested
  static class MetadataTypes {

    @Test
    void parsesStringMetadata() {
      byte[] data = new SyntheticGgufBuilder().addString("general.name", "TestModel").build();
      var segment = Arena.ofConfined().allocate(data.length);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);

      var result = GgufHeaderParser.parse(segment);

      assertThat(result.metadata().getString("general.name")).hasValue("TestModel");
    }

    @Test
    void parsesUint32Metadata() {
      byte[] data = new SyntheticGgufBuilder().addUint32("llama.block_count", 22).build();
      var segment = Arena.ofConfined().allocate(data.length);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);

      var result = GgufHeaderParser.parse(segment);

      assertThat(result.metadata().getUint32("llama.block_count")).hasValue(22);
    }

    @Test
    void parsesFloat32Metadata() {
      byte[] data = new SyntheticGgufBuilder().addFloat32("llama.rope.freq_base", 10000.0f).build();
      var segment = Arena.ofConfined().allocate(data.length);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);

      var result = GgufHeaderParser.parse(segment);

      assertThat(result.metadata().getFloat32("llama.rope.freq_base")).hasValue(10000.0f);
    }

    @Test
    void parsesBoolMetadata() {
      byte[] data = new SyntheticGgufBuilder().addBool("general.quantized", true).build();
      var segment = Arena.ofConfined().allocate(data.length);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);

      var result = GgufHeaderParser.parse(segment);

      assertThat(result.metadata().getBool("general.quantized")).hasValue(true);
    }

    @Test
    void parsesStringArrayMetadata() {
      List<String> tokens = List.of("hello", "world", "test");
      byte[] data =
          new SyntheticGgufBuilder().addStringArray("tokenizer.ggml.tokens", tokens).build();
      var segment = Arena.ofConfined().allocate(data.length);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);

      var result = GgufHeaderParser.parse(segment);

      assertThat(result.metadata().getStringArray("tokenizer.ggml.tokens")).hasValue(tokens);
    }

    @Test
    void parsesMixedMetadata() {
      byte[] data =
          new SyntheticGgufBuilder()
              .addString("general.name", "MyModel")
              .addUint32("llama.block_count", 16)
              .addFloat32("llama.rope.freq_base", 500000.0f)
              .addBool("general.quantized", false)
              .build();
      var segment = Arena.ofConfined().allocate(data.length);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);

      var result = GgufHeaderParser.parse(segment);

      assertThat(result.header().metadataKvCount()).isEqualTo(4);
      assertThat(result.metadata().getString("general.name")).hasValue("MyModel");
      assertThat(result.metadata().getUint32("llama.block_count")).hasValue(16);
      assertThat(result.metadata().getFloat32("llama.rope.freq_base")).hasValue(500000.0f);
      assertThat(result.metadata().getBool("general.quantized")).hasValue(false);
    }
  }

  @Nested
  static class ErrorCases {

    @Test
    void wrongMagicThrows() {
      byte[] data = new byte[24];
      ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).putInt(0, 0xDEADBEEF);

      var segment = Arena.ofConfined().allocate(data.length);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);

      assertThatThrownBy(() -> GgufHeaderParser.parse(segment))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("Invalid GGUF magic");
    }

    @Test
    void unsupportedVersionThrows() {
      byte[] data = new byte[24];
      ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(0, GgufConstants.MAGIC);
      buf.putInt(4, 1); // unsupported version

      var segment = Arena.ofConfined().allocate(data.length);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);

      assertThatThrownBy(() -> GgufHeaderParser.parse(segment))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("Unsupported GGUF version");
    }
  }

  @Nested
  static class TypedAccessors {

    @Test
    void missingKeyReturnsEmpty() {
      byte[] data = new SyntheticGgufBuilder().addString("foo", "bar").build();
      var segment = Arena.ofConfined().allocate(data.length);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);

      var result = GgufHeaderParser.parse(segment);

      assertThat(result.metadata().getString("nonexistent")).isEmpty();
      assertThat(result.metadata().getUint32("foo")).isEmpty(); // wrong type
    }

    @Test
    void wrongTypeReturnsEmpty() {
      byte[] data = new SyntheticGgufBuilder().addUint32("count", 5).build();
      var segment = Arena.ofConfined().allocate(data.length);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);

      var result = GgufHeaderParser.parse(segment);

      assertThat(result.metadata().getString("count")).isEmpty();
      assertThat(result.metadata().getFloat32("count")).isEmpty();
    }
  }
}
