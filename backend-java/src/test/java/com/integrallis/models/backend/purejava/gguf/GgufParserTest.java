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

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class GgufParserTest {

  @Nested
  class SingleTensor {

    @Test
    void parsesOneF32Tensor() {
      // Create a 2x3 F32 tensor with known values
      float[] values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
      byte[] tensorData = new byte[values.length * 4];
      ByteBuffer.wrap(tensorData).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().put(values);

      byte[] data =
          new SyntheticGgufBuilder()
              .addTensor("weight", GgufTensorType.F32, new long[] {3, 2}, tensorData)
              .build();

      var segment = Arena.ofConfined().allocate(data.length);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);

      GgufFile file = GgufParser.parseSegment(segment);

      assertThat(file.tensorInfos()).hasSize(1);
      GgufTensorInfo info = file.tensorInfos().getFirst();
      assertThat(info.name()).isEqualTo("weight");
      assertThat(info.type()).isEqualTo(GgufTensorType.F32);
      assertThat(info.nDimensions()).isEqualTo(2);
      assertThat(info.shape()).containsExactly(3, 2);
      assertThat(info.elementCount()).isEqualTo(6);
      assertThat(info.byteSize()).isEqualTo(24);

      // Read back the tensor data
      GgufTensorData tensorDataResult = file.getTensor("weight");
      assertThat(tensorDataResult.name()).isEqualTo("weight");
      MemorySegment slice = tensorDataResult.dataSegment();
      for (int i = 0; i < values.length; i++) {
        float read =
            slice.get(
                ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), (long) i * 4);
        assertThat(read).isEqualTo(values[i]);
      }
    }
  }

  @Nested
  class MultipleTensors {

    @Test
    void parsesTwoTensorsOfDifferentTypes() {
      // F32 tensor: 4 elements
      float[] f32Values = {1.0f, 2.0f, 3.0f, 4.0f};
      byte[] f32Data = new byte[16];
      ByteBuffer.wrap(f32Data).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().put(f32Values);

      // Q4_0 tensor: 32 elements (one block = 18 bytes)
      byte[] q4Data = new byte[18];
      // scale (f16 for 1.0 = 0x3C00) + 16 nibble bytes
      ByteBuffer.wrap(q4Data)
          .order(ByteOrder.LITTLE_ENDIAN)
          .putShort(0, (short) 0x3C00); // scale = 1.0 in f16

      byte[] data =
          new SyntheticGgufBuilder()
              .addTensor("embed", GgufTensorType.F32, new long[] {4}, f32Data)
              .addTensor("qweight", GgufTensorType.Q4_0, new long[] {32}, q4Data)
              .build();

      var segment = Arena.ofConfined().allocate(data.length);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);

      GgufFile file = GgufParser.parseSegment(segment);

      assertThat(file.tensorInfos()).hasSize(2);
      assertThat(file.getTensor("embed").type()).isEqualTo(GgufTensorType.F32);
      assertThat(file.getTensor("qweight").type()).isEqualTo(GgufTensorType.Q4_0);
    }
  }

  @Nested
  class Alignment {

    @Test
    void tensorDataIsAlignedToDefaultAlignment() {
      float[] values = {42.0f};
      byte[] tensorData = new byte[4];
      ByteBuffer.wrap(tensorData).order(ByteOrder.LITTLE_ENDIAN).putFloat(0, 42.0f);

      byte[] data =
          new SyntheticGgufBuilder()
              .addString("general.name", "test")
              .addTensor("x", GgufTensorType.F32, new long[] {1}, tensorData)
              .build();

      var segment = Arena.ofConfined().allocate(data.length);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);

      GgufFile file = GgufParser.parseSegment(segment);

      // Tensor data offset should be aligned to 32 bytes
      assertThat(file.tensorDataOffset() % GgufConstants.DEFAULT_ALIGNMENT).isZero();
    }
  }

  @Nested
  class FileParsing {

    @Test
    void parsesFromDisk(@TempDir Path tempDir) throws IOException {
      float[] values = {1.0f, 2.0f, 3.0f};
      byte[] tensorData = new byte[12];
      ByteBuffer.wrap(tensorData).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().put(values);

      byte[] data =
          new SyntheticGgufBuilder()
              .addString("general.name", "DiskModel")
              .addTensor("w", GgufTensorType.F32, new long[] {3}, tensorData)
              .build();

      Path ggufPath = tempDir.resolve("test.gguf");
      Files.write(ggufPath, data);

      try (Arena arena = Arena.ofConfined()) {
        GgufFile file = GgufParser.parse(ggufPath, arena);
        assertThat(file.metadata().getString("general.name")).hasValue("DiskModel");
        assertThat(file.tensorInfos()).hasSize(1);

        GgufTensorData tensor = file.getTensor("w");
        MemorySegment slice = tensor.dataSegment();
        float first =
            slice.get(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), 0);
        assertThat(first).isEqualTo(1.0f);
      }
    }

    @Test
    void nonExistentFileThrows(@TempDir Path tempDir) {
      Path noFile = tempDir.resolve("nonexistent.gguf");
      try (Arena arena = Arena.ofConfined()) {
        assertThatThrownBy(() -> GgufParser.parse(noFile, arena)).isInstanceOf(IOException.class);
      }
    }
  }

  @Nested
  class TensorNotFound {

    @Test
    void throwsForNonExistentTensor() {
      byte[] data = new SyntheticGgufBuilder().build();
      var segment = Arena.ofConfined().allocate(data.length);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);

      GgufFile file = GgufParser.parseSegment(segment);

      assertThatThrownBy(() -> file.getTensor("missing"))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("Tensor not found");
    }
  }
}
