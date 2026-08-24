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
package com.integrallis.models.backend.purejava.tensor;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import com.integrallis.models.backend.purejava.safetensors.SafetensorsBundle;
import com.integrallis.models.backend.purejava.safetensors.SyntheticSafetensorsBuilder;
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
class TensorSourceTest {

  @Test
  void ggufAndSafetensorsExposeTheSameLogicalF32Matrix(@TempDir Path directory) throws IOException {
    byte[] values = f32(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
    byte[] ggufArtifact =
        new SyntheticGgufBuilder()
            .addTensor("weight", GgufTensorType.F32, new long[] {3, 2}, values)
            .build();
    GgufFile gguf = GgufParser.parseSegment(MemorySegment.ofArray(ggufArtifact));
    Path safetensorsPath = directory.resolve("model.safetensors");
    Files.write(
        safetensorsPath,
        new SyntheticSafetensorsBuilder()
            .add("weight", "F32", new long[] {2, 3}, unsigned(values))
            .build());

    try (Arena arena = Arena.ofConfined()) {
      TensorSource ggufSource = new GgufTensorSource(gguf);
      TensorSource safetensorsSource =
          new SafetensorsTensorSource(SafetensorsBundle.open(safetensorsPath, arena));

      assertThat(ggufSource.format()).isEqualTo("gguf");
      assertThat(safetensorsSource.format()).isEqualTo("safetensors");
      assertThat(ggufSource.tensorNames()).containsExactly("weight");
      assertThat(safetensorsSource.tensorNames()).containsExactly("weight");

      TensorView ggufView = ggufSource.tensor("weight");
      TensorView safetensorsView = safetensorsSource.tensor("weight");
      assertThat(ggufView.shape()).containsExactly(2, 3);
      assertThat(safetensorsView.shape()).containsExactly(2, 3);
      assertThat(ggufView.storage()).isEqualTo(new TensorStorage("gguf", "F32", 1, 4));
      assertThat(safetensorsView.storage())
          .isEqualTo(new TensorStorage("safetensors", "F32", 1, 4));
      assertThat(ggufView.data().toArray(ValueLayout.JAVA_BYTE)).containsExactly(values);
      assertThat(safetensorsView.data().toArray(ValueLayout.JAVA_BYTE)).containsExactly(values);
      assertThat(ggufView.data().isReadOnly()).isTrue();
      assertThat(safetensorsView.data().isReadOnly()).isTrue();

      // The adapter normalizes GGUF's fastest-dimension-first shape without changing GGUF itself.
      assertThat(gguf.getTensor("weight").shape()).containsExactly(3, 2);
    }
  }

  @Test
  void preservesFormatSpecificPackedBlockGeometry(@TempDir Path directory) throws IOException {
    byte[] q4 = new byte[36];
    GgufFile gguf =
        GgufParser.parseSegment(
            MemorySegment.ofArray(
                new SyntheticGgufBuilder()
                    .addTensor("packed", GgufTensorType.Q4_0, new long[] {32, 2}, q4)
                    .build()));
    Path safetensorsPath = directory.resolve("model.safetensors");
    Files.write(
        safetensorsPath,
        new SyntheticSafetensorsBuilder()
            .add("packed", "F4", new long[] {2, 32}, new int[32])
            .build());

    try (Arena arena = Arena.ofConfined()) {
      TensorView ggufView = new GgufTensorSource(gguf).tensor("packed");
      TensorView safetensorsView =
          new SafetensorsTensorSource(SafetensorsBundle.open(safetensorsPath, arena))
              .tensor("packed");

      assertThat(ggufView.storage()).isEqualTo(new TensorStorage("gguf", "Q4_0", 32, 18));
      assertThat(safetensorsView.storage()).isEqualTo(new TensorStorage("safetensors", "F4", 2, 1));
      assertThat(ggufView.data().byteSize()).isEqualTo(36);
      assertThat(safetensorsView.data().byteSize()).isEqualTo(32);
    }
  }

  private static byte[] f32(float... values) {
    byte[] bytes = new byte[values.length * Float.BYTES];
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().put(values);
    return bytes;
  }

  private static int[] unsigned(byte[] values) {
    int[] result = new int[values.length];
    for (int index = 0; index < values.length; index++) {
      result[index] = Byte.toUnsignedInt(values[index]);
    }
    return result;
  }
}
