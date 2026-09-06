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
package com.integrallis.models.backend.purejava.bert;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class BertClassificationHeadTest {

  @Test
  void loadsTheEarlierCorrectedClassifierNames(@TempDir Path directory) throws IOException {
    assertHeadScore(
        directory,
        "classifier.dense.weight",
        "classifier.dense.bias",
        "classifier.out_proj.weight",
        "classifier.out_proj.bias");
  }

  @Test
  void loadsTheStandardGgufClassifierNames(@TempDir Path directory) throws IOException {
    assertHeadScore(directory, "cls.weight", "cls.bias", "cls.output.weight", "cls.output.bias");
  }

  private static void assertHeadScore(
      Path directory, String denseWeight, String denseBias, String outputWeight, String outputBias)
      throws IOException {
    byte[] gguf =
        new SyntheticGgufBuilder()
            .addTensor(
                denseWeight, GgufTensorType.F32, new long[] {2, 2}, floats(1.0f, 0.0f, 0.0f, 1.0f))
            .addTensor(denseBias, GgufTensorType.F32, new long[] {2}, floats(0.0f, 0.0f))
            .addTensor(outputWeight, GgufTensorType.F32, new long[] {2, 1}, floats(1.0f, 1.0f))
            .addTensor(outputBias, GgufTensorType.F32, new long[] {1}, floats(0.5f))
            .build();
    Path artifact = directory.resolve("head.gguf");
    Files.write(artifact, gguf);

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(artifact, arena);
      var head = BertClassificationHead.fromGgufFile(file, 2);

      assertThat(head.score(new float[] {1.0f, 2.0f}))
          .isCloseTo(0.5 + Math.tanh(1.0) + Math.tanh(2.0), within(1.0e-6));
    }
  }

  private static byte[] floats(float... values) {
    ByteBuffer bytes =
        ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : values) {
      bytes.putFloat(value);
    }
    return bytes.array();
  }
}
