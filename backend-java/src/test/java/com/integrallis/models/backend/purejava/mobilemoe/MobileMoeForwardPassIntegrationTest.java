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
package com.integrallis.models.backend.purejava.mobilemoe;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.backend.purejava.safetensors.SafetensorsBundle;
import com.integrallis.models.backend.purejava.tensor.SafetensorsTensorSource;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class MobileMoeForwardPassIntegrationTest {

  @Test
  void executesTheRealQatGraphWithIsolatedRewindableSessions() throws Exception {
    Path directory = fixtureDirectory();
    assumeTrue(Files.isRegularFile(directory.resolve("model.safetensors")));
    MobileMoeHuggingFaceConfig config =
        MobileMoeHuggingFaceConfig.parse(directory.resolve("config.json"));
    try (Arena arena = Arena.ofConfined()) {
      MobileMoeForwardPass graph =
          MobileMoeForwardPass.load(
              config, new SafetensorsTensorSource(SafetensorsBundle.open(directory, arena)), 16);
      MobileMoeForwardPass.Session first = graph.openSession();
      MobileMoeForwardPass.Session second = graph.openSession();

      float[] firstLogits = graph.forward(first, 128000, 0);
      float[] secondLogits = graph.forward(second, 128000, 0);
      assertEquals(config.vocabSize(), firstLogits.length);
      assertTrue(allFinite(firstLogits));
      assertArrayEquals(firstLogits, secondLogits);

      graph.advance(first, 9906, 1);
      assertEquals(2, first.checkpoint());
      graph.rewind(first, 1);
      assertEquals(1, first.checkpoint());
      graph.reset(first);
      assertEquals(0, first.checkpoint());
      assertArrayEquals(firstLogits, graph.forward(first, 128000, 0));
    }
  }

  @Test
  void matchesTheOfficialPytorchOracleAtPositionZeroAndWithKvHistory() throws Exception {
    Path directory = fixtureDirectory();
    Path bosOracle = directory.resolve("bos.f32le");
    Path helloOracle = directory.resolve("bos_hello.f32le");
    assumeTrue(Files.isRegularFile(directory.resolve("model.safetensors")));
    assumeTrue(Files.isRegularFile(bosOracle) && Files.isRegularFile(helloOracle));
    MobileMoeHuggingFaceConfig config =
        MobileMoeHuggingFaceConfig.parse(directory.resolve("config.json"));
    try (Arena arena = Arena.ofConfined()) {
      MobileMoeForwardPass graph =
          MobileMoeForwardPass.load(
              config, new SafetensorsTensorSource(SafetensorsBundle.open(directory, arena)), 16);
      MobileMoeForwardPass.Session session = graph.openSession();
      float[] bos = graph.forward(session, 128000, 0);
      float[] expectedBos = readFloats(bosOracle);
      assertEquals(argmax(expectedBos), argmax(bos));
      assertTrue(cosine(expectedBos, bos) > 0.9999, () -> "BOS cosine=" + cosine(expectedBos, bos));

      float[] hello = graph.forward(session, 9906, 1);
      float[] expectedHello = readFloats(helloOracle);
      assertEquals(argmax(expectedHello), argmax(hello));
      assertTrue(
          cosine(expectedHello, hello) > 0.9997,
          () -> "BOS+hello cosine=" + cosine(expectedHello, hello));
    }
  }

  private static boolean allFinite(float[] values) {
    for (float value : values) {
      if (!Float.isFinite(value)) {
        return false;
      }
    }
    return true;
  }

  private static float[] readFloats(Path path) throws Exception {
    ByteBuffer bytes = ByteBuffer.wrap(Files.readAllBytes(path)).order(ByteOrder.LITTLE_ENDIAN);
    float[] values = new float[bytes.remaining() / Float.BYTES];
    bytes.asFloatBuffer().get(values);
    return values;
  }

  private static int argmax(float[] values) {
    int best = 0;
    for (int index = 1; index < values.length; index++) {
      if (values[index] > values[best]) {
        best = index;
      }
    }
    return best;
  }

  private static double cosine(float[] left, float[] right) {
    double dot = 0.0;
    double leftNorm = 0.0;
    double rightNorm = 0.0;
    for (int index = 0; index < left.length; index++) {
      dot += (double) left[index] * right[index];
      leftNorm += (double) left[index] * left[index];
      rightNorm += (double) right[index] * right[index];
    }
    return dot / Math.sqrt(leftNorm * rightNorm);
  }

  private static Path fixtureDirectory() {
    String configured = System.getProperty("models.fixtures.mobileMoeQatDirectory", "");
    return configured.isBlank() ? Path.of("missing-mobilemoe-qat-fixture") : Path.of(configured);
  }
}
