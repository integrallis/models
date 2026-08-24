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
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class CactCqMatrixTest {

  @Test
  void executesAHeapBackedCqTensorWithoutExpandingIt() {
    CactFile file =
        CactParser.parseSegment(
            MemorySegment.ofArray(new SyntheticCactBuilder().addCq(1, 8, 8, 2).build()));
    CactCqMatrix matrix = CactCqMatrix.from(file.tensor(0), file.codebook());
    float[] input = new float[8];
    Arrays.fill(input, 1.0f);
    float[] output = new float[1];
    float[] row = new float[8];

    matrix.multiply(matrix.prepare(input), output);
    matrix.decodeRow(0, row);

    assertThat(output[0]).isCloseTo(-0.29831067f, within(1.0e-6f));
    assertThat(row[0]).isCloseTo(-0.29831067f, within(1.0e-6f));
    assertThat(Arrays.copyOfRange(row, 1, row.length)).containsOnly(0.0f);
    assertThat(matrix.rowSlice(0, 1).rows()).isEqualTo(1);
  }

  @Test
  void multipliesPinnedOfficialCq4AndCq2MatricesWhenProvided() throws IOException {
    String configured = System.getProperty("models.fixtures.needle2Cact", "");
    assumeTrue(!configured.isBlank(), "set -Dmodels.fixtures.needle2Cact=<needle2.cact>");
    Path path = Path.of(configured);
    assumeTrue(Files.isRegularFile(path), "Needle 2 fixture is not installed");

    try (Arena arena = Arena.ofConfined()) {
      CactFile file = CactParser.parse(path, arena);
      CactNeedle2Layout layout = CactNeedle2Layout.from(file);

      assertReference(
          CactCqMatrix.from(layout.tensor("embedding"), file.codebook()),
          new float[] {
            377.4586f, 37.30397f, 19.889656f, 162.97917f,
            33.143265f, -9.136576f, 15.693462f, 29.56194f
          },
          73_071.0820253,
          3_595.49609375);
      assertReference(
          CactCqMatrix.from(layout.tensor("layer00.q_proj"), file.codebook()),
          new float[] {
            5.417707f, -5.05807f, 6.407853f, -4.022003f,
            -7.8131833f, 9.420242f, 12.049175f, -9.097509f
          },
          161.83702159,
          227.51327515);
    }
  }

  private static void assertReference(
      CactCqMatrix matrix, float[] first, double expectedSum, double expectedL2) {
    float[] input = new float[matrix.columns()];
    for (int index = 0; index < input.length; index++) {
      input[index] = (float) (Math.sin(index * 0.03125) + 0.25 * Math.cos(index * 0.0078125));
    }
    float[] output = new float[matrix.rows()];

    matrix.multiply(matrix.prepare(input), output);

    assertThat(output).startsWith(first, within(1.0e-3f));
    double sum = 0.0;
    double squared = 0.0;
    for (float value : output) {
      sum += value;
      squared += (double) value * value;
    }
    assertThat(sum).isCloseTo(expectedSum, within(0.01));
    assertThat(Math.sqrt(squared)).isCloseTo(expectedL2, within(0.01));
  }
}
