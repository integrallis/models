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
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
class Needle2ForwardPassTest {

  @Test
  void matchesPinnedNeedleJaxReferenceForBosTokenWhenProvided() throws IOException {
    String configured = System.getProperty("models.fixtures.needle2Cact", "");
    assumeTrue(!configured.isBlank(), "set -Dmodels.fixtures.needle2Cact=<needle2.cact>");
    Path path = Path.of(configured);
    assumeTrue(Files.isRegularFile(path), "Needle 2 fixture is not installed");

    try (Arena arena = Arena.ofConfined()) {
      CactFile file = CactParser.parse(path, arena);
      Needle2Weights weights = Needle2Weights.load(CactNeedle2Layout.from(file));
      Needle2ForwardPass forwardPass = new Needle2ForwardPass(weights, 8);

      assertThat(weights.contrastiveHead()).isPresent();
      assertThat(weights.contrastiveHead().orElseThrow().outputWidth()).isEqualTo(128);
      assertThat(weights.confidenceHead()).isPresent();
      assertThat(weights.confidenceHead().orElseThrow().outputWidth()).isEqualTo(1);

      float[] logits = forwardPass.forward(2, 0);

      assertThat(logits)
          .startsWith(
              new float[] {
                -72.174896f,
                -5.2631702f,
                7.9116025f,
                -103.985565f,
                -3.7915947f,
                -4.0640283f,
                -6.7531204f,
                -5.2831926f
              },
              within(0.005f));
      assertThat(argmax(logits)).isEqualTo(2);
      assertThat(sum(logits)).isCloseTo(-128_023.94884, within(1.0));
      assertThat(l2(logits)).isCloseTo(1_850.20093, within(0.05));

      float[] retrievalEmbedding = forwardPass.encodeContrastive(new int[] {2});
      assertThat(retrievalEmbedding).hasSize(128);
      assertThat(allFinite(retrievalEmbedding)).isTrue();
      assertThat(l2(retrievalEmbedding)).isCloseTo(1.0, within(1.0e-5));
      assertThat(forwardPass.scoreConfidence(new int[] {2})).isBetween(0.0f, 1.0f);
    }
  }

  @Test
  void advancingWithoutLogitsPreservesTheNextTokenResultWhenProvided() throws IOException {
    String configured = System.getProperty("models.fixtures.needle2Cact", "");
    assumeTrue(!configured.isBlank(), "set -Dmodels.fixtures.needle2Cact=<needle2.cact>");
    Path path = Path.of(configured);
    assumeTrue(Files.isRegularFile(path), "Needle 2 fixture is not installed");

    try (Arena arena = Arena.ofConfined()) {
      Needle2Weights weights =
          Needle2Weights.load(CactNeedle2Layout.from(CactParser.parse(path, arena)));
      Needle2ForwardPass optimized = new Needle2ForwardPass(weights, 8);
      Needle2ForwardPass reference = new Needle2ForwardPass(weights, 8);

      optimized.advance(2, 0);
      float[] actual = optimized.forward(3, 1);
      reference.forward(2, 0);
      float[] expected = reference.forward(3, 1);

      assertThat(actual).containsExactly(expected);
      assertThat(optimized.checkpoint()).isEqualTo(2);
    }
  }

  private static int argmax(float[] values) {
    int result = 0;
    for (int index = 1; index < values.length; index++) {
      if (values[index] > values[result]) {
        result = index;
      }
    }
    return result;
  }

  private static double sum(float[] values) {
    double result = 0.0;
    for (float value : values) {
      result += value;
    }
    return result;
  }

  private static double l2(float[] values) {
    double squared = 0.0;
    for (float value : values) {
      squared += (double) value * value;
    }
    return Math.sqrt(squared);
  }

  private static boolean allFinite(float[] values) {
    for (float value : values) {
      if (!Float.isFinite(value)) {
        return false;
      }
    }
    return true;
  }

  @Test
  void reusesLaneScratchAcrossTokensRatherThanAllocatingPerLayer() throws IOException {
    // executeLayer allocated preWeights, postWeights and a laneCount-by-modelWidth lane array on
    // every layer of every token. Allocating per call produces identical numbers, so only identity
    // shows the reuse.
    //
    // This assertion is gated on the Needle 2 fixture like every other test in this file, because
    // the graph cannot be built without one. It therefore does NOT guard the reuse in CI; it
    // guards it for whoever runs the fixture. #373 asked for an observable assertion and this is
    // as observable as the fixture situation allows.
    String configured = System.getProperty("models.fixtures.needle2Cact", "");
    assumeTrue(!configured.isBlank(), "set -Dmodels.fixtures.needle2Cact=<needle2.cact>");
    Path path = Path.of(configured);
    assumeTrue(Files.isRegularFile(path), "Needle 2 fixture is not installed");

    try (Arena arena = Arena.ofConfined()) {
      Needle2Weights weights =
          Needle2Weights.load(CactNeedle2Layout.from(CactParser.parse(path, arena)));
      Needle2ForwardPass forwardPass = new Needle2ForwardPass(weights, 8);

      forwardPass.forward(2, 0);
      float[][][] afterFirst = forwardPass.laneScratchIdentity();
      float[][] firstLaneRows = afterFirst[0];
      forwardPass.forward(3, 1);
      float[][][] afterSecond = forwardPass.laneScratchIdentity();

      assertThat(afterSecond)
          .as("the lane buffers must not be reallocated per token")
          .isSameAs(afterFirst);
      assertThat(afterSecond[0])
          .as("each lane buffer must keep its identity, not just the holder array")
          .isSameAs(firstLaneRows);
      assertThat(afterSecond[0]).isNotSameAs(afterSecond[1]);
    }
  }
}
