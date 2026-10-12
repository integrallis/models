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

import com.integrallis.vectors.core.VectorUtil;
import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Pins row-order arithmetic while the production kernel tiles columns and partitions heads. */
@Tag("unit")
class GatedDeltaNetKernelTest {
  @Test
  void matchesRowOrderAcrossTailsPrefixesAndStrongDecay() {
    for (int dimension : new int[] {7, 128, 256}) {
      for (int tokens : new int[] {1, 7, 18}) {
        int keyHeads = 2, valueHeads = 4;
        Random random = new Random(91L + dimension + tokens);
        float[] query = values(random, tokens * keyHeads * dimension);
        float[] key = values(random, query.length);
        float[] value = values(random, tokens * valueHeads * dimension);
        float[] beta = values(random, tokens * valueHeads);
        float[] decay = values(random, beta.length);
        for (int n = 0; n < decay.length; n++) decay[n] = n % 3 == 0 ? -70f : -Math.abs(decay[n]);
        float[] actualState = values(random, valueHeads * dimension * dimension);
        float[] expectedState = actualState.clone();
        float[] actual = new float[value.length + 5];
        Arrays.fill(actual, 919f);
        float[] expected = actual.clone();
        reference(
            query,
            key,
            value,
            decay,
            beta,
            expectedState,
            0,
            expected,
            tokens,
            keyHeads,
            valueHeads,
            dimension);
        GatedDeltaNetRecurrence.forwardPrefixInPlace(
            query,
            key,
            value,
            decay,
            beta,
            actualState,
            actual,
            tokens,
            keyHeads,
            valueHeads,
            dimension,
            dimension);
        assertThat(actual)
            .as("outputs and tail, dimension=%d tokens=%d", dimension, tokens)
            .containsExactly(expected);
        assertThat(actualState)
            .as("state, dimension=%d tokens=%d", dimension, tokens)
            .containsExactly(expectedState);
      }
    }
  }

  @Test
  void advancesOnlyTheRequestedStateSlot() {
    int dimension = 7, keyHeads = 2, valueHeads = 4;
    int slotSize = valueHeads * dimension * dimension;
    Random random = new Random(723);
    float[] query = values(random, keyHeads * dimension), key = values(random, query.length);
    float[] value = values(random, valueHeads * dimension);
    float[] decay = new float[valueHeads], beta = values(random, valueHeads);
    float[] state = values(random, 4 * slotSize), expectedState = state.clone();
    float[] output = new float[value.length], expected = output.clone();
    reference(
        query,
        key,
        value,
        decay,
        beta,
        expectedState,
        2 * slotSize,
        expected,
        1,
        keyHeads,
        valueHeads,
        dimension);
    GatedDeltaNetRecurrence.forwardSlotInPlace(
        query,
        key,
        value,
        decay,
        beta,
        state,
        2 * slotSize,
        output,
        keyHeads,
        valueHeads,
        dimension,
        dimension);
    assertThat(output).containsExactly(expected);
    assertThat(state).containsExactly(expectedState);
  }

  @Test
  void rejectsAliasedWritesBeforeStartingWorkers() {
    float[] shared = new float[4];
    assertThatThrownBy(
            () ->
                GatedDeltaNetRecurrence.forwardPrefixInPlace(
                    new float[2],
                    new float[2],
                    new float[2],
                    new float[1],
                    new float[1],
                    shared,
                    shared,
                    1,
                    1,
                    1,
                    2,
                    2))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("alias");
  }

  private static float[] values(Random random, int count) {
    float[] values = new float[count];
    for (int n = 0; n < count; n++) values[n] = random.nextFloat() * 0.5f - 0.25f;
    return values;
  }

  private static void normalize(float[] input, int offset, float[] output) {
    float sum = 0;
    for (int n = 0; n < output.length; n++) sum += input[offset + n] * input[offset + n];
    float inverse = (float) (1.0 / Math.sqrt(sum + 1e-6f));
    for (int n = 0; n < output.length; n++) output[n] = input[offset + n] * inverse;
  }

  private static void reference(
      float[] query,
      float[] key,
      float[] value,
      float[] logDecay,
      float[] beta,
      float[] state,
      int stateBase,
      float[] output,
      int tokens,
      int keyHeads,
      int valueHeads,
      int dimension) {
    float[] q = new float[dimension], k = new float[dimension];
    float[] memory = new float[dimension], delta = new float[dimension];
    float scale = (float) (1.0 / Math.sqrt(dimension));
    for (int token = 0; token < tokens; token++) {
      for (int head = 0; head < valueHeads; head++) {
        int th = token * valueHeads + head;
        int inputAt = (token * keyHeads + head % keyHeads) * dimension;
        normalize(query, inputAt, q);
        normalize(key, inputAt, k);
        for (int n = 0; n < dimension; n++) q[n] *= scale;
        int stateAt = stateBase + head * dimension * dimension;
        float decay = (float) Math.exp(logDecay[th]);
        for (int n = 0; n < dimension * dimension; n++) state[stateAt + n] *= decay;
        Arrays.fill(memory, 0);
        for (int row = 0; row < dimension; row++)
          VectorUtil.addScaledInPlace(
              memory, 0, state, stateAt + row * dimension, dimension, k[row]);
        int outAt = th * dimension;
        for (int n = 0; n < dimension; n++) delta[n] = (value[outAt + n] - memory[n]) * beta[th];
        for (int row = 0; row < dimension; row++)
          VectorUtil.addScaledInPlace(
              state, stateAt + row * dimension, delta, 0, dimension, k[row]);
        Arrays.fill(output, outAt, outAt + dimension, 0);
        for (int row = 0; row < dimension; row++)
          VectorUtil.addScaledInPlace(
              output, outAt, state, stateAt + row * dimension, dimension, q[row]);
      }
    }
  }
}
