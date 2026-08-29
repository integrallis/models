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
package com.integrallis.models.accelerator;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Random;
import org.junit.jupiter.api.Test;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

class CausalAttentionKernelTest {
  private static final int EXECUTION_BATCH = 4;
  private static final int HEADS = 4;
  private static final int KV_HEADS = 2;
  private static final int HEAD_LENGTH = 8;
  private static final int QUERY_DIM = HEADS * HEAD_LENGTH;
  private static final int KV_DIM = KV_HEADS * HEAD_LENGTH;
  private static final int MAX_SEQUENCE = 12;

  @Test
  void matchesGroupedQueryAttentionAcrossIncrementalPromptChunks() {
    FloatArray keyCache = new FloatArray(MAX_SEQUENCE * KV_DIM);
    FloatArray valueCache = new FloatArray(MAX_SEQUENCE * KV_DIM);
    FloatArray scores = new FloatArray(EXECUTION_BATCH * HEADS * MAX_SEQUENCE);
    FloatArray output = new FloatArray(EXECUTION_BATCH * QUERY_DIM);
    IntArray state = new IntArray(2);
    float[] referenceKeys = new float[MAX_SEQUENCE * KV_DIM];
    float[] referenceValues = new float[MAX_SEQUENCE * KV_DIM];

    runAndCompareChunk(
        0, 4, 41L, keyCache, valueCache, scores, output, state, referenceKeys, referenceValues);
    runAndCompareChunk(
        4, 3, 43L, keyCache, valueCache, scores, output, state, referenceKeys, referenceValues);
  }

  private static void runAndCompareChunk(
      int startPosition,
      int actualBatch,
      long seed,
      FloatArray keyCache,
      FloatArray valueCache,
      FloatArray scores,
      FloatArray output,
      IntArray state,
      float[] referenceKeys,
      float[] referenceValues) {
    float[] query = randomFloats(EXECUTION_BATCH * QUERY_DIM, seed);
    float[] key = randomFloats(EXECUTION_BATCH * KV_DIM, seed + 1);
    float[] value = randomFloats(EXECUTION_BATCH * KV_DIM, seed + 2);
    state.set(0, startPosition);
    state.set(1, actualBatch);
    FloatArray deviceQuery = FloatArray.fromArray(query);
    FloatArray deviceKey = FloatArray.fromArray(key);
    FloatArray deviceValue = FloatArray.fromArray(value);

    CausalAttentionKernel.store(
        state, deviceKey, deviceValue, keyCache, valueCache, KV_DIM, KV_DIM);
    CausalAttentionKernel.attend(
        state,
        deviceQuery,
        keyCache,
        valueCache,
        scores,
        output,
        HEADS,
        KV_HEADS,
        HEAD_LENGTH,
        HEAD_LENGTH,
        KV_DIM,
        KV_DIM,
        MAX_SEQUENCE,
        0);

    storeReference(
        startPosition, actualBatch, key, value, referenceKeys, referenceValues, KV_DIM, KV_DIM);
    float[] expected =
        referenceAttention(
            startPosition,
            actualBatch,
            query,
            referenceKeys,
            referenceValues,
            HEADS,
            KV_HEADS,
            HEAD_LENGTH,
            HEAD_LENGTH,
            KV_DIM,
            KV_DIM);
    assertThat(output.toHeapArray())
        .usingComparatorWithPrecision(1.0e-5f)
        .containsExactly(expected);
  }

  private static float[] randomFloats(int length, long seed) {
    Random random = new Random(seed);
    float[] values = new float[length];
    for (int index = 0; index < values.length; index++) {
      values[index] = random.nextFloat(-1.0f, 1.0f);
    }
    return values;
  }

  private static void storeReference(
      int startPosition,
      int actualBatch,
      float[] key,
      float[] value,
      float[] keyCache,
      float[] valueCache,
      int keyDim,
      int valueDim) {
    for (int batch = 0; batch < actualBatch; batch++) {
      System.arraycopy(key, batch * keyDim, keyCache, (startPosition + batch) * keyDim, keyDim);
      System.arraycopy(
          value, batch * valueDim, valueCache, (startPosition + batch) * valueDim, valueDim);
    }
  }

  private static float[] referenceAttention(
      int startPosition,
      int actualBatch,
      float[] query,
      float[] keyCache,
      float[] valueCache,
      int heads,
      int kvHeads,
      int keyLength,
      int valueLength,
      int keyDim,
      int valueDim) {
    float[] output = new float[EXECUTION_BATCH * heads * valueLength];
    float scale = (float) (1.0 / Math.sqrt(keyLength));
    int groupSize = heads / kvHeads;
    for (int batch = 0; batch < actualBatch; batch++) {
      int position = startPosition + batch;
      for (int head = 0; head < heads; head++) {
        int kvHead = head / groupSize;
        float[] attention = new float[position + 1];
        float max = Float.NEGATIVE_INFINITY;
        for (int cached = 0; cached <= position; cached++) {
          float dot = 0.0f;
          for (int column = 0; column < keyLength; column++) {
            dot +=
                query[batch * heads * keyLength + head * keyLength + column]
                    * keyCache[cached * keyDim + kvHead * keyLength + column];
          }
          attention[cached] = dot * scale;
          max = Math.max(max, attention[cached]);
        }
        float sum = 0.0f;
        for (int cached = 0; cached <= position; cached++) {
          attention[cached] = (float) Math.exp(attention[cached] - max);
          sum += attention[cached];
        }
        for (int column = 0; column < valueLength; column++) {
          float weighted = 0.0f;
          for (int cached = 0; cached <= position; cached++) {
            weighted +=
                attention[cached] * valueCache[cached * valueDim + kvHead * valueLength + column];
          }
          output[batch * heads * valueLength + head * valueLength + column] = weighted / sum;
        }
      }
    }
    return output;
  }
}
