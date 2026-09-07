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

import java.util.Objects;

/** One ridge mapper per KV head for position-major, head-interleaved samples. */
public final class PerHeadRidgeMapper {
  private final RidgeMapper[] mappers;

  private PerHeadRidgeMapper(RidgeMapper[] mappers) {
    this.mappers = mappers;
  }

  public static PerHeadRidgeMapper fit(
      float[][] inputs, float[][] targets, int heads, double regularization) {
    requireRows(inputs, targets, heads);
    RidgeMapper[] mappers = new RidgeMapper[heads];
    for (int head = 0; head < heads; head++) {
      mappers[head] =
          RidgeMapper.fit(
              rowsForHead(inputs, head, heads), rowsForHead(targets, head, heads), regularization);
    }
    return new PerHeadRidgeMapper(mappers);
  }

  public float[][] predict(float[][] inputs) {
    Objects.requireNonNull(inputs, "inputs");
    if (inputs.length % mappers.length != 0) {
      throw new IllegalArgumentException("sample count must be divisible by head count");
    }
    float[][] result = new float[inputs.length][];
    for (int head = 0; head < mappers.length; head++) {
      float[][] predicted = mappers[head].predict(rowsForHead(inputs, head, mappers.length));
      for (int position = 0; position < predicted.length; position++) {
        result[position * mappers.length + head] = predicted[position];
      }
    }
    return result;
  }

  private static void requireRows(float[][] inputs, float[][] targets, int heads) {
    Objects.requireNonNull(inputs, "inputs");
    Objects.requireNonNull(targets, "targets");
    if (heads <= 0) {
      throw new IllegalArgumentException("heads must be positive");
    }
    if (inputs.length == 0 || inputs.length != targets.length) {
      throw new IllegalArgumentException(
          "input and target sample counts must match and be positive");
    }
    if (inputs.length % heads != 0) {
      throw new IllegalArgumentException("sample count must be divisible by head count");
    }
  }

  private static float[][] rowsForHead(float[][] samples, int head, int heads) {
    float[][] result = new float[samples.length / heads][];
    for (int position = 0; position < result.length; position++) {
      result[position] = samples[position * heads + head];
    }
    return result;
  }
}
