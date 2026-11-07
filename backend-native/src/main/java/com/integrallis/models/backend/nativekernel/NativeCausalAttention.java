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
package com.integrallis.models.backend.nativekernel;

import com.integrallis.models.backend.purejava.spi.BatchedCausalAttentionKernel;
import java.util.HashMap;
import java.util.Map;

/** Owned continuous MHA prefix for the LLaVA decoder, sharing the dense GEMM implementation. */
final class NativeCausalAttention implements BatchedCausalAttentionKernel {
  private final com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel kernels;
  private final NativeDebertaKernels dense;
  private float[] scores = new float[0];
  private final Map<Integer, Layer> layers = new HashMap<>();
  private boolean closed;

  private static final class Layer {
    final float[] keys;
    final float[] values;
    final int dimension;
    int positions;

    Layer(int dimension, int capacity) {
      this.dimension = dimension;
      keys = new float[Math.multiplyExact(dimension, capacity)];
      values = new float[keys.length];
    }
  }

  NativeCausalAttention(
      com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel kernels,
      NativeDebertaKernels dense) {
    this.kernels = kernels;
    this.dense = dense;
  }

  @Override
  public boolean isEligible(
      int layer,
      int start,
      int batch,
      int heads,
      int kvHeads,
      int keyLength,
      int valueLength,
      int capacity,
      int window) {
    if (closed
        || layer < 0
        || start < 0
        || batch < 1
        || heads < 1
        || heads != kvHeads
        || keyLength < 1
        || keyLength != valueLength
        || window != 0
        || capacity < batch
        || start > capacity - batch) return false;
    long dimension = (long) heads * keyLength;
    if (dimension > Integer.MAX_VALUE || dimension * capacity > Integer.MAX_VALUE) return false;
    Layer state = layers.get(layer);
    return state == null
        ? start == 0
        : state.positions == start
            && state.dimension == dimension
            && state.keys.length == (long) capacity * state.dimension;
  }

  @Override
  public void attend(
      float[] output,
      float[] query,
      float[] key,
      float[] value,
      int layer,
      int start,
      int batch,
      int heads,
      int kvHeads,
      int keyLength,
      int valueLength,
      int capacity,
      int window) {
    if (!isEligible(layer, start, batch, heads, kvHeads, keyLength, valueLength, capacity, window))
      throw new IllegalArgumentException("Discontinuous or unsupported native attention");
    int dimension = Math.multiplyExact(heads, keyLength);
    int count = Math.multiplyExact(batch, dimension);
    if (query.length < count || key.length < count || value.length < count || output.length < count)
      throw new IllegalArgumentException("Attention buffer too small");
    if (output == query || output == key || output == value)
      throw new IllegalArgumentException("Attention output must not alias inputs");
    Layer state = layers.computeIfAbsent(layer, ignored -> new Layer(dimension, capacity));
    System.arraycopy(key, 0, state.keys, start * dimension, count);
    System.arraycopy(value, 0, state.values, start * dimension, count);
    if (batch > 1) {
      dense.causalAttention(
          output, query, state.keys, state.values, batch, start + batch, dimension, heads);
    } else {
      int needed = Math.multiplyExact(heads, start + 1);
      if (scores.length < needed) scores = new float[needed];
      kernels.groupedAttention(
          query,
          0,
          state.keys,
          0,
          state.values,
          0,
          start + 1,
          null,
          0,
          null,
          0,
          0,
          output,
          0,
          scores,
          dimension,
          dimension,
          keyLength,
          valueLength,
          heads,
          kvHeads,
          (float) (1.0 / Math.sqrt(keyLength)));
    }
    state.positions = start + batch;
  }

  @Override
  public void reset() {
    layers.values().forEach(layer -> layer.positions = 0);
  }

  @Override
  public void rewind(int checkpoint) {
    if (checkpoint < 0) throw new IllegalArgumentException("Negative checkpoint");
    layers.values().forEach(layer -> layer.positions = Math.min(layer.positions, checkpoint));
  }

  @Override
  public void close() {
    if (!closed) {
      closed = true;
      layers.clear();
      scores = new float[0];
      dense.close();
    }
  }
}
