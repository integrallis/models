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
  private final RustGgufBatchedMatrixKernel kernels;
  private final NativeDebertaKernels dense;
  private float[] scores = new float[0];
  private final Map<Integer, HeadMajorKvStore> layers = new HashMap<>();
  private boolean closed;

  NativeCausalAttention(RustGgufBatchedMatrixKernel kernels, NativeDebertaKernels dense) {
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
    HeadMajorKvStore state = layers.get(layer);
    return state == null
        ? start == 0
        : state.positions() == start
            && state.dimension() == dimension
            && state.heads() == heads
            && state.ceiling() == capacity;
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
    HeadMajorKvStore state = storeRows(key, value, layer, start, batch, heads, keyLength, capacity);
    if (batch > 1) {
      dense.causalAttention(
          output,
          query,
          state.keys(),
          state.values(),
          batch,
          start + batch,
          dimension,
          heads,
          state.stride());
    } else {
      int needed = Math.multiplyExact(heads, start + 1);
      if (scores.length < needed) scores = new float[needed];
      kernels.headMajorAttention(
          query,
          state.keys(),
          state.values(),
          output,
          scores,
          heads,
          keyLength,
          state.stride(),
          start + 1,
          (float) (1.0 / Math.sqrt(keyLength)));
    }
  }

  @Override
  public boolean append(
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
      return false;
    int count = Math.multiplyExact(batch, Math.multiplyExact(heads, keyLength));
    if (key.length < count || value.length < count)
      throw new IllegalArgumentException("Attention cache input too small");
    storeRows(key, value, layer, start, batch, heads, keyLength, capacity);
    return true;
  }

  private HeadMajorKvStore storeRows(
      float[] key,
      float[] value,
      int layer,
      int start,
      int batch,
      int heads,
      int width,
      int capacity) {
    HeadMajorKvStore state =
        layers.computeIfAbsent(layer, ignored -> new HeadMajorKvStore(heads, width, capacity));
    state.store(key, value, start, batch);
    return state;
  }

  @Override
  public long allocatedStateBytes() {
    long bytes = 0L;
    for (HeadMajorKvStore layer : layers.values()) {
      bytes = Math.addExact(bytes, layer.allocatedBytes());
    }
    return bytes;
  }

  @Override
  public void reset() {
    layers.values().forEach(HeadMajorKvStore::reset);
  }

  @Override
  public void rewind(int checkpoint) {
    if (checkpoint < 0) throw new IllegalArgumentException("Negative checkpoint");
    layers.values().forEach(layer -> layer.rewind(checkpoint));
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
