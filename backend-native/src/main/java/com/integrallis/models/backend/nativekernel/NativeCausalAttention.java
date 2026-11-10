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
  private final NativeDebertaKernels dense;
  private final Map<Integer, HeadMajorKvStore> layers = new HashMap<>();
  private boolean closed;

  NativeCausalAttention(NativeDebertaKernels dense) {
    this.dense = dense;
  }

  @Override
  public boolean usesFloatRotaryRecurrence() {
    return true;
  }

  @Override
  public boolean rotaryFactors(
      int position,
      int pairs,
      float base,
      float scale,
      float magnitude,
      float[] divisors,
      float[] cosine,
      float[] sine,
      int offset) {
    if (closed) throw new IllegalStateException("Native attention is closed");
    return dense.rotaryFactors(
        position, pairs, base, scale, magnitude, divisors, cosine, sine, offset);
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
        : state.positions == start
            && state.dimension() == dimension
            && state.heads() == heads
            // The ceiling rather than the array length: the store holds only the positions a
            // sequence has reached, so its length is not capacity * dimension until the context
            // fills. What must still not change under a retained prefix is the declared capacity.
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
    attendWithBatch(
        output,
        query,
        key,
        value,
        layer,
        start,
        batch,
        heads,
        kvHeads,
        keyLength,
        valueLength,
        capacity,
        window,
        batch);
  }

  @Override
  public void attendFinalPrefill(
      float[] output,
      float[] query,
      float[] key,
      float[] value,
      int layer,
      int position,
      int originalBatchSize,
      int heads,
      int kvHeads,
      int keyLength,
      int valueLength,
      int capacity,
      int window) {
    attendWithBatch(
        output,
        query,
        key,
        value,
        layer,
        position,
        1,
        heads,
        kvHeads,
        keyLength,
        valueLength,
        capacity,
        window,
        originalBatchSize);
  }

  private void attendWithBatch(
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
      int window,
      int originalBatchSize) {
    if (originalBatchSize < batch || originalBatchSize > start + batch)
      throw new IllegalArgumentException("Invalid original attention batch size");
    if (!isEligible(layer, start, batch, heads, kvHeads, keyLength, valueLength, capacity, window))
      throw new IllegalArgumentException("Discontinuous or unsupported native attention");
    int dimension = Math.multiplyExact(heads, keyLength);
    int count = Math.multiplyExact(batch, dimension);
    if (query.length < count || key.length < count || value.length < count || output.length < count)
      throw new IllegalArgumentException("Attention buffer too small");
    if (output == query || output == key || output == value)
      throw new IllegalArgumentException("Attention output must not alias inputs");
    HeadMajorKvStore state = storeRows(key, value, layer, start, batch, heads, keyLength, capacity);
    dense.visionAttention(
        output,
        query,
        state.keys(),
        state.values(),
        batch,
        start + batch,
        dimension,
        heads,
        // The allocated stride, not the ceiling: visionAttention uses this as the per-head stride
        // and requires the buffers to be stride*dimension, which is what the store now holds.
        state.stride(),
        originalBatchSize);
    state.positions = start + batch;
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
    HeadMajorKvStore state = storeRows(key, value, layer, start, batch, heads, keyLength, capacity);
    state.positions = start + batch;
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
    // Grown before addressing, and rows are addressed by the allocated stride rather than the
    // ceiling. The rounding below is unchanged: same operand, same order, same per-element
    // rounding, so the stored values are bit-identical to the preallocating form.
    int stride = state.ensureCapacity(Math.addExact(start, batch));
    for (int row = 0; row < batch; row++) {
      for (int head = 0; head < heads; head++) {
        int source = row * state.dimension() + head * width;
        int destination = (head * stride + start + row) * width;
        // Flash Attention consumes F16 operands even when the decoder KV setting is F32.
        // Round each appended row once and retain the F16 bits, so native attention borrows the
        // Java-owned prefix directly instead of copying and rounding it at every decode step.
        // Storing the bits rather than widening them back keeps the same values in half the bytes:
        // the kernel's widening load is exact, so every operand it reads is the one it read when
        // this loop widened them here.
        for (int element = 0; element < width; element++) {
          state.keys()[destination + element] = Float.floatToFloat16(key[source + element]);
          state.values()[destination + element] = Float.floatToFloat16(value[source + element]);
        }
      }
    }
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
      dense.close();
    }
  }
}
