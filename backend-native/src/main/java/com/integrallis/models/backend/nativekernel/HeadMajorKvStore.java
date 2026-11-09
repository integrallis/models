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

import com.integrallis.models.backend.purejava.cache.SequenceGrowth;
import java.util.Objects;

/**
 * One layer's retained K/V rows in the head-major layout the native attention kernels read, grown
 * on demand rather than allocated at the session's declared context.
 *
 * <p>Head-major means the per-head stride <em>is</em> the allocated position count, so growing
 * restrides: each head's run moves to a new offset. {@link #stride()} is therefore the number the
 * kernels must be given as their capacity argument, and it is not the declared ceiling.
 *
 * <p>Growth follows {@link SequenceGrowth}, the one definition the Java caches use. This store was
 * the fifth copy of the preallocate-the-ceiling arithmetic that {@code SequenceGrowth}'s own
 * documentation describes: a 32-layer MHA 7B at its declared 4,096 context held 4 GiB here before a
 * sequence reached a tenth of it.
 */
final class HeadMajorKvStore {

  private static final float[] EMPTY = new float[0];

  private final int heads;
  private final int width;
  private final int dimension;
  private final int ceiling;
  private float[] keys = EMPTY;
  private float[] values = EMPTY;
  private int stride;
  private int positions;

  HeadMajorKvStore(int heads, int width, int ceiling) {
    if (heads < 1) throw new IllegalArgumentException("heads must be >= 1: " + heads);
    if (width < 1) throw new IllegalArgumentException("width must be >= 1: " + width);
    if (ceiling < 1) throw new IllegalArgumentException("ceiling must be >= 1: " + ceiling);
    this.heads = heads;
    this.width = width;
    this.dimension = Math.multiplyExact(heads, width);
    this.ceiling = ceiling;
  }

  int heads() {
    return heads;
  }

  int dimension() {
    return dimension;
  }

  int ceiling() {
    return ceiling;
  }

  /** Positions currently stored. */
  int positions() {
    return positions;
  }

  /** Allocated positions per head: the capacity argument the native kernels must receive. */
  int stride() {
    return stride;
  }

  float[] keys() {
    return keys;
  }

  float[] values() {
    return values;
  }

  /** Bytes held by this layer's two buffers. */
  long allocatedBytes() {
    return ((long) keys.length + values.length) * Float.BYTES;
  }

  void rewind(int checkpoint) {
    positions = Math.min(positions, checkpoint);
  }

  void reset() {
    positions = 0;
  }

  /**
   * Stores {@code batch} rows beginning at {@code start}, growing first if they do not fit.
   *
   * @param key batch-major key rows, {@code batch * dimension} floats
   * @param value batch-major value rows, the same shape
   */
  void store(float[] key, float[] value, int start, int batch) {
    Objects.requireNonNull(key, "key");
    Objects.requireNonNull(value, "value");
    if (start < 0) throw new IllegalArgumentException("start must be >= 0: " + start);
    if (batch < 1) throw new IllegalArgumentException("batch must be >= 1: " + batch);
    int required = Math.addExact(start, batch);
    int count = Math.multiplyExact(batch, dimension);
    if (key.length < count || value.length < count) {
      throw new IllegalArgumentException("Attention cache input too small");
    }
    grow(required);
    for (int row = 0; row < batch; row++) {
      for (int head = 0; head < heads; head++) {
        int source = row * dimension + head * width;
        int destination = (head * stride + start + row) * width;
        System.arraycopy(key, source, keys, destination, width);
        System.arraycopy(value, source, values, destination, width);
      }
    }
    positions = required;
  }

  private void grow(int required) {
    int grown = SequenceGrowth.nextCapacity(stride, required, ceiling);
    if (grown == stride) {
      return;
    }
    float[] grownKeys = new float[Math.multiplyExact(grown, dimension)];
    float[] grownValues = new float[grownKeys.length];
    int retained = Math.min(positions, stride);
    if (retained > 0) {
      // Head-major: every head's run starts at its own multiple of the stride, so a wider stride
      // moves all of them. One arraycopy per head, and doubling keeps the total linear.
      int run = Math.multiplyExact(retained, width);
      for (int head = 0; head < heads; head++) {
        System.arraycopy(keys, head * stride * width, grownKeys, head * grown * width, run);
        System.arraycopy(values, head * stride * width, grownValues, head * grown * width, run);
      }
    }
    keys = grownKeys;
    values = grownValues;
    stride = grown;
  }
}
