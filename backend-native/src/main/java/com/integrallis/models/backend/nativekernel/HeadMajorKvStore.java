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

/**
 * One layer's retained K/V rows in the head-major layout the native attention kernel reads, grown
 * on demand rather than allocated at the session's declared context.
 *
 * <p>Head-major means the per-head stride <em>is</em> the allocated position count, so growing
 * restrides: every head's run moves to a new offset. {@link #stride()} is therefore the number the
 * kernel must receive as its capacity argument, and it is not the declared ceiling.
 *
 * <p>Rows are held as the IEEE binary16 bit patterns the caller rounds them to, two bytes an
 * element rather than four. Widening binary16 to binary32 is exact, so the kernel reads the same
 * operands it read when this store widened them itself, and halving the width halves the bytes a
 * decode step's read of the retained prefix has to touch. The decode path reads the whole prefix
 * once per token, which is what makes that the dominant read.
 *
 * <p>This store holds the rows and nothing else. It deliberately does not copy them in: the caller
 * owns that loop because the values are rounded to F16 on the way, and moving that arithmetic here
 * would put a published numerical contract behind an allocation concern.
 *
 * <p>Growth follows {@link SequenceGrowth}, the one definition the Java caches use. This was the
 * fifth copy of the preallocate-the-ceiling arithmetic that {@code SequenceGrowth}'s own
 * documentation describes: a 32-layer MHA 7B at its declared 4,096 context held 4 GiB here before a
 * sequence reached a tenth of it, and #357 did not reach this file.
 */
final class HeadMajorKvStore {

  private static final short[] EMPTY = new short[0];

  private final int heads;
  private final int width;
  private final int dimension;
  private final int ceiling;
  private short[] keys = EMPTY;
  private short[] values = EMPTY;
  private int stride;
  int positions;

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

  /** The declared context this store will never grow past. */
  int ceiling() {
    return ceiling;
  }

  /** Allocated positions per head: the capacity argument the native kernel must receive. */
  int stride() {
    return stride;
  }

  /** The retained keys as F16 bit patterns, addressed by {@link #stride()}. */
  short[] keys() {
    return keys;
  }

  /** The retained values as F16 bit patterns, addressed by {@link #stride()}. */
  short[] values() {
    return values;
  }

  /** Restore a store from previously serialized F16 arrays with the exact layout. */
  static HeadMajorKvStore restore(int heads, int width, int ceiling, int positions, int stride, short[] keys, short[] values) {
    HeadMajorKvStore store = new HeadMajorKvStore(heads, width, ceiling);
    if (stride < positions) throw new IllegalArgumentException("stride < positions");
    int expectedLength = Math.multiplyExact(stride, store.dimension);
    if (keys.length != expectedLength || values.length != expectedLength)
        throw new IllegalArgumentException("keys/values length mismatch");
    store.stride = stride;
    store.positions = positions;
    store.keys = keys;
    store.values = values;
    return store;
  }

  /** Bytes held by this layer's two buffers. */
  long allocatedBytes() {
    return ((long) keys.length + values.length) * Short.BYTES;
  }

  /**
   * Grows so that every position below {@code required} can be stored, restriding if it grew.
   *
   * @return the stride to address rows by, which the caller must use rather than the ceiling
   */
  int ensureCapacity(int required) {
    int grown = SequenceGrowth.nextCapacity(stride, required, ceiling);
    if (grown == stride) {
      return stride;
    }
    short[] grownKeys = new short[Math.multiplyExact(grown, dimension)];
    short[] grownValues = new short[grownKeys.length];
    int retained = Math.min(positions, stride);
    if (retained > 0) {
      // Head-major: each head's run starts at its own multiple of the stride, so a wider stride
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
    return stride;
  }
}
