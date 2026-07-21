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
package com.integrallis.models.backend.purejava.cache;

import java.util.Arrays;
import java.util.Objects;

/** Per-layer key-value cache for autoregressive transformer decoding. */
public final class KvCache {

  private static final int INITIAL_SEQUENCE_CAPACITY = 16;

  private final int numLayers;
  private final int maxSeqLen;
  private final int keyDim;
  private final int valueDim;
  private int allocatedSequenceCapacity;
  private float[] keys;
  private float[] values;
  private boolean[] populated;

  public KvCache(int numLayers, int maxSeqLen, int keyDim, int valueDim) {
    if (numLayers <= 0) throw new IllegalArgumentException("numLayers must be > 0");
    if (maxSeqLen <= 0) throw new IllegalArgumentException("maxSeqLen must be > 0");
    if (keyDim <= 0) throw new IllegalArgumentException("keyDim must be > 0");
    if (valueDim <= 0) throw new IllegalArgumentException("valueDim must be > 0");
    this.numLayers = numLayers;
    this.maxSeqLen = maxSeqLen;
    this.keyDim = keyDim;
    this.valueDim = valueDim;
    this.allocatedSequenceCapacity = Math.min(maxSeqLen, INITIAL_SEQUENCE_CAPACITY);
    this.keys = allocateStore("key cache", allocatedSequenceCapacity, keyDim);
    this.values = allocateStore("value cache", allocatedSequenceCapacity, valueDim);
    this.populated = new boolean[checkedSlots(allocatedSequenceCapacity)];
  }

  /** Stores a key and value vector for a given layer and position. */
  public void store(int layer, int position, float[] key, float[] value) {
    checkVector("key", key, keyDim);
    checkVector("value", value, valueDim);
    store(layer, position, key, 0, value, 0);
  }

  /** Stores key and value vectors from offsets in batch-major buffers. */
  public void store(
      int layer, int position, float[] key, int keyOffset, float[] value, int valueOffset) {
    checkBounds(layer, position);
    checkVectorRange("key", key, keyOffset, keyDim);
    checkVectorRange("value", value, valueOffset, valueDim);
    ensureCapacity(position + 1);
    int slot = slotIndex(layer, position);
    System.arraycopy(key, keyOffset, keys, slot * keyDim, keyDim);
    System.arraycopy(value, valueOffset, values, slot * valueDim, valueDim);
    populated[slot] = true;
  }

  /**
   * Returns the concatenated key vectors for a layer from fromPos (inclusive) to toPos (exclusive).
   */
  public float[] keySlice(int layer, int fromPos, int toPos) {
    return slice(keys, keyDim, layer, fromPos, toPos);
  }

  /**
   * Returns the concatenated value vectors for a layer from fromPos (inclusive) to toPos
   * (exclusive).
   */
  public float[] valueSlice(int layer, int fromPos, int toPos) {
    return slice(values, valueDim, layer, fromPos, toPos);
  }

  /** Returns the key vector for a specific layer and position. */
  public float[] key(int layer, int position) {
    checkBounds(layer, position);
    return vectorOrNull(keys, keyDim, layer, position);
  }

  /** Returns the value vector for a specific layer and position. */
  public float[] value(int layer, int position) {
    checkBounds(layer, position);
    return vectorOrNull(values, valueDim, layer, position);
  }

  /** Returns the contiguous key buffer. Callers should use {@link #keyOffset} for addressing. */
  public float[] keyBuffer() {
    return keys;
  }

  /**
   * Returns the contiguous value buffer. Callers should use {@link #valueOffset} for addressing.
   */
  public float[] valueBuffer() {
    return values;
  }

  /** Returns the starting offset of a cached key vector in the contiguous key buffer. */
  public int keyOffset(int layer, int position) {
    checkBounds(layer, position);
    checkAllocated(position);
    return slotIndex(layer, position) * keyDim;
  }

  /** Returns the starting offset of a cached value vector in the contiguous value buffer. */
  public int valueOffset(int layer, int position) {
    checkBounds(layer, position);
    checkAllocated(position);
    return slotIndex(layer, position) * valueDim;
  }

  /** Clears all cached keys and values. */
  public void clear() {
    Arrays.fill(populated, false);
  }

  /** Discards cached keys and values at and after a speculative sequence position. */
  public void discardFrom(int position) {
    if (position < 0 || position > maxSeqLen) {
      throw new IllegalArgumentException("position out of range: " + position);
    }
    if (position >= allocatedSequenceCapacity) {
      return;
    }
    for (int layer = 0; layer < numLayers; layer++) {
      int fromIndex = layer * allocatedSequenceCapacity + position;
      int toIndex = (layer + 1) * allocatedSequenceCapacity;
      Arrays.fill(populated, fromIndex, toIndex, false);
    }
  }

  /** Returns the number of sequence positions currently backed by physical storage. */
  public int allocatedSequenceCapacity() {
    return allocatedSequenceCapacity;
  }

  public int keyDim() {
    return keyDim;
  }

  public int valueDim() {
    return valueDim;
  }

  public int maxSeqLen() {
    return maxSeqLen;
  }

  public int numLayers() {
    return numLayers;
  }

  private float[] slice(float[] store, int dimension, int layer, int fromPos, int toPos) {
    if (layer < 0 || layer >= numLayers) {
      throw new IllegalArgumentException("layer out of range: " + layer);
    }
    if (fromPos < 0 || toPos > maxSeqLen || fromPos > toPos) {
      throw new IllegalArgumentException(
          "invalid position range: [" + fromPos + ", " + toPos + ")");
    }
    int len = toPos - fromPos;
    float[] result = new float[len * dimension];
    for (int p = 0; p < len; p++) {
      int position = fromPos + p;
      if (position < allocatedSequenceCapacity) {
        int slot = slotIndex(layer, position);
        if (!populated[slot]) {
          continue;
        }
        System.arraycopy(store, slot * dimension, result, p * dimension, dimension);
      }
    }
    return result;
  }

  private void checkBounds(int layer, int position) {
    if (layer < 0 || layer >= numLayers) {
      throw new IllegalArgumentException("layer out of range: " + layer);
    }
    if (position < 0 || position >= maxSeqLen) {
      throw new IllegalArgumentException("position out of range: " + position);
    }
  }

  private float[] vectorOrNull(float[] store, int dimension, int layer, int position) {
    if (position >= allocatedSequenceCapacity) {
      return null;
    }
    int slot = slotIndex(layer, position);
    if (!populated[slot]) {
      return null;
    }
    int offset = slot * dimension;
    return Arrays.copyOfRange(store, offset, offset + dimension);
  }

  private static void checkVector(String name, float[] vector, int dimension) {
    Objects.requireNonNull(vector, name);
    if (vector.length != dimension) {
      throw new IllegalArgumentException(
          name + ".length must equal " + dimension + ": " + vector.length);
    }
  }

  private static void checkVectorRange(String name, float[] vector, int offset, int dimension) {
    Objects.requireNonNull(vector, name);
    if (offset < 0 || offset > vector.length - dimension) {
      throw new IllegalArgumentException(
          name
              + " range must fit vector: offset="
              + offset
              + ", dimension="
              + dimension
              + ", length="
              + vector.length);
    }
  }

  private int slotIndex(int layer, int position) {
    return layer * allocatedSequenceCapacity + position;
  }

  private void ensureCapacity(int requiredCapacity) {
    if (requiredCapacity <= allocatedSequenceCapacity) {
      return;
    }

    int newCapacity = allocatedSequenceCapacity;
    while (newCapacity < requiredCapacity) {
      newCapacity = Math.min(maxSeqLen, Math.multiplyExact(newCapacity, 2));
    }

    float[] grownKeys = allocateStore("key cache", newCapacity, keyDim);
    float[] grownValues = allocateStore("value cache", newCapacity, valueDim);
    boolean[] grownPopulated = new boolean[checkedSlots(newCapacity)];
    for (int layer = 0; layer < numLayers; layer++) {
      System.arraycopy(
          keys,
          layer * allocatedSequenceCapacity * keyDim,
          grownKeys,
          layer * newCapacity * keyDim,
          allocatedSequenceCapacity * keyDim);
      System.arraycopy(
          values,
          layer * allocatedSequenceCapacity * valueDim,
          grownValues,
          layer * newCapacity * valueDim,
          allocatedSequenceCapacity * valueDim);
      System.arraycopy(
          populated,
          layer * allocatedSequenceCapacity,
          grownPopulated,
          layer * newCapacity,
          allocatedSequenceCapacity);
    }

    allocatedSequenceCapacity = newCapacity;
    keys = grownKeys;
    values = grownValues;
    populated = grownPopulated;
  }

  private void checkAllocated(int position) {
    if (position >= allocatedSequenceCapacity) {
      throw new IllegalStateException("position has no allocated storage: " + position);
    }
  }

  private float[] allocateStore(String name, int sequenceCapacity, int dimension) {
    return new float[checkedProduct(name, checkedSlots(sequenceCapacity), dimension)];
  }

  private int checkedSlots(int sequenceCapacity) {
    return checkedProduct("numLayers * sequenceCapacity", numLayers, sequenceCapacity);
  }

  private static int checkedProduct(String name, int a, int b) {
    long product = (long) a * b;
    if (product > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(name + " is too large: " + a + " * " + b);
    }
    return (int) product;
  }
}
