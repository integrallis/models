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

  private final int numLayers;
  private final int maxSeqLen;
  private final int kvDim;
  private final float[] keys; // [(layer * maxSeqLen + position) * kvDim + d]
  private final float[] values;
  private final boolean[] populated;

  public KvCache(int numLayers, int maxSeqLen, int kvDim) {
    if (numLayers <= 0) throw new IllegalArgumentException("numLayers must be > 0");
    if (maxSeqLen <= 0) throw new IllegalArgumentException("maxSeqLen must be > 0");
    if (kvDim <= 0) throw new IllegalArgumentException("kvDim must be > 0");
    this.numLayers = numLayers;
    this.maxSeqLen = maxSeqLen;
    this.kvDim = kvDim;
    int slots = checkedProduct("numLayers * maxSeqLen", numLayers, maxSeqLen);
    int elements = checkedProduct("numLayers * maxSeqLen * kvDim", slots, kvDim);
    this.keys = new float[elements];
    this.values = new float[elements];
    this.populated = new boolean[slots];
  }

  /** Stores a key and value vector for a given layer and position. */
  public void store(int layer, int position, float[] key, float[] value) {
    checkBounds(layer, position);
    checkVector("key", key);
    checkVector("value", value);
    int slot = slotIndex(layer, position);
    int offset = slot * kvDim;
    System.arraycopy(key, 0, keys, offset, kvDim);
    System.arraycopy(value, 0, values, offset, kvDim);
    populated[slot] = true;
  }

  /**
   * Returns the concatenated key vectors for a layer from fromPos (inclusive) to toPos (exclusive).
   */
  public float[] keySlice(int layer, int fromPos, int toPos) {
    return slice(keys, layer, fromPos, toPos);
  }

  /**
   * Returns the concatenated value vectors for a layer from fromPos (inclusive) to toPos
   * (exclusive).
   */
  public float[] valueSlice(int layer, int fromPos, int toPos) {
    return slice(values, layer, fromPos, toPos);
  }

  /** Returns the key vector for a specific layer and position. */
  public float[] key(int layer, int position) {
    checkBounds(layer, position);
    return vectorOrNull(keys, layer, position);
  }

  /** Returns the value vector for a specific layer and position. */
  public float[] value(int layer, int position) {
    checkBounds(layer, position);
    return vectorOrNull(values, layer, position);
  }

  /** Returns the contiguous key buffer. Callers should use {@link #vectorOffset} for addressing. */
  public float[] keyBuffer() {
    return keys;
  }

  /**
   * Returns the contiguous value buffer. Callers should use {@link #vectorOffset} for addressing.
   */
  public float[] valueBuffer() {
    return values;
  }

  /** Returns the starting offset of a cached key/value vector in the contiguous buffers. */
  public int vectorOffset(int layer, int position) {
    checkBounds(layer, position);
    return slotIndex(layer, position) * kvDim;
  }

  /** Clears all cached keys and values. */
  public void clear() {
    Arrays.fill(keys, 0.0f);
    Arrays.fill(values, 0.0f);
    Arrays.fill(populated, false);
  }

  public int kvDim() {
    return kvDim;
  }

  public int maxSeqLen() {
    return maxSeqLen;
  }

  public int numLayers() {
    return numLayers;
  }

  private float[] slice(float[] store, int layer, int fromPos, int toPos) {
    if (layer < 0 || layer >= numLayers) {
      throw new IllegalArgumentException("layer out of range: " + layer);
    }
    if (fromPos < 0 || toPos > maxSeqLen || fromPos > toPos) {
      throw new IllegalArgumentException(
          "invalid position range: [" + fromPos + ", " + toPos + ")");
    }
    int len = toPos - fromPos;
    float[] result = new float[len * kvDim];
    for (int p = 0; p < len; p++) {
      int slot = slotIndex(layer, fromPos + p);
      if (populated[slot]) {
        System.arraycopy(store, slot * kvDim, result, p * kvDim, kvDim);
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

  private float[] vectorOrNull(float[] store, int layer, int position) {
    int slot = slotIndex(layer, position);
    if (!populated[slot]) {
      return null;
    }
    int offset = slot * kvDim;
    return Arrays.copyOfRange(store, offset, offset + kvDim);
  }

  private void checkVector(String name, float[] vector) {
    Objects.requireNonNull(vector, name);
    if (vector.length != kvDim) {
      throw new IllegalArgumentException(name + ".length must equal kvDim: " + vector.length);
    }
  }

  private int slotIndex(int layer, int position) {
    return layer * maxSeqLen + position;
  }

  private static int checkedProduct(String name, int a, int b) {
    long product = (long) a * b;
    if (product > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(name + " is too large: " + a + " * " + b);
    }
    return (int) product;
  }
}
