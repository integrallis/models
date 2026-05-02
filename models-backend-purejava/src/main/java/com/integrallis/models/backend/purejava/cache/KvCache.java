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

/** Per-layer key-value cache for autoregressive transformer decoding. */
public final class KvCache {

  private final int numLayers;
  private final int maxSeqLen;
  private final int kvDim;
  private final float[][] keys; // [layer * maxSeqLen + position][kvDim]
  private final float[][] values;

  public KvCache(int numLayers, int maxSeqLen, int kvDim) {
    if (numLayers <= 0) throw new IllegalArgumentException("numLayers must be > 0");
    if (maxSeqLen <= 0) throw new IllegalArgumentException("maxSeqLen must be > 0");
    if (kvDim <= 0) throw new IllegalArgumentException("kvDim must be > 0");
    this.numLayers = numLayers;
    this.maxSeqLen = maxSeqLen;
    this.kvDim = kvDim;
    this.keys = new float[numLayers * maxSeqLen][];
    this.values = new float[numLayers * maxSeqLen][];
  }

  /** Stores a key and value vector for a given layer and position. */
  public void store(int layer, int position, float[] key, float[] value) {
    checkBounds(layer, position);
    int idx = layer * maxSeqLen + position;
    keys[idx] = key.clone();
    values[idx] = value.clone();
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
    return keys[layer * maxSeqLen + position];
  }

  /** Returns the value vector for a specific layer and position. */
  public float[] value(int layer, int position) {
    checkBounds(layer, position);
    return values[layer * maxSeqLen + position];
  }

  /** Clears all cached keys and values. */
  public void clear() {
    java.util.Arrays.fill(keys, null);
    java.util.Arrays.fill(values, null);
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

  private float[] slice(float[][] store, int layer, int fromPos, int toPos) {
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
      float[] vec = store[layer * maxSeqLen + fromPos + p];
      if (vec != null) {
        System.arraycopy(vec, 0, result, p * kvDim, kvDim);
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
}
