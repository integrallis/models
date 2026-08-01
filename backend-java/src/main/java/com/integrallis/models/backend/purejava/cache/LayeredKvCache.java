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
import java.util.OptionalInt;

/**
 * Per-layer KV storage supporting different head widths and linear or bounded-ring retention.
 *
 * <p>The cache is mutable session state and is not thread-safe. Attention views describe one or two
 * contiguous spans in the owned buffers without copying cached vectors.
 */
public final class LayeredKvCache {

  private static final int INITIAL_LINEAR_CAPACITY = 16;
  private static final float[] EMPTY_FLOATS = new float[0];
  private static final int[] EMPTY_POSITIONS = new int[0];

  /** Physical retention policy for one decoder layer. */
  public enum Retention {
    LINEAR,
    RING
  }

  /** Immutable dimensions and retention policy for one layer. */
  public record LayerSpec(int keyDim, int valueDim, Retention retention, int ringCapacity) {

    public LayerSpec {
      positive("keyDim", keyDim);
      positive("valueDim", valueDim);
      Objects.requireNonNull(retention, "retention");
      if (retention == Retention.LINEAR && ringCapacity != 0) {
        throw new IllegalArgumentException("linear layer ringCapacity must be 0: " + ringCapacity);
      }
      if (retention == Retention.RING && ringCapacity <= 0) {
        throw new IllegalArgumentException("ring layer ringCapacity must be > 0: " + ringCapacity);
      }
    }

    /** Creates a layer that retains absolute history up to the logical context limit. */
    public static LayerSpec linear(int keyDim, int valueDim) {
      return new LayerSpec(keyDim, valueDim, Retention.LINEAR, 0);
    }

    /** Creates a layer backed by a fixed number of reusable sequence slots. */
    public static LayerSpec ring(int keyDim, int valueDim, int physicalCapacity) {
      return new LayerSpec(keyDim, valueDim, Retention.RING, physicalCapacity);
    }
  }

  /** One chronological run of positions stored contiguously in the key and value buffers. */
  public record AttentionSpan(
      int firstPosition, int positionCount, int keyOffset, int valueOffset) {

    public AttentionSpan {
      if (firstPosition < 0) {
        throw new IllegalArgumentException("firstPosition must be >= 0: " + firstPosition);
      }
      positive("positionCount", positionCount);
      if (keyOffset < 0 || valueOffset < 0) {
        throw new IllegalArgumentException("attention offsets must be >= 0");
      }
    }
  }

  /** A chronological attention range represented by one linear span or two ring spans. */
  public static final class AttentionView {
    private final AttentionSpan first;
    private final AttentionSpan second;
    private final int positionCount;

    private AttentionView(AttentionSpan first, AttentionSpan second) {
      this.first = Objects.requireNonNull(first, "first");
      this.second = second;
      this.positionCount =
          second == null
              ? first.positionCount()
              : Math.addExact(first.positionCount(), second.positionCount());
    }

    /** Returns the total number of chronological positions in this view. */
    public int positionCount() {
      return positionCount;
    }

    /** Returns one for a linear range and two when a ring range crosses its physical end. */
    public int spanCount() {
      return second == null ? 1 : 2;
    }

    /** Returns a contiguous span by chronological index. */
    public AttentionSpan span(int index) {
      if (index == 0) {
        return first;
      }
      if (index == 1 && second != null) {
        return second;
      }
      throw new IllegalArgumentException("span index out of range: " + index);
    }
  }

  private static final class LayerState {
    private final LayerSpec spec;
    private final int physicalLimit;
    private int allocatedCapacity;
    private float[] keys = EMPTY_FLOATS;
    private float[] values = EMPTY_FLOATS;
    private int[] positions = EMPTY_POSITIONS;

    private LayerState(LayerSpec spec, int maxSeqLen) {
      this.spec = spec;
      this.physicalLimit =
          spec.retention() == Retention.RING ? Math.min(spec.ringCapacity(), maxSeqLen) : maxSeqLen;
    }
  }

  private final int maxSeqLen;
  private final LayerState[] layers;

  /** Creates lazy physical storage for the supplied per-layer layouts. */
  public LayeredKvCache(int maxSeqLen, LayerSpec... layerSpecs) {
    positive("maxSeqLen", maxSeqLen);
    Objects.requireNonNull(layerSpecs, "layerSpecs");
    if (layerSpecs.length == 0) {
      throw new IllegalArgumentException("layerSpecs must not be empty");
    }
    this.maxSeqLen = maxSeqLen;
    this.layers = new LayerState[layerSpecs.length];
    for (int layer = 0; layer < layerSpecs.length; layer++) {
      this.layers[layer] =
          new LayerState(
              Objects.requireNonNull(layerSpecs[layer], "layerSpecs[" + layer + "]"), maxSeqLen);
    }
  }

  /** Stores complete key and value vectors at an absolute sequence position. */
  public void store(int layer, int position, float[] key, float[] value) {
    store(layer, position, key, 0, value, 0);
  }

  /** Stores key and value vectors from offsets in batch-major source arrays. */
  public void store(
      int layer, int position, float[] key, int keyOffset, float[] value, int valueOffset) {
    LayerState state = requireCoordinates(layer, position);
    checkVectorRange("key", key, keyOffset, state.spec.keyDim());
    checkVectorRange("value", value, valueOffset, state.spec.valueDim());
    ensureCapacity(state, position + 1);
    int slot = slot(state, position);
    System.arraycopy(key, keyOffset, state.keys, slot * state.spec.keyDim(), state.spec.keyDim());
    System.arraycopy(
        value, valueOffset, state.values, slot * state.spec.valueDim(), state.spec.valueDim());
    state.positions[slot] = position;
  }

  /** Returns whether the absolute position remains available for the layer. */
  public boolean contains(int layer, int position) {
    LayerState state = requireCoordinates(layer, position);
    return state.allocatedCapacity != 0 && state.positions[slot(state, position)] == position;
  }

  /** Returns a zero-copy chronological view over an available half-open attention range. */
  public AttentionView attentionView(int layer, int fromPosition, int toPosition) {
    LayerState state = requireLayer(layer);
    if (fromPosition < 0 || toPosition > maxSeqLen || fromPosition >= toPosition) {
      throw new IllegalArgumentException(
          "invalid attention range: [" + fromPosition + ", " + toPosition + ")");
    }
    int positionCount = toPosition - fromPosition;
    if (state.spec.retention() == Retention.RING && positionCount > state.physicalLimit) {
      throw new IllegalArgumentException(
          "attention range length "
              + positionCount
              + " exceeds ring capacity "
              + state.physicalLimit);
    }
    for (int position = fromPosition; position < toPosition; position++) {
      if (state.allocatedCapacity == 0 || state.positions[slot(state, position)] != position) {
        throw new IllegalStateException(
            "cache position " + position + " for layer " + layer + " is overwritten or absent");
      }
    }

    int firstSlot = slot(state, fromPosition);
    if (state.spec.retention() == Retention.LINEAR) {
      return new AttentionView(span(state, fromPosition, positionCount, firstSlot), null);
    }
    int firstCount = Math.min(positionCount, state.allocatedCapacity - firstSlot);
    AttentionSpan first = span(state, fromPosition, firstCount, firstSlot);
    int secondCount = positionCount - firstCount;
    AttentionSpan second =
        secondCount == 0 ? null : span(state, fromPosition + firstCount, secondCount, 0);
    return new AttentionView(first, second);
  }

  /** Discards speculative entries at and after the supplied absolute position. */
  public void discardFrom(int position) {
    if (position < 0 || position > maxSeqLen) {
      throw new IllegalArgumentException("position out of range: " + position);
    }
    for (LayerState state : layers) {
      for (int slot = 0; slot < state.positions.length; slot++) {
        if (state.positions[slot] >= position) {
          state.positions[slot] = -1;
        }
      }
    }
  }

  /** Clears every retained position without reallocating physical storage. */
  public void clear() {
    for (LayerState state : layers) {
      Arrays.fill(state.positions, -1);
    }
  }

  /** Returns the oldest absolute position still available in a layer. */
  public OptionalInt oldestRetainedPosition(int layer) {
    return retainedPosition(layer, true);
  }

  /** Returns the newest absolute position still available in a layer. */
  public OptionalInt newestRetainedPosition(int layer) {
    return retainedPosition(layer, false);
  }

  /** Returns the layer's key width. */
  public int keyDim(int layer) {
    return requireLayer(layer).spec.keyDim();
  }

  /** Returns the layer's value width. */
  public int valueDim(int layer) {
    return requireLayer(layer).spec.valueDim();
  }

  /** Returns the layer's currently allocated sequence capacity. */
  public int allocatedSequenceCapacity(int layer) {
    return requireLayer(layer).allocatedCapacity;
  }

  /** Returns the layer's maximum physical sequence capacity. */
  public int physicalSequenceCapacity(int layer) {
    return requireLayer(layer).physicalLimit;
  }

  /** Returns the mutable key buffer owned by a layer. */
  public float[] keyBuffer(int layer) {
    return requireLayer(layer).keys;
  }

  /** Returns the mutable value buffer owned by a layer. */
  public float[] valueBuffer(int layer) {
    return requireLayer(layer).values;
  }

  /** Returns the key-vector offset for an available absolute position. */
  public int keyOffset(int layer, int position) {
    LayerState state = requireAvailable(layer, position);
    return Math.multiplyExact(slot(state, position), state.spec.keyDim());
  }

  /** Returns the value-vector offset for an available absolute position. */
  public int valueOffset(int layer, int position) {
    LayerState state = requireAvailable(layer, position);
    return Math.multiplyExact(slot(state, position), state.spec.valueDim());
  }

  /** Returns the bytes currently owned by all key, value, and position arrays. */
  public long allocatedBytes() {
    long bytes = 0;
    for (LayerState state : layers) {
      bytes = Math.addExact(bytes, storageBytes(state.spec, state.allocatedCapacity));
    }
    return bytes;
  }

  /** Returns the maximum bytes required when every layer reaches its physical capacity. */
  public long maximumBytes() {
    long bytes = 0;
    for (LayerState state : layers) {
      bytes = Math.addExact(bytes, storageBytes(state.spec, state.physicalLimit));
    }
    return bytes;
  }

  public int numLayers() {
    return layers.length;
  }

  public int maxSeqLen() {
    return maxSeqLen;
  }

  private OptionalInt retainedPosition(int layer, boolean oldest) {
    LayerState state = requireLayer(layer);
    int result = oldest ? Integer.MAX_VALUE : -1;
    for (int position : state.positions) {
      if (position >= 0) {
        result = oldest ? Math.min(result, position) : Math.max(result, position);
      }
    }
    return result == Integer.MAX_VALUE || result < 0 ? OptionalInt.empty() : OptionalInt.of(result);
  }

  private LayerState requireAvailable(int layer, int position) {
    LayerState state = requireCoordinates(layer, position);
    if (state.allocatedCapacity == 0 || state.positions[slot(state, position)] != position) {
      throw new IllegalStateException(
          "cache position " + position + " for layer " + layer + " is overwritten or absent");
    }
    return state;
  }

  private LayerState requireCoordinates(int layer, int position) {
    LayerState state = requireLayer(layer);
    if (position < 0 || position >= maxSeqLen) {
      throw new IllegalArgumentException("position out of range: " + position);
    }
    return state;
  }

  private LayerState requireLayer(int layer) {
    if (layer < 0 || layer >= layers.length) {
      throw new IllegalArgumentException("layer out of range: " + layer);
    }
    return layers[layer];
  }

  private static AttentionSpan span(
      LayerState state, int firstPosition, int positionCount, int firstSlot) {
    return new AttentionSpan(
        firstPosition,
        positionCount,
        Math.multiplyExact(firstSlot, state.spec.keyDim()),
        Math.multiplyExact(firstSlot, state.spec.valueDim()));
  }

  private static int slot(LayerState state, int position) {
    return state.spec.retention() == Retention.RING ? position % state.physicalLimit : position;
  }

  private void ensureCapacity(LayerState state, int requiredCapacity) {
    if (state.spec.retention() == Retention.RING) {
      if (state.allocatedCapacity == 0) {
        allocate(state, state.physicalLimit);
      }
      return;
    }
    if (requiredCapacity <= state.allocatedCapacity) {
      return;
    }

    int newCapacity = state.allocatedCapacity;
    if (newCapacity == 0) {
      newCapacity = Math.min(maxSeqLen, INITIAL_LINEAR_CAPACITY);
    }
    while (newCapacity < requiredCapacity) {
      int doubled = newCapacity <= maxSeqLen / 2 ? newCapacity * 2 : maxSeqLen;
      newCapacity = Math.min(maxSeqLen, doubled);
    }
    grow(state, newCapacity);
  }

  private static void allocate(LayerState state, int capacity) {
    state.keys = new float[checkedProduct("key storage", capacity, state.spec.keyDim())];
    state.values = new float[checkedProduct("value storage", capacity, state.spec.valueDim())];
    state.positions = new int[capacity];
    Arrays.fill(state.positions, -1);
    state.allocatedCapacity = capacity;
  }

  private static void grow(LayerState state, int capacity) {
    float[] oldKeys = state.keys;
    float[] oldValues = state.values;
    int[] oldPositions = state.positions;
    allocate(state, capacity);
    System.arraycopy(oldKeys, 0, state.keys, 0, oldKeys.length);
    System.arraycopy(oldValues, 0, state.values, 0, oldValues.length);
    System.arraycopy(oldPositions, 0, state.positions, 0, oldPositions.length);
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

  private static int checkedProduct(String name, int left, int right) {
    long product = (long) left * right;
    if (product > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(name + " is too large: " + left + " * " + right);
    }
    return (int) product;
  }

  private static long storageBytes(LayerSpec spec, int capacity) {
    long vectorWidth = Math.addExact((long) spec.keyDim(), spec.valueDim());
    long vectorBytes =
        Math.multiplyExact(Math.multiplyExact((long) capacity, vectorWidth), Float.BYTES);
    long positionBytes = Math.multiplyExact((long) capacity, Integer.BYTES);
    return Math.addExact(vectorBytes, positionBytes);
  }

  private static void positive(String name, int value) {
    if (value <= 0) {
      throw new IllegalArgumentException(name + " must be > 0: " + value);
    }
  }
}
