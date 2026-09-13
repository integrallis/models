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

import com.integrallis.vectors.core.VectorUtil;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/** Per-layer key-value cache for autoregressive transformer decoding. */
public final class KvCache {

  private static final int INITIAL_SEQUENCE_CAPACITY = 16;

  private final int numLayers;
  private final int maxSeqLen;
  private final int keyDim;
  private final int valueDim;
  private final SharedPrefix sharedPrefix;
  private final int localStartPosition;
  private int allocatedSequenceCapacity;
  private float[] keys;
  private float[] values;
  private boolean[] populated;
  private boolean frozen;

  /** One contiguous chronological run in either shared-prefix or branch-owned storage. */
  public static final class AttentionSpan {
    private final int firstPosition;
    private final int positionCount;
    private final float[] keyBuffer;
    private final int keyOffset;
    private final float[] valueBuffer;
    private final int valueOffset;

    private AttentionSpan(
        int firstPosition,
        int positionCount,
        float[] keyBuffer,
        int keyOffset,
        float[] valueBuffer,
        int valueOffset) {
      if (firstPosition < 0) {
        throw new IllegalArgumentException("firstPosition must be >= 0: " + firstPosition);
      }
      if (positionCount <= 0) {
        throw new IllegalArgumentException("positionCount must be > 0: " + positionCount);
      }
      Objects.requireNonNull(keyBuffer, "keyBuffer");
      Objects.requireNonNull(valueBuffer, "valueBuffer");
      if (keyOffset < 0 || valueOffset < 0) {
        throw new IllegalArgumentException("attention offsets must be >= 0");
      }
      this.firstPosition = firstPosition;
      this.positionCount = positionCount;
      this.keyBuffer = keyBuffer;
      this.keyOffset = keyOffset;
      this.valueBuffer = valueBuffer;
      this.valueOffset = valueOffset;
    }

    public int firstPosition() {
      return firstPosition;
    }

    public int positionCount() {
      return positionCount;
    }
  }

  /** A chronological attention range represented without copying by one or more spans. */
  public static final class AttentionView {
    private final List<AttentionSpan> spans;

    private AttentionView(List<AttentionSpan> spans) {
      Objects.requireNonNull(spans, "spans");
      if (spans.isEmpty()) {
        throw new IllegalArgumentException("attention view must contain at least one span");
      }
      this.spans = List.copyOf(spans);
    }

    public int positionCount() {
      int positions = 0;
      for (AttentionSpan span : spans) {
        positions = Math.addExact(positions, span.positionCount());
      }
      return positions;
    }

    public int spanCount() {
      return spans.size();
    }

    public AttentionSpan span(int index) {
      if (index < 0 || index >= spans.size()) {
        throw new IllegalArgumentException("span index out of range: " + index);
      }
      return spans.get(index);
    }
  }

  /** One immutable array-backed interval in a persistent shared-prefix chain. */
  private static final class StorageSegment {
    private final int firstPosition;
    private final int positionCount;
    private final int storageCapacity;
    private final float[] keys;
    private final float[] values;
    private final boolean[] populated;

    private StorageSegment(
        int firstPosition,
        int positionCount,
        int storageCapacity,
        float[] keys,
        float[] values,
        boolean[] populated) {
      this.firstPosition = firstPosition;
      this.positionCount = positionCount;
      this.storageCapacity = storageCapacity;
      this.keys = keys;
      this.values = values;
      this.populated = populated;
    }

    private int endPosition() {
      return Math.addExact(firstPosition, positionCount);
    }

    private boolean contains(int position) {
      return position >= firstPosition && position < endPosition();
    }
  }

  /**
   * Immutable KV storage that can be referenced by multiple divergent sequence branches.
   *
   * <p>{@link #fork()} allocates storage only for positions after this prefix. The key and value
   * arrays holding the prefix are never copied.
   */
  public static final class SharedPrefix {
    private final int numLayers;
    private final int maxSeqLen;
    private final int keyDim;
    private final int valueDim;
    private final int length;
    private final List<StorageSegment> segments;

    private SharedPrefix(KvCache source, int length) {
      this.numLayers = source.numLayers;
      this.maxSeqLen = source.maxSeqLen;
      this.keyDim = source.keyDim;
      this.valueDim = source.valueDim;
      this.length = length;
      List<StorageSegment> collected = new ArrayList<>();
      if (source.sharedPrefix != null) {
        for (StorageSegment segment : source.sharedPrefix.segments) {
          if (segment.firstPosition >= length) {
            break;
          }
          int included = Math.min(segment.endPosition(), length) - segment.firstPosition;
          collected.add(
              new StorageSegment(
                  segment.firstPosition,
                  included,
                  segment.storageCapacity,
                  segment.keys,
                  segment.values,
                  segment.populated));
        }
      }
      if (length > source.localStartPosition) {
        collected.add(
            new StorageSegment(
                source.localStartPosition,
                length - source.localStartPosition,
                source.allocatedSequenceCapacity,
                source.keys,
                source.values,
                source.populated));
      }
      this.segments = List.copyOf(collected);
    }

    /** Returns the immutable prefix length in tokens. */
    public int length() {
      return length;
    }

    /** Opens an independent suffix that references this exact prefix storage. */
    public KvCache fork() {
      return new KvCache(this);
    }

    /** Returns the key/value and occupancy bytes held by the shared physical storage. */
    public long allocatedBytes() {
      Set<Object> counted = Collections.newSetFromMap(new IdentityHashMap<>());
      return allocatedBytes(counted);
    }

    private long allocatedBytes(Set<Object> counted) {
      long bytes = 0;
      for (StorageSegment segment : segments) {
        if (counted.add(segment.keys)) {
          bytes = Math.addExact(bytes, Math.multiplyExact((long) segment.keys.length, Float.BYTES));
        }
        if (counted.add(segment.values)) {
          bytes =
              Math.addExact(bytes, Math.multiplyExact((long) segment.values.length, Float.BYTES));
        }
        if (counted.add(segment.populated)) {
          bytes = Math.addExact(bytes, segment.populated.length);
        }
      }
      return bytes;
    }
  }

  public KvCache(int numLayers, int maxSeqLen, int keyDim, int valueDim) {
    if (numLayers <= 0) throw new IllegalArgumentException("numLayers must be > 0");
    if (maxSeqLen <= 0) throw new IllegalArgumentException("maxSeqLen must be > 0");
    if (keyDim <= 0) throw new IllegalArgumentException("keyDim must be > 0");
    if (valueDim <= 0) throw new IllegalArgumentException("valueDim must be > 0");
    this.numLayers = numLayers;
    this.maxSeqLen = maxSeqLen;
    this.keyDim = keyDim;
    this.valueDim = valueDim;
    this.sharedPrefix = null;
    this.localStartPosition = 0;
    this.allocatedSequenceCapacity = Math.min(maxSeqLen, INITIAL_SEQUENCE_CAPACITY);
    this.keys = allocateStore("key cache", allocatedSequenceCapacity, keyDim);
    this.values = allocateStore("value cache", allocatedSequenceCapacity, valueDim);
    this.populated = new boolean[checkedSlots(allocatedSequenceCapacity)];
  }

  private KvCache(SharedPrefix prefix) {
    this.numLayers = prefix.numLayers;
    this.maxSeqLen = prefix.maxSeqLen;
    this.keyDim = prefix.keyDim;
    this.valueDim = prefix.valueDim;
    this.sharedPrefix = prefix;
    this.localStartPosition = prefix.length;
    int remainingPositions = maxSeqLen - localStartPosition;
    this.allocatedSequenceCapacity = Math.min(remainingPositions, INITIAL_SEQUENCE_CAPACITY);
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
    requireMutable();
    checkBounds(layer, position);
    if (position < localStartPosition) {
      throw new IllegalArgumentException(
          "position "
              + position
              + " belongs to immutable shared prefix [0, "
              + localStartPosition
              + ")");
    }
    checkVectorRange("key", key, keyOffset, keyDim);
    checkVectorRange("value", value, valueOffset, valueDim);
    ensureCapacity(position + 1);
    int slot = localSlotIndex(layer, position);
    if (HALF_PRECISION) {
      storeHalfPrecision(key, keyOffset, keys, slot * keyDim, keyDim);
      storeHalfPrecision(value, valueOffset, values, slot * valueDim, valueDim);
    } else {
      System.arraycopy(key, keyOffset, keys, slot * keyDim, keyDim);
      System.arraycopy(value, valueOffset, values, slot * valueDim, valueDim);
    }
    populated[slot] = true;
  }

  /**
   * Whether to hold cached keys and values at half precision, off by default.
   *
   * <p>llama.cpp holds {@code cache_k} and {@code cache_v} as F16, so attention there reads back
   * rounded keys and values. Enabling this reproduces its layer output exactly on Qwen3 and raises
   * embedding agreement, but it flips a greedy token on the pinned SQLCoder Q5_K_M fixture, which
   * is itself a llama.cpp equivalence reference. The two references disagree, so this cannot be the
   * default until that is resolved; see the qwen3 K-quant divergence investigation.
   */
  private static final boolean HALF_PRECISION =
      Boolean.getBoolean("models.purejava.halfPrecisionKvCache");

  /**
   * Copies a vector into the cache at half precision.
   *
   * <p>llama.cpp holds {@code cache_k} and {@code cache_v} as F16, so attention there reads back
   * rounded keys and values. Storing F32 made this runtime more precise than the reference rather
   * than equivalent to it, which the embedding equivalence gate reports as a failure.
   */
  private static void storeHalfPrecision(
      float[] source, int sourceOffset, float[] destination, int destinationOffset, int length) {
    for (int index = 0; index < length; index++) {
      destination[destinationOffset + index] =
          Float.float16ToFloat(Float.floatToFloat16(source[sourceOffset + index]));
    }
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
    if (isSharedPosition(position)) {
      StorageSegment segment = sharedSegment(position);
      return vectorOrNull(
          segment.keys,
          keyDim,
          layer,
          position,
          segment.storageCapacity,
          segment.firstPosition,
          segment.populated);
    }
    return vectorOrNull(keys, keyDim, layer, position);
  }

  /** Returns the value vector for a specific layer and position. */
  public float[] value(int layer, int position) {
    checkBounds(layer, position);
    if (isSharedPosition(position)) {
      StorageSegment segment = sharedSegment(position);
      return vectorOrNull(
          segment.values,
          valueDim,
          layer,
          position,
          segment.storageCapacity,
          segment.firstPosition,
          segment.populated);
    }
    return vectorOrNull(values, valueDim, layer, position);
  }

  /**
   * Freezes a complete prefix in place and returns a handle that can create zero-copy forks.
   * Subsequent mutation of this source cache is rejected.
   */
  public SharedPrefix freezePrefix(int prefixLength) {
    requireMutable();
    if (prefixLength <= 0 || prefixLength > maxSeqLen) {
      throw new IllegalArgumentException(
          "prefixLength must be > 0 and <= " + maxSeqLen + ": " + prefixLength);
    }
    for (int layer = 0; layer < numLayers; layer++) {
      for (int position = 0; position < prefixLength; position++) {
        requirePopulated(layer, position);
      }
    }
    frozen = true;
    return new SharedPrefix(this, prefixLength);
  }

  /** Returns a zero-copy view over a populated half-open attention range. */
  public AttentionView attentionView(int layer, int fromPosition, int toPosition) {
    if (layer < 0 || layer >= numLayers) {
      throw new IllegalArgumentException("layer out of range: " + layer);
    }
    if (fromPosition < 0 || toPosition > maxSeqLen || fromPosition >= toPosition) {
      throw new IllegalArgumentException(
          "invalid position range: [" + fromPosition + ", " + toPosition + ")");
    }
    for (int position = fromPosition; position < toPosition; position++) {
      requirePopulated(layer, position);
    }

    List<AttentionSpan> spans = new ArrayList<>();
    if (sharedPrefix != null) {
      for (StorageSegment segment : sharedPrefix.segments) {
        int segmentStart = Math.max(fromPosition, segment.firstPosition);
        int segmentEnd = Math.min(toPosition, segment.endPosition());
        if (segmentStart < segmentEnd) {
          spans.add(
              span(
                  layer,
                  segmentStart,
                  segmentEnd - segmentStart,
                  segment.keys,
                  segment.values,
                  segment.storageCapacity,
                  segment.firstPosition));
        }
      }
    }
    int localStart = Math.max(fromPosition, localStartPosition);
    if (localStart < toPosition) {
      spans.add(
          span(
              layer,
              localStart,
              toPosition - localStart,
              keys,
              values,
              allocatedSequenceCapacity,
              localStartPosition));
    }
    return new AttentionView(spans);
  }

  /** Computes scaled query/key scores across shared and branch-owned spans. */
  public void writeAttentionScores(
      int layer,
      int fromPosition,
      int toPosition,
      int keyHeadOffset,
      float[] query,
      int queryOffset,
      int vectorLength,
      float scale,
      float[] scores,
      int scoresOffset,
      boolean batched) {
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(scores, "scores");
    Objects.checkFromIndexSize(queryOffset, vectorLength, query.length);
    checkHeadRange("key", keyHeadOffset, vectorLength, keyDim);
    Objects.checkFromIndexSize(
        scoresOffset + fromPosition, toPosition - fromPosition, scores.length);
    AttentionView view = attentionView(layer, fromPosition, toPosition);
    for (int spanIndex = 0; spanIndex < view.spanCount(); spanIndex++) {
      AttentionSpan span = view.span(spanIndex);
      if (batched) {
        VectorUtil.batchDotProductExact(
            query,
            queryOffset,
            span.keyBuffer,
            span.keyOffset + keyHeadOffset,
            keyDim,
            span.positionCount,
            vectorLength,
            scores,
            scoresOffset + span.firstPosition);
        for (int row = 0; row < span.positionCount; row++) {
          scores[scoresOffset + span.firstPosition + row] *= scale;
        }
      } else {
        for (int row = 0; row < span.positionCount; row++) {
          int keyOffset = span.keyOffset + row * keyDim + keyHeadOffset;
          scores[scoresOffset + span.firstPosition + row] =
              VectorUtil.dotProduct(query, queryOffset, span.keyBuffer, keyOffset, vectorLength)
                  * scale;
        }
      }
    }
  }

  /** Adds score-weighted cached values across shared and branch-owned spans. */
  public void addAttentionValues(
      int layer,
      int fromPosition,
      int toPosition,
      int valueHeadOffset,
      float[] output,
      int outputOffset,
      int vectorLength,
      float[] scores,
      int scoresOffset,
      boolean batched) {
    Objects.requireNonNull(output, "output");
    Objects.requireNonNull(scores, "scores");
    Objects.checkFromIndexSize(outputOffset, vectorLength, output.length);
    checkHeadRange("value", valueHeadOffset, vectorLength, valueDim);
    Objects.checkFromIndexSize(
        scoresOffset + fromPosition, toPosition - fromPosition, scores.length);
    AttentionView view = attentionView(layer, fromPosition, toPosition);
    for (int spanIndex = 0; spanIndex < view.spanCount(); spanIndex++) {
      AttentionSpan span = view.span(spanIndex);
      if (batched) {
        VectorUtil.addWeightedRowsInPlace(
            output,
            outputOffset,
            span.valueBuffer,
            span.valueOffset + valueHeadOffset,
            valueDim,
            scores,
            scoresOffset + span.firstPosition,
            span.positionCount,
            vectorLength);
      } else {
        for (int row = 0; row < span.positionCount; row++) {
          VectorUtil.addScaledInPlace(
              output,
              outputOffset,
              span.valueBuffer,
              span.valueOffset + row * valueDim + valueHeadOffset,
              vectorLength,
              scores[scoresOffset + span.firstPosition + row]);
        }
      }
    }
  }

  /** Returns the number of physical bytes shared by this branch's immutable prefix. */
  public long sharedPrefixBytes() {
    return sharedPrefix == null ? 0L : sharedPrefix.allocatedBytes();
  }

  /** Returns all currently allocated key/value and occupancy storage reachable by this cache. */
  public long allocatedBytes() {
    Set<Object> counted = Collections.newSetFromMap(new IdentityHashMap<>());
    long bytes = sharedPrefix == null ? 0L : sharedPrefix.allocatedBytes(counted);
    if (counted.add(keys)) {
      bytes = Math.addExact(bytes, Math.multiplyExact((long) keys.length, Float.BYTES));
    }
    if (counted.add(values)) {
      bytes = Math.addExact(bytes, Math.multiplyExact((long) values.length, Float.BYTES));
    }
    if (counted.add(populated)) {
      bytes = Math.addExact(bytes, populated.length);
    }
    return bytes;
  }

  /** Returns whether two branches reference the exact same immutable prefix arrays. */
  public boolean sharesPrefixStorageWith(KvCache other) {
    Objects.requireNonNull(other, "other");
    return sharedPrefix != null && sharedPrefix == other.sharedPrefix;
  }

  /** Returns the immutable shared prefix length, or zero for an ordinary cache. */
  public int sharedPrefixLength() {
    return localStartPosition;
  }

  /** Returns a caller-owned contiguous-storage snapshot for diagnostics and tests. */
  public float[] keyBuffer() {
    requireContiguousOwnedStorage();
    return keys.clone();
  }

  /** Returns a caller-owned contiguous-storage snapshot for diagnostics and tests. */
  public float[] valueBuffer() {
    requireContiguousOwnedStorage();
    return values.clone();
  }

  /** Returns the starting offset of a cached key vector in the contiguous key buffer. */
  public int keyOffset(int layer, int position) {
    requireContiguousOwnedStorage();
    checkBounds(layer, position);
    checkAllocated(position);
    return localSlotIndex(layer, position) * keyDim;
  }

  /** Returns the starting offset of a cached value vector in the contiguous value buffer. */
  public int valueOffset(int layer, int position) {
    requireContiguousOwnedStorage();
    checkBounds(layer, position);
    checkAllocated(position);
    return localSlotIndex(layer, position) * valueDim;
  }

  /** Clears all cached keys and values. */
  public void clear() {
    requireMutable();
    Arrays.fill(populated, false);
  }

  /** Discards cached keys and values at and after a speculative sequence position. */
  public void discardFrom(int position) {
    requireMutable();
    if (position < 0 || position > maxSeqLen) {
      throw new IllegalArgumentException("position out of range: " + position);
    }
    if (position < localStartPosition) {
      throw new IllegalArgumentException(
          "cannot discard immutable shared prefix before position " + localStartPosition);
    }
    int localPosition = position - localStartPosition;
    if (localPosition >= allocatedSequenceCapacity) {
      return;
    }
    for (int layer = 0; layer < numLayers; layer++) {
      int fromIndex = layer * allocatedSequenceCapacity + localPosition;
      int toIndex = (layer + 1) * allocatedSequenceCapacity;
      Arrays.fill(populated, fromIndex, toIndex, false);
    }
  }

  /** Returns the number of sequence positions currently backed by physical storage. */
  public int allocatedSequenceCapacity() {
    return Math.addExact(localStartPosition, allocatedSequenceCapacity);
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

  /** Ensures physical storage exists for the requested number of sequence positions. */
  public void reserveSequenceCapacity(int requiredSequenceCapacity) {
    requireMutable();
    if (requiredSequenceCapacity < 0 || requiredSequenceCapacity > maxSeqLen) {
      throw new IllegalArgumentException(
          "requiredSequenceCapacity must be between 0 and "
              + maxSeqLen
              + ": "
              + requiredSequenceCapacity);
    }
    ensureCapacity(requiredSequenceCapacity);
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
      float[] vector =
          dimension == keyDim && store == keys ? key(layer, position) : value(layer, position);
      if (vector == null) throw unpopulatedPosition(layer, position);
      System.arraycopy(vector, 0, result, p * dimension, dimension);
    }
    return result;
  }

  private static IllegalStateException unpopulatedPosition(int layer, int position) {
    return new IllegalStateException(
        "cache position " + position + " for layer " + layer + " is not populated");
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
    return vectorOrNull(
        store,
        dimension,
        layer,
        position,
        allocatedSequenceCapacity,
        localStartPosition,
        populated);
  }

  private static float[] vectorOrNull(
      float[] store,
      int dimension,
      int layer,
      int position,
      int sequenceCapacity,
      int startPosition,
      boolean[] occupancy) {
    int localPosition = position - startPosition;
    if (localPosition < 0 || localPosition >= sequenceCapacity) {
      return null;
    }
    int slot = layer * sequenceCapacity + localPosition;
    if (!occupancy[slot]) {
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

  private int localSlotIndex(int layer, int position) {
    return layer * allocatedSequenceCapacity + position - localStartPosition;
  }

  private void ensureCapacity(int requiredCapacity) {
    int requiredLocalCapacity = Math.max(0, requiredCapacity - localStartPosition);
    if (requiredLocalCapacity <= allocatedSequenceCapacity) {
      return;
    }

    int newCapacity = allocatedSequenceCapacity;
    int localLimit = maxSeqLen - localStartPosition;
    if (newCapacity == 0) newCapacity = Math.min(localLimit, INITIAL_SEQUENCE_CAPACITY);
    while (newCapacity < requiredLocalCapacity) {
      newCapacity = Math.min(localLimit, Math.multiplyExact(newCapacity, 2));
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
    if (position < localStartPosition
        || position - localStartPosition >= allocatedSequenceCapacity) {
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

  private static void checkHeadRange(String name, int offset, int length, int dimension) {
    if (offset < 0 || length <= 0 || offset > dimension - length) {
      throw new IllegalArgumentException(
          name
              + " head range must fit dimension: offset="
              + offset
              + ", length="
              + length
              + ", dimension="
              + dimension);
    }
  }

  private boolean isSharedPosition(int position) {
    return sharedPrefix != null && position < localStartPosition;
  }

  private void requirePopulated(int layer, int position) {
    if (isSharedPosition(position)) {
      StorageSegment segment = sharedSegment(position);
      int slot = layer * segment.storageCapacity + position - segment.firstPosition;
      if (!segment.populated[slot]) throw unpopulatedPosition(layer, position);
      return;
    }
    int localPosition = position - localStartPosition;
    if (localPosition < 0
        || localPosition >= allocatedSequenceCapacity
        || !populated[layer * allocatedSequenceCapacity + localPosition]) {
      throw unpopulatedPosition(layer, position);
    }
  }

  private AttentionSpan span(
      int layer,
      int firstPosition,
      int positionCount,
      float[] keyStore,
      float[] valueStore,
      int storageCapacity,
      int storageStartPosition) {
    int firstLocalPosition = firstPosition - storageStartPosition;
    return new AttentionSpan(
        firstPosition,
        positionCount,
        keyStore,
        (layer * storageCapacity + firstLocalPosition) * keyDim,
        valueStore,
        (layer * storageCapacity + firstLocalPosition) * valueDim);
  }

  private StorageSegment sharedSegment(int position) {
    for (StorageSegment segment : sharedPrefix.segments) {
      if (segment.contains(position)) {
        return segment;
      }
    }
    throw new IllegalStateException("shared prefix has no storage for position " + position);
  }

  private void requireMutable() {
    if (frozen) {
      throw new IllegalStateException("KV cache is frozen as immutable shared prefix storage");
    }
  }

  private void requireContiguousOwnedStorage() {
    if (sharedPrefix != null) {
      throw new IllegalStateException(
          "forked KV cache has segmented storage; use attentionView() instead");
    }
  }
}
