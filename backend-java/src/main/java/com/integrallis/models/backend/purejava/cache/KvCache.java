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

import com.integrallis.models.backend.purejava.ops.GroupedQueryAttentionKernel;
import com.integrallis.vectors.core.VectorUtil;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Per-layer key-value cache for autoregressive transformer decoding.
 *
 * <p>Token storage is allocated on the first reservation or store. Creating an unused session or
 * shared-prefix branch allocates no token buffers.
 */
public final class KvCache {

  private static final float[] EMPTY_FLOATS = new float[0];
  private static final boolean[] EMPTY_OCCUPANCY = new boolean[0];

  // ---- Compact F16 storage abstraction (#408, #447) ----
  private interface KvStorage {
    void store(int slot, float[] src, int srcOffset, int length);

    float[] read(int slot, int length);

    /** Reads {@code length} elements from {@code slot} into {@code dst} at {@code dstOffset}. */
    void readInto(int slot, float[] dst, int dstOffset, int length);

    void copyTo(int fromSlot, KvStorage dst, int dstSlot, int slotCount, int length);

    void writeTo(DataOutput out, int offset, int count, byte[] scratch) throws IOException;

    void readFrom(DataInput in, int offset, int count, byte[] scratch) throws IOException;

    KvStorage grow(int oldSlots, int newSlots, int length);

    int capacity();

    /** Returns the underlying float[] for zero-copy views, or null if compact F16. */
    float[] rawFloatArray();
  }

  private static final class F32Store implements KvStorage {
    private float[] data;

    F32Store(int capacity) {
      this.data = new float[capacity];
    }

    @Override
    public void store(int slot, float[] src, int srcOffset, int length) {
      System.arraycopy(src, srcOffset, data, slot * length, length);
    }

    @Override
    public float[] read(int slot, int length) {
      return Arrays.copyOfRange(data, slot * length, (slot + 1) * length);
    }

    @Override
    public void readInto(int slot, float[] dst, int dstOffset, int length) {
      System.arraycopy(data, slot * length, dst, dstOffset, length);
    }

    @Override
    public void copyTo(int fromSlot, KvStorage dst, int dstSlot, int slotCount, int length) {
      F32Store d = (F32Store) dst;
      System.arraycopy(data, fromSlot * length, d.data, dstSlot * length, slotCount * length);
    }

    @Override
    public void writeTo(DataOutput out, int offset, int count, byte[] scratch) throws IOException {
      writeFloats(out, data, offset, count, scratch);
    }

    @Override
    public void readFrom(DataInput in, int offset, int count, byte[] scratch) throws IOException {
      readFloats(in, data, offset, count, scratch);
    }

    @Override
    public KvStorage grow(int oldSlots, int newSlots, int length) {
      float[] grown = new float[newSlots * length];
      System.arraycopy(data, 0, grown, 0, oldSlots * length);
      data = grown;
      return this;
    }

    @Override
    public int capacity() {
      return data.length;
    }

    @Override
    public float[] rawFloatArray() {
      return data;
    }
  }

  private static final class F16Store implements KvStorage {
    private short[] data;

    F16Store(int capacity) {
      this.data = new short[capacity];
    }

    @Override
    public void store(int slot, float[] src, int srcOffset, int length) {
      int base = slot * length;
      for (int i = 0; i < length; i++) data[base + i] = Float.floatToFloat16(src[srcOffset + i]);
    }

    @Override
    public float[] read(int slot, int length) {
      float[] result = new float[length];
      int base = slot * length;
      for (int i = 0; i < length; i++) result[i] = Float.float16ToFloat(data[base + i]);
      return result;
    }

    @Override
    public void readInto(int slot, float[] dst, int dstOffset, int length) {
      int base = slot * length;
      for (int i = 0; i < length; i++) {
        dst[dstOffset + i] = Float.float16ToFloat(data[base + i]);
      }
    }

    @Override
    public void copyTo(int fromSlot, KvStorage dst, int dstSlot, int slotCount, int length) {
      F16Store d = (F16Store) dst;
      System.arraycopy(data, fromSlot * length, d.data, dstSlot * length, slotCount * length);
    }

    @Override
    public void writeTo(DataOutput out, int offset, int count, byte[] scratch) throws IOException {
      int batch = scratch.length / Float.BYTES;
      float[] buffer = new float[batch];
      int remaining = count, pos = offset;
      while (remaining > 0) {
        int n = Math.min(batch, remaining);
        for (int i = 0; i < n; i++) buffer[i] = Float.float16ToFloat(data[pos + i]);
        writeFloats(out, buffer, 0, n, scratch);
        pos += n;
        remaining -= n;
      }
    }

    @Override
    public void readFrom(DataInput in, int offset, int count, byte[] scratch) throws IOException {
      int batch = scratch.length / Float.BYTES;
      float[] buffer = new float[batch];
      int remaining = count, pos = offset;
      while (remaining > 0) {
        int n = Math.min(batch, remaining);
        readFloats(in, buffer, 0, n, scratch);
        for (int i = 0; i < n; i++) data[pos + i] = Float.floatToFloat16(buffer[i]);
        pos += n;
        remaining -= n;
      }
    }

    @Override
    public KvStorage grow(int oldSlots, int newSlots, int length) {
      short[] grown = new short[newSlots * length];
      System.arraycopy(data, 0, grown, 0, oldSlots * length);
      data = grown;
      return this;
    }

    @Override
    public int capacity() {
      return data.length;
    }

    @Override
    public float[] rawFloatArray() {
      return null;
    }
  }

  private static KvStorage allocateStore(boolean compact, int slots, int length) {
    return compact ? new F16Store(slots * length) : new F32Store(slots * length);
  }

  private static KvStorage emptyStore() {
    return new F32Store(0);
  }

  // ---- end compact F16 storage ----

  private final int numLayers;
  private final int maxSeqLen;
  private final int keyDim;
  private final int valueDim;
  private final SharedPrefix sharedPrefix;
  private final int localStartPosition;
  private final boolean compact;
  private int allocatedSequenceCapacity;
  private KvStorage keys = emptyStore();
  private KvStorage values = emptyStore();
  private boolean[] populated = EMPTY_OCCUPANCY;
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

    /** Backing key rows; row {@code r} of this span starts at {@code keyOffset() + r * keyDim}. */
    public float[] keyBuffer() {
      return keyBuffer;
    }

    public int keyOffset() {
      return keyOffset;
    }

    /** Backing value rows; row {@code r} starts at {@code valueOffset() + r * valueDim}. */
    public float[] valueBuffer() {
      return valueBuffer;
    }

    public int valueOffset() {
      return valueOffset;
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
    private final KvStorage keys;
    private final KvStorage values;
    private final boolean[] populated;

    private StorageSegment(
        int firstPosition,
        int positionCount,
        int storageCapacity,
        KvStorage keys,
        KvStorage values,
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
    private final boolean compact;
    private final List<StorageSegment> segments;

    private SharedPrefix(KvCache source, int length) {
      this.numLayers = source.numLayers;
      this.maxSeqLen = source.maxSeqLen;
      this.keyDim = source.keyDim;
      this.valueDim = source.valueDim;
      this.length = length;
      this.compact = source.compact;
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

    /** Returns whether the underlying storage uses compact F16 instead of F32. */
    public boolean compact() {
      return compact;
    }

    /** Returns a shorter immutable view without copying tensor storage. */
    public SharedPrefix truncate(int tokenCount) {
      if (tokenCount <= 0 || tokenCount > length) {
        throw new IllegalArgumentException("invalid prefix length: " + tokenCount);
      }
      return tokenCount == length ? this : new SharedPrefix(fork(), tokenCount);
    }

    /** Streams canonical big-endian FP32 rows, retaining raw float bits and token positions. */
    public void writeTo(DataOutput output, int tokenCount) throws IOException {
      SharedPrefix prefix = truncate(tokenCount);
      output.writeInt(0x4b563031); // KV01: layer-major K rows, then V rows, for each layer.
      output.writeInt(numLayers);
      output.writeInt(tokenCount);
      output.writeInt(keyDim);
      output.writeInt(valueDim);
      byte[] scratch = new byte[64 * 1024];
      for (int layer = 0; layer < numLayers; layer++) {
        for (StorageSegment segment : prefix.segments) {
          segment.keys.writeTo(
              output,
              layer * segment.storageCapacity * keyDim,
              segment.positionCount * keyDim,
              scratch);
        }
        for (StorageSegment segment : prefix.segments) {
          segment.values.writeTo(
              output,
              layer * segment.storageCapacity * valueDim,
              segment.positionCount * valueDim,
              scratch);
        }
      }
    }

    /**
     * Streams only the positions {@code [startToken, endToken)} for every layer as raw big-endian
     * FP32. The payload is intended to follow a higher-level block descriptor (identity, start,
     * end, payload length) in the wire format.
     */
    public void writeBlockTo(DataOutput output, int startToken, int endToken) throws IOException {
      if (startToken < 0 || endToken > length || startToken >= endToken) {
        throw new IllegalArgumentException(
            "invalid block range [" + startToken + ", " + endToken + ") for length " + length);
      }
      byte[] scratch = new byte[64 * 1024];
      for (int layer = 0; layer < numLayers; layer++) {
        for (StorageSegment segment : segments) {
          int overlapStart = Math.max(startToken, segment.firstPosition);
          int overlapEnd = Math.min(endToken, segment.endPosition());
          if (overlapStart < overlapEnd) {
            int keyOffset =
                layer * segment.storageCapacity * keyDim
                    + (overlapStart - segment.firstPosition) * keyDim;
            int keyCount = (overlapEnd - overlapStart) * keyDim;
            segment.keys.writeTo(output, keyOffset, keyCount, scratch);
          }
        }
        for (StorageSegment segment : segments) {
          int overlapStart = Math.max(startToken, segment.firstPosition);
          int overlapEnd = Math.min(endToken, segment.endPosition());
          if (overlapStart < overlapEnd) {
            int valueOffset =
                layer * segment.storageCapacity * valueDim
                    + (overlapStart - segment.firstPosition) * valueDim;
            int valueCount = (overlapEnd - overlapStart) * valueDim;
            segment.values.writeTo(output, valueOffset, valueCount, scratch);
          }
        }
      }
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
          bytes =
              Math.addExact(
                  bytes,
                  (long) segment.keys.capacity()
                      * (segment.keys instanceof F16Store ? Short.BYTES : Float.BYTES));
        }
        if (counted.add(segment.values)) {
          bytes =
              Math.addExact(
                  bytes,
                  (long) segment.values.capacity()
                      * (segment.values instanceof F16Store ? Short.BYTES : Float.BYTES));
        }
        if (counted.add(segment.populated)) {
          bytes = Math.addExact(bytes, segment.populated.length);
        }
      }
      return bytes;
    }
  }

  public KvCache(int numLayers, int maxSeqLen, int keyDim, int valueDim) {
    this(numLayers, maxSeqLen, keyDim, valueDim, HALF_PRECISION);
  }

  /** Package-visible constructor for compact-mode testing and cross-backend parity. */
  KvCache(int numLayers, int maxSeqLen, int keyDim, int valueDim, boolean compact) {
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
    this.compact = compact;
    validateInitialStorageSize(Math.min(maxSeqLen, SequenceGrowth.INITIAL_CAPACITY));
  }

  private KvCache(SharedPrefix prefix) {
    this.numLayers = prefix.numLayers;
    this.maxSeqLen = prefix.maxSeqLen;
    this.keyDim = prefix.keyDim;
    this.valueDim = prefix.valueDim;
    this.sharedPrefix = prefix;
    this.localStartPosition = prefix.length;
    this.compact = prefix.compact();
    int remainingPositions = maxSeqLen - localStartPosition;
    validateInitialStorageSize(Math.min(remainingPositions, SequenceGrowth.INITIAL_CAPACITY));
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
    keys.store(slot, key, keyOffset, keyDim);
    values.store(slot, value, valueOffset, valueDim);
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

  /**
   * Reads a detached immutable prefix. Dimensions come from the receiving model, never the peer. No
   * existing session is mutated, including when validation or input fails.
   */
  public static SharedPrefix readPrefix(
      DataInput input,
      int layers,
      int context,
      int keyWidth,
      int valueWidth,
      int tokenCount,
      long maxBytes)
      throws IOException {
    if (layers <= 0
        || keyWidth <= 0
        || valueWidth <= 0
        || tokenCount <= 0
        || tokenCount > context) {
      throw new IOException("invalid KV dimensions or context length");
    }
    long keyElements;
    long valueElements;
    long bytes;
    try {
      keyElements = Math.multiplyExact(Math.multiplyExact((long) layers, tokenCount), keyWidth);
      valueElements = Math.multiplyExact(Math.multiplyExact((long) layers, tokenCount), valueWidth);
      bytes = Math.addExact(20L, Math.multiplyExact(4L, Math.addExact(keyElements, valueElements)));
    } catch (ArithmeticException invalid) {
      throw new IOException("KV size overflow", invalid);
    }
    if (bytes > maxBytes
        || keyElements > Integer.MAX_VALUE - 8
        || valueElements > Integer.MAX_VALUE - 8) {
      throw new IOException("KV prefix exceeds import budget");
    }
    if (input.readInt() != 0x4b563031
        || input.readInt() != layers
        || input.readInt() != tokenCount
        || input.readInt() != keyWidth
        || input.readInt() != valueWidth) {
      throw new IOException("incompatible KV format or dimensions");
    }
    var cache = new KvCache(layers, context, keyWidth, valueWidth);
    cache.allocatedSequenceCapacity = tokenCount;
    int slots = Math.multiplyExact(layers, tokenCount);
    cache.keys = allocateStore(HALF_PRECISION, slots, keyWidth);
    cache.values = allocateStore(HALF_PRECISION, slots, valueWidth);
    cache.populated = new boolean[Math.multiplyExact(layers, tokenCount)];
    byte[] scratch = new byte[64 * 1024];
    for (int layer = 0; layer < layers; layer++) {
      cache.keys.readFrom(input, layer * tokenCount * keyWidth, tokenCount * keyWidth, scratch);
      cache.values.readFrom(
          input, layer * tokenCount * valueWidth, tokenCount * valueWidth, scratch);
    }
    Arrays.fill(cache.populated, true);
    return cache.freezePrefix(tokenCount);
  }

  /**
   * Creates a mutable cache pre-allocated for exactly {@code tokenCount} positions, ready to
   * receive block payloads via {@link #readBlock}. Callers must populate every position and then
   * invoke {@link #freezePrefix}.
   */
  public static KvCache createForAssembly(
      int layers, int context, int keyWidth, int valueWidth, int tokenCount) {
    var cache = new KvCache(layers, context, keyWidth, valueWidth);
    cache.allocatedSequenceCapacity = tokenCount;
    int slots = Math.multiplyExact(layers, tokenCount);
    cache.keys = allocateStore(HALF_PRECISION, slots, keyWidth);
    cache.values = allocateStore(HALF_PRECISION, slots, valueWidth);
    cache.populated = new boolean[Math.multiplyExact(layers, tokenCount)];
    return cache;
  }

  /**
   * Reads one block payload from {@code input} and copies it into positions {@code [startToken,
   * endToken)}. The caller must have created this cache via {@link #createForAssembly}.
   */
  public void readBlock(int startToken, int endToken, DataInput input) throws IOException {
    if (startToken < 0 || endToken > allocatedSequenceCapacity || startToken >= endToken) {
      throw new IllegalArgumentException(
          "invalid block range [" + startToken + ", " + endToken + ")");
    }
    int count = endToken - startToken;
    byte[] scratch = new byte[64 * 1024];
    for (int layer = 0; layer < numLayers; layer++) {
      keys.readFrom(
          input,
          layer * allocatedSequenceCapacity * keyDim + startToken * keyDim,
          count * keyDim,
          scratch);
      values.readFrom(
          input,
          layer * allocatedSequenceCapacity * valueDim + startToken * valueDim,
          count * valueDim,
          scratch);
    }
    for (int layer = 0; layer < numLayers; layer++) {
      for (int i = startToken; i < endToken; i++) {
        populated[layer * allocatedSequenceCapacity + i] = true;
      }
    }
  }

  public static void writeFloats(
      DataOutput output, float[] values, int offset, int count, byte[] scratch) throws IOException {
    var buffer = ByteBuffer.wrap(scratch);
    while (count > 0) {
      int n = Math.min(count, scratch.length / Float.BYTES);
      buffer.clear();
      for (int i = 0; i < n; i++) buffer.putInt(Float.floatToRawIntBits(values[offset + i]));
      output.write(scratch, 0, n * Float.BYTES);
      offset += n;
      count -= n;
    }
  }

  public static void readFloats(
      DataInput input, float[] values, int offset, int count, byte[] scratch) throws IOException {
    var buffer = ByteBuffer.wrap(scratch);
    while (count > 0) {
      int n = Math.min(count, scratch.length / Float.BYTES);
      input.readFully(scratch, 0, n * Float.BYTES);
      buffer.clear();
      for (int i = 0; i < n; i++) values[offset + i] = Float.intBitsToFloat(buffer.getInt());
      offset += n;
      count -= n;
    }
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
    writeAttentionScores(
        attentionView(layer, fromPosition, toPosition),
        fromPosition,
        toPosition,
        keyHeadOffset,
        query,
        queryOffset,
        vectorLength,
        scale,
        scores,
        scoresOffset,
        batched);
  }

  /**
   * Scores against an {@link AttentionView} the caller resolved once for this layer and position
   * range, so a row's heads share one population check and span resolution.
   */
  public void writeAttentionScores(
      AttentionView view,
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
    Objects.requireNonNull(view, "view");
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(scores, "scores");
    Objects.checkFromIndexSize(queryOffset, vectorLength, query.length);
    checkHeadRange("key", keyHeadOffset, vectorLength, keyDim);
    Objects.checkFromIndexSize(
        scoresOffset + fromPosition, toPosition - fromPosition, scores.length);
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

  /**
   * Scores every query head of one grouped-query group against the cached keys of its KV head,
   * reading each key row once. Head {@code h} of the group writes {@code scores[scoresOffset + h *
   * scoresHeadStride + position]}. Arithmetic matches the per-head exact scoring kernel.
   */
  public void writeGroupedAttentionScores(
      AttentionView view,
      int fromPosition,
      int toPosition,
      int keyHeadOffset,
      float[] query,
      int queryOffset,
      int queryHeadStride,
      int groupSize,
      int vectorLength,
      float scale,
      float[] scores,
      int scoresOffset,
      int scoresHeadStride) {
    Objects.requireNonNull(view, "view");
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(scores, "scores");
    checkHeadRange("key", keyHeadOffset, vectorLength, keyDim);
    Objects.checkFromIndexSize(
        scoresOffset + (groupSize - 1) * scoresHeadStride + fromPosition,
        toPosition - fromPosition,
        scores.length);
    for (int spanIndex = 0; spanIndex < view.spanCount(); spanIndex++) {
      AttentionSpan span = view.span(spanIndex);
      GroupedQueryAttentionKernel.scoreGroup(
          query,
          queryOffset,
          queryHeadStride,
          groupSize,
          span.keyBuffer,
          span.keyOffset + keyHeadOffset,
          keyDim,
          span.positionCount,
          vectorLength,
          scale,
          scores,
          scoresOffset + span.firstPosition,
          scoresHeadStride);
    }
  }

  /**
   * Accumulates score-weighted cached values for every query head of one group, reading each value
   * row once. Arithmetic matches the per-head weighted-rows kernel.
   */
  public void addGroupedAttentionValues(
      AttentionView view,
      int fromPosition,
      int toPosition,
      int valueHeadOffset,
      float[] output,
      int outputOffset,
      int outputHeadStride,
      int groupSize,
      int vectorLength,
      float[] scores,
      int scoresOffset,
      int scoresHeadStride) {
    Objects.requireNonNull(view, "view");
    Objects.requireNonNull(output, "output");
    Objects.requireNonNull(scores, "scores");
    checkHeadRange("value", valueHeadOffset, vectorLength, valueDim);
    Objects.checkFromIndexSize(
        scoresOffset + (groupSize - 1) * scoresHeadStride + fromPosition,
        toPosition - fromPosition,
        scores.length);
    for (int spanIndex = 0; spanIndex < view.spanCount(); spanIndex++) {
      AttentionSpan span = view.span(spanIndex);
      GroupedQueryAttentionKernel.accumulateGroup(
          output,
          outputOffset,
          outputHeadStride,
          groupSize,
          span.valueBuffer,
          span.valueOffset + valueHeadOffset,
          valueDim,
          span.positionCount,
          vectorLength,
          scores,
          scoresOffset + span.firstPosition,
          scoresHeadStride);
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
    addAttentionValues(
        attentionView(layer, fromPosition, toPosition),
        fromPosition,
        toPosition,
        valueHeadOffset,
        output,
        outputOffset,
        vectorLength,
        scores,
        scoresOffset,
        batched);
  }

  /** Value accumulation against a caller-resolved {@link AttentionView}. */
  public void addAttentionValues(
      AttentionView view,
      int fromPosition,
      int toPosition,
      int valueHeadOffset,
      float[] output,
      int outputOffset,
      int vectorLength,
      float[] scores,
      int scoresOffset,
      boolean batched) {
    Objects.requireNonNull(view, "view");
    Objects.requireNonNull(output, "output");
    Objects.requireNonNull(scores, "scores");
    Objects.checkFromIndexSize(outputOffset, vectorLength, output.length);
    checkHeadRange("value", valueHeadOffset, vectorLength, valueDim);
    Objects.checkFromIndexSize(
        scoresOffset + fromPosition, toPosition - fromPosition, scores.length);
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
      bytes =
          Math.addExact(
              bytes,
              (long) keys.capacity() * (keys instanceof F16Store ? Short.BYTES : Float.BYTES));
    }
    if (counted.add(values)) {
      bytes =
          Math.addExact(
              bytes,
              (long) values.capacity() * (values instanceof F16Store ? Short.BYTES : Float.BYTES));
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
    int slots = numLayers * allocatedSequenceCapacity;
    float[] out = new float[slots * keyDim];
    for (int i = 0; i < slots; i++) {
      keys.readInto(i, out, i * keyDim, keyDim);
    }
    return out;
  }

  /** Returns a caller-owned contiguous-storage snapshot for diagnostics and tests. */
  public float[] valueBuffer() {
    requireContiguousOwnedStorage();
    int slots = numLayers * allocatedSequenceCapacity;
    float[] out = new float[slots * valueDim];
    for (int i = 0; i < slots; i++) {
      values.readInto(i, out, i * valueDim, valueDim);
    }
    return out;
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

  private float[] slice(KvStorage store, int dimension, int layer, int fromPos, int toPos) {
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

  private float[] vectorOrNull(KvStorage store, int dimension, int layer, int position) {
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
      KvStorage store,
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
    return store.read(slot, dimension);
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

    int localLimit = maxSeqLen - localStartPosition;
    int newCapacity =
        SequenceGrowth.nextCapacity(allocatedSequenceCapacity, requiredLocalCapacity, localLimit);

    KvStorage grownKeys = allocateStore(compact, checkedSlots(newCapacity), keyDim);
    KvStorage grownValues = allocateStore(compact, checkedSlots(newCapacity), valueDim);
    boolean[] grownPopulated = new boolean[checkedSlots(newCapacity)];
    if (allocatedSequenceCapacity > 0) {
      for (int layer = 0; layer < numLayers; layer++) {
        keys.copyTo(
            layer * allocatedSequenceCapacity,
            grownKeys,
            layer * newCapacity,
            allocatedSequenceCapacity,
            keyDim);
        values.copyTo(
            layer * allocatedSequenceCapacity,
            grownValues,
            layer * newCapacity,
            allocatedSequenceCapacity,
            valueDim);
        System.arraycopy(
            populated,
            layer * allocatedSequenceCapacity,
            grownPopulated,
            layer * newCapacity,
            allocatedSequenceCapacity);
      }
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

  // Reject the same oversized shapes at construction, without allocating storage that a known
  // prompt reservation would immediately replace. The first reservation/store allocates once.
  private void validateInitialStorageSize(int sequenceCapacity) {
    int slots = checkedSlots(sequenceCapacity);
    checkedProduct("key cache", slots, keyDim);
    checkedProduct("value cache", slots, valueDim);
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
      KvStorage keyStore,
      KvStorage valueStore,
      int storageCapacity,
      int storageStartPosition) {
    int firstLocalPosition = firstPosition - storageStartPosition;
    float[] rawKeys = keyStore.rawFloatArray();
    float[] rawValues = valueStore.rawFloatArray();
    if (rawKeys != null && rawValues != null) {
      return new AttentionSpan(
          firstPosition,
          positionCount,
          rawKeys,
          (layer * storageCapacity + firstLocalPosition) * keyDim,
          rawValues,
          (layer * storageCapacity + firstLocalPosition) * valueDim);
    }
    float[] keyBuf = new float[positionCount * keyDim];
    float[] valBuf = new float[positionCount * valueDim];
    for (int p = 0; p < positionCount; p++) {
      int slot = layer * storageCapacity + firstLocalPosition + p;
      keyStore.readInto(slot, keyBuf, p * keyDim, keyDim);
      valueStore.readInto(slot, valBuf, p * valueDim, valueDim);
    }
    return new AttentionSpan(firstPosition, positionCount, keyBuf, 0, valBuf, 0);
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
