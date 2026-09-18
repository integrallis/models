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
package com.integrallis.models.backend.tornado;

import com.integrallis.models.backend.purejava.spi.BatchedCausalAttentionKernel;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Causal grouped-query attention on a TornadoVM device, with one KV mirror per layer.
 *
 * <h2>Scope</h2>
 *
 * <p>Single-token decode only. Prefill chunks are refused with {@link #REFUSAL_PREFILL_CHUNK} and
 * stay on the Java path; the reason is in the code below, at the gate that closes. Decode is also
 * where the arithmetic that grows with context sits, and the one real-model measurement this
 * codebase has of device attention is a prefill regression, so this is the half of the problem that
 * is both unmeasured and worth measuring.
 *
 * <h2>Where the KV cache lives</h2>
 *
 * <p>The host {@code KvCache} stays the source of truth. Nothing about forking, freezing a shared
 * prefix, rewinding, or clearing changes, and no session's cache is moved, aliased, or handed to
 * the device to own. What the device holds is a <em>mirror</em>: a position-major copy of the
 * window this kernel has been shown, written forward one chunk at a time.
 *
 * <p>The alternative — leaving the cache on the host and uploading the attended window per step —
 * is ruled out by its own arithmetic, which {@link
 * TornadoAttentionShape#hostResidentUploadBytesPerDecodeStep} states and {@code
 * TornadoAttentionShapeTest} checks. It grows linearly with position, so it crosses any link budget
 * while the arithmetic it is trying to accelerate stays constant per token.
 *
 * <h2>What that costs physical prefix sharing</h2>
 *
 * <p>Two branches forked from one frozen prefix share the host prefix arrays by identity, and they
 * still do: {@code sharesPrefixStorage} is unaffected by anything here. On the device they do not.
 * A mirror belongs to exactly one sequence, named by {@link AttentionScope#sequenceId()}; when a
 * scope names a different sequence the mirror is discarded and rebuilt from the cache, which is
 * counted as {@link TornadoAttentionRouting#mirrorRebuilds()}. So the host-side guarantee is
 * preserved and the device-side saving is not: interleaving decode across two forks of one prefix
 * rebuilds a mirror per switch, and the counter says so rather than the cost hiding in a timing.
 * Extending the mirror to hold a shared prefix segment addressed by several branches — the device
 * analogue of the cache's own span view — is the way to recover it, and it is not implemented here.
 */
public final class TornadoCausalAttentionKernel implements BatchedCausalAttentionKernel {

  static final String REFUSAL_CLOSED = "kernel closed";
  static final String REFUSAL_NO_SCOPE = "no sequence scope selected";
  static final String REFUSAL_SLIDING_WINDOW = "sliding-window attention is not supported";
  static final String REFUSAL_FUSED_ARITHMETIC =
      "fused grouped arithmetic has a different Java reference";
  static final String REFUSAL_HEAD_GEOMETRY = "grouped-query head geometry is not supported";
  static final String REFUSAL_KEY_VALUE_LENGTH = "key and value lengths differ";
  static final String REFUSAL_BATCH_SHAPE = "batch must be at least one row";
  static final String REFUSAL_PREFILL_CHUNK =
      "prefill chunks would need a second KV mirror per layer";
  static final String REFUSAL_CONTEXT = "chunk exceeds the compiled sequence length";
  static final String REFUSAL_MIRROR_AHEAD = "mirror is ahead of the requested position";
  static final String REFUSAL_LAYER = "layer index is out of range";
  static final String REFUSAL_BUDGET = "device budget for the attention mirror is exhausted";

  private static final int DEFAULT_MIRROR_CHUNK_POSITIONS = 64;

  private final int mirrorChunkPositions;
  private final long deviceByteBudget;
  private final AttentionPlan.Factory planFactory;
  private final Map<PlanKey, AttentionPlan> plans = new LinkedHashMap<>();
  private final Map<Integer, Integer> mirrored = new LinkedHashMap<>();
  private final Map<String, Long> refusalReasons = new LinkedHashMap<>();

  private AttentionScope scope;
  private long mirrorSequenceId;
  private TornadoAttentionShape acceptedShape;
  private int acceptedLayer = -1;
  private String lastRefusal = REFUSAL_NO_SCOPE;
  private long deviceDecodeLayerSteps;
  private long devicePrefillLayerChunks;
  private long refusals;
  private long mirroredPositions;
  private long mirrorRebuilds;
  private long totalNanos;
  private long committedDeviceBytes;
  private boolean closed;

  public TornadoCausalAttentionKernel() {
    this(Long.MAX_VALUE);
  }

  public TornadoCausalAttentionKernel(long deviceByteBudget) {
    this(deviceByteBudget, DEFAULT_MIRROR_CHUNK_POSITIONS, TornadoAttentionPlan::new);
  }

  TornadoCausalAttentionKernel(
      long deviceByteBudget, int mirrorChunkPositions, AttentionPlan.Factory planFactory) {
    if (deviceByteBudget < 1) {
      throw new IllegalArgumentException("deviceByteBudget must be positive");
    }
    if (mirrorChunkPositions < 1) {
      throw new IllegalArgumentException("mirrorChunkPositions must be positive");
    }
    this.deviceByteBudget = deviceByteBudget;
    this.mirrorChunkPositions = mirrorChunkPositions;
    this.planFactory = Objects.requireNonNull(planFactory, "planFactory");
  }

  /** Device bytes this kernel has committed to compiled plans and their retained KV mirrors. */
  public synchronized long committedDeviceBytes() {
    return committedDeviceBytes;
  }

  /** Sequence positions this kernel backfills into a mirror per device execution. */
  public int mirrorChunkPositions() {
    return mirrorChunkPositions;
  }

  @Override
  public synchronized void selectScope(AttentionScope newScope) {
    Objects.requireNonNull(newScope, "scope");
    this.scope = newScope;
    if (mirrorSequenceId != newScope.sequenceId()) {
      if (mirrorSequenceId != 0L && !mirrored.isEmpty()) {
        mirrorRebuilds++;
      }
      mirrorSequenceId = newScope.sequenceId();
      mirrored.clear();
    }
  }

  @Override
  public synchronized int mirroredPosition(int layer) {
    return mirrored.getOrDefault(layer, 0);
  }

  @Override
  public synchronized boolean isEligible(
      int layer,
      int startPosition,
      int batchSize,
      int numHeads,
      int numKvHeads,
      int keyLength,
      int valueLength,
      int maxSequenceLength,
      int slidingWindow) {
    acceptedShape = null;
    acceptedLayer = -1;
    if (closed) {
      return refuse(REFUSAL_CLOSED);
    }
    if (scope == null) {
      return refuse(REFUSAL_NO_SCOPE);
    }
    if (layer < 0) {
      return refuse(REFUSAL_LAYER);
    }
    if (slidingWindow != 0) {
      return refuse(REFUSAL_SLIDING_WINDOW);
    }
    if (scope.fusedGroupedArithmetic()) {
      return refuse(REFUSAL_FUSED_ARITHMETIC);
    }
    if (numHeads < 1 || numKvHeads < 1 || numHeads % numKvHeads != 0) {
      return refuse(REFUSAL_HEAD_GEOMETRY);
    }
    if (keyLength < 1 || keyLength != valueLength) {
      return refuse(REFUSAL_KEY_VALUE_LENGTH);
    }
    if (batchSize < 1) {
      return refuse(REFUSAL_BATCH_SHAPE);
    }
    if (batchSize > 1) {
      // One layer holds one mirror. A prefill chunk is a different execution shape, so it would
      // compile a second plan for the layer, and a TornadoVM plan owns its own buffers: that
      // second plan's mirror would start empty while this one's is current, and the layer would
      // silently attend against the wrong window. Sharing one mirror between two task graphs
      // depends on how the runtime associates a device buffer across plans, which is not something
      // to assume without a device to check it on; sizing the single plan to the prefill batch
      // instead makes every decode step dispatch a grid that is mostly idle work items, and decode
      // latency is what this path exists for. So prefill chunks stay on the Java path and say so.
      return refuse(REFUSAL_PREFILL_CHUNK);
    }
    if (startPosition < 0
        || maxSequenceLength < 1
        || startPosition + batchSize > maxSequenceLength) {
      return refuse(REFUSAL_CONTEXT);
    }
    if (mirroredPosition(layer) > startPosition) {
      return refuse(REFUSAL_MIRROR_AHEAD);
    }
    TornadoAttentionShape shape =
        shapeFor(numHeads, numKvHeads, keyLength, valueLength, maxSequenceLength);
    if (!plans.containsKey(new PlanKey(layer, shape))
        && committedDeviceBytes + shape.deviceBytes(1) > deviceByteBudget) {
      return refuse(REFUSAL_BUDGET);
    }
    acceptedShape = shape;
    acceptedLayer = layer;
    return true;
  }

  @Override
  public synchronized void mirrorSpan(
      int layer,
      int firstPosition,
      int positionCount,
      float[] keys,
      int keyOffset,
      int keyRowStride,
      float[] values,
      int valueOffset,
      int valueRowStride) {
    requireAccepted(layer);
    if (positionCount < 1) {
      return;
    }
    if (firstPosition != mirroredPosition(layer)) {
      throw new IllegalArgumentException(
          "mirror span must continue at " + mirroredPosition(layer) + ", got " + firstPosition);
    }
    long started = System.nanoTime();
    AttentionPlan plan = plan(layer, acceptedShape);
    int chunk = plan.mirrorChunkPositions();
    for (int row = 0; row < positionCount; row += chunk) {
      int rows = Math.min(chunk, positionCount - row);
      plan.mirror(
          firstPosition + row,
          rows,
          keys,
          keyOffset + row * keyRowStride,
          keyRowStride,
          values,
          valueOffset + row * valueRowStride,
          valueRowStride);
    }
    mirrored.put(layer, firstPosition + positionCount);
    mirroredPositions += positionCount;
    totalNanos += System.nanoTime() - started;
  }

  @Override
  public synchronized void attend(
      float[] output,
      float[] query,
      float[] key,
      float[] value,
      int layer,
      int startPosition,
      int batchSize,
      int numHeads,
      int numKvHeads,
      int keyLength,
      int valueLength,
      int maxSequenceLength,
      int slidingWindow) {
    requireAccepted(layer);
    if (mirroredPosition(layer) != startPosition) {
      throw new IllegalStateException(
          "attend requires a mirror up to " + startPosition + ", have " + mirroredPosition(layer));
    }
    long started = System.nanoTime();
    plan(layer, acceptedShape).attend(output, query, key, value, startPosition, batchSize);
    mirrored.put(layer, startPosition + batchSize);
    // Only single-row chunks are accepted today; the prefill counter is the slot a shared mirror
    // would fill, and it stays visibly at zero until there is one.
    deviceDecodeLayerSteps++;
    totalNanos += System.nanoTime() - started;
  }

  @Override
  public synchronized String lastRefusal() {
    return lastRefusal;
  }

  /** What accelerated attention did, so a kernel that never ran cannot read as one that did. */
  public synchronized TornadoAttentionRouting routing() {
    return new TornadoAttentionRouting(
        deviceDecodeLayerSteps,
        devicePrefillLayerChunks,
        refusals,
        mirroredPositions,
        mirrorRebuilds,
        plans.size(),
        totalNanos / 1_000_000.0,
        new LinkedHashMap<>(refusalReasons));
  }

  @Override
  public synchronized void rewind(int checkpoint) {
    mirrored.replaceAll((layer, position) -> Math.min(position, Math.max(checkpoint, 0)));
  }

  @Override
  public synchronized void reset() {
    mirrored.clear();
    mirrorSequenceId = 0L;
    scope = null;
    acceptedShape = null;
    acceptedLayer = -1;
  }

  @Override
  public synchronized void close() {
    if (closed) {
      return;
    }
    RuntimeException failure = null;
    for (AttentionPlan plan : plans.values()) {
      try {
        plan.close();
      } catch (RuntimeException exception) {
        if (failure == null) {
          failure = exception;
        } else {
          failure.addSuppressed(exception);
        }
      }
    }
    plans.clear();
    mirrored.clear();
    closed = true;
    if (failure != null) {
      throw failure;
    }
  }

  private boolean refuse(String reason) {
    lastRefusal = reason;
    refusals++;
    refusalReasons.merge(reason, 1L, Long::sum);
    return false;
  }

  private void requireAccepted(int layer) {
    if (acceptedShape == null || acceptedLayer != layer) {
      throw new IllegalStateException(
          "attention call for layer " + layer + " was not accepted by isEligible");
    }
  }

  private TornadoAttentionShape shapeFor(
      int numHeads, int numKvHeads, int keyLength, int valueLength, int maxSequenceLength) {
    int stagingRows = Math.min(maxSequenceLength, mirrorChunkPositions);
    return new TornadoAttentionShape(
        1,
        stagingRows,
        numHeads,
        numKvHeads,
        keyLength,
        valueLength,
        maxSequenceLength,
        0,
        scope.attentionScale());
  }

  private AttentionPlan plan(int layer, TornadoAttentionShape shape) {
    PlanKey key = new PlanKey(layer, shape);
    AttentionPlan existing = plans.get(key);
    if (existing != null) {
      return existing;
    }
    AttentionPlan created =
        planFactory.create("attention-layer-" + layer + "-b" + shape.executionBatchSize(), shape);
    plans.put(key, created);
    committedDeviceBytes = Math.addExact(committedDeviceBytes, shape.deviceBytes(1));
    return created;
  }

  private record PlanKey(int layer, TornadoAttentionShape shape) {}
}
