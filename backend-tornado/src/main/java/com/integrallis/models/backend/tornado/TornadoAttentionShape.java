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

/**
 * One compiled attention plan's fixed geometry, and the device memory it costs.
 *
 * <p>The memory arithmetic lives here rather than inside the plan because it decides where the KV
 * cache is allowed to live, and that decision has to be made — and testable — before any device is
 * touched. A device-resident mirror is what makes accelerated attention worth doing at all: with
 * the cache left on the host, every decode step would have to re-upload the whole attended window,
 * {@code numLayers * position * (keyDim + valueDim) * 4} bytes per token, which crosses the link
 * budget long before it saves any arithmetic. Mirroring moves that to a one-off backfill plus one
 * new row per layer per step. What it costs is {@link #mirrorBytesPerLayer()} held for the life of
 * the sequence, per sequence.
 */
public record TornadoAttentionShape(
    int executionBatchSize,
    int stagingRows,
    int numHeads,
    int numKvHeads,
    int keyLength,
    int valueLength,
    int maxSequenceLength,
    int slidingWindow,
    float scale) {

  public TornadoAttentionShape {
    if (executionBatchSize < 1) {
      throw new IllegalArgumentException("executionBatchSize must be positive");
    }
    if (stagingRows < executionBatchSize) {
      throw new IllegalArgumentException("stagingRows must cover the execution batch");
    }
    if (numHeads < 1 || numKvHeads < 1 || numHeads % numKvHeads != 0) {
      throw new IllegalArgumentException(
          "numHeads must be a positive multiple of numKvHeads: " + numHeads + "/" + numKvHeads);
    }
    if (keyLength < 1 || valueLength < 1) {
      throw new IllegalArgumentException("keyLength and valueLength must be positive");
    }
    if (slidingWindow < 0) {
      throw new IllegalArgumentException("slidingWindow must not be negative");
    }
    if (maxSequenceLength < executionBatchSize || maxSequenceLength < stagingRows) {
      throw new IllegalArgumentException("maxSequenceLength must cover one execution");
    }
    if (!Float.isFinite(scale)) {
      throw new IllegalArgumentException("scale must be finite: " + scale);
    }
  }

  public int groupSize() {
    return numHeads / numKvHeads;
  }

  public int queryDim() {
    return Math.multiplyExact(numHeads, keyLength);
  }

  public int keyDim() {
    return Math.multiplyExact(numKvHeads, keyLength);
  }

  public int valueDim() {
    return Math.multiplyExact(numKvHeads, valueLength);
  }

  public int outputDim() {
    return Math.multiplyExact(numHeads, valueLength);
  }

  /** Bytes one mirrored sequence position costs in one layer. */
  public long mirrorBytesPerPosition() {
    return Math.multiplyExact((long) keyDim() + valueDim(), Float.BYTES);
  }

  /** Device bytes one layer's full-context KV mirror holds for the life of a sequence. */
  public long mirrorBytesPerLayer() {
    return Math.multiplyExact(mirrorBytesPerPosition(), maxSequenceLength);
  }

  /**
   * Device bytes one layer's plan holds besides the mirror: staged rows, the score scratch, and the
   * output block.
   *
   * <p>The score scratch dominates and is sized for the whole context, because a plan is compiled
   * once and reused at every position of the sequence.
   */
  public long scratchBytesPerLayer() {
    long staged =
        Math.multiplyExact(
            (long) stagingRows, Math.multiplyExact((long) keyDim() + valueDim(), Float.BYTES));
    long queries =
        Math.multiplyExact(
            (long) executionBatchSize, Math.multiplyExact((long) queryDim(), Float.BYTES));
    long outputs =
        Math.multiplyExact(
            (long) executionBatchSize, Math.multiplyExact((long) outputDim(), Float.BYTES));
    long scoreRows = Math.multiplyExact((long) executionBatchSize, numHeads);
    long scores =
        Math.multiplyExact(scoreRows, Math.multiplyExact((long) maxSequenceLength, Float.BYTES));
    return staged + queries + outputs + scores;
  }

  /** Total device bytes accelerated attention holds for one sequence across every layer. */
  public long deviceBytes(int numLayers) {
    if (numLayers < 1) {
      throw new IllegalArgumentException("numLayers must be positive");
    }
    return Math.multiplyExact(
        (long) numLayers, Math.addExact(mirrorBytesPerLayer(), scratchBytesPerLayer()));
  }

  /**
   * Bytes a host-resident cache would have to re-upload for one decode step across every layer: the
   * whole attended window, every step.
   *
   * <p>Recorded as arithmetic rather than prose so the choice to mirror can be checked rather than
   * asserted.
   */
  public long hostResidentUploadBytesPerDecodeStep(int numLayers, int position) {
    if (numLayers < 1) {
      throw new IllegalArgumentException("numLayers must be positive");
    }
    if (position < 0) {
      throw new IllegalArgumentException("position must not be negative");
    }
    int attended = position + 1;
    if (slidingWindow > 0) {
      attended = Math.min(attended, slidingWindow);
    }
    return Math.multiplyExact(
        (long) numLayers, Math.multiplyExact(mirrorBytesPerPosition(), attended));
  }

  /** Bytes a device-resident mirror uploads for one decode step across every layer. */
  public long mirroredUploadBytesPerDecodeStep(int numLayers) {
    if (numLayers < 1) {
      throw new IllegalArgumentException("numLayers must be positive");
    }
    long perLayer =
        Math.addExact(
            mirrorBytesPerPosition(),
            Math.multiplyExact((long) queryDim() + outputDim(), Float.BYTES));
    return Math.multiplyExact((long) numLayers, perLayer);
  }
}
