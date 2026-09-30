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
 * One layer's retained attention execution, with its KV mirror.
 *
 * <p>Separating this from {@link TornadoCausalAttentionKernel} is what lets the routing, the mirror
 * bookkeeping, and the refusal reasons be tested without a device: the tests supply an
 * implementation that runs {@link TornadoAttentionKernel} as ordinary Java, and production supplies
 * {@link TornadoAttentionPlan}, which hands the same kernel methods to TornadoVM. Both drive the
 * identical kernel source, so an off-device test is exercising the code that will be compiled to
 * PTX and not a stand-in for it.
 */
interface AttentionPlan extends AutoCloseable {

  /** Writes one contiguous run of already-cached rows into the mirror at their own positions. */
  void mirror(
      int firstPosition,
      int positionCount,
      float[] keys,
      int keyOffset,
      int keyRowStride,
      float[] values,
      int valueOffset,
      int valueRowStride);

  /** Appends the staged rows of this chunk and attends them against the mirrored window. */
  void attend(
      float[] output, float[] query, float[] key, float[] value, int startPosition, int batchSize);

  /** Positions this plan can absorb in one mirror call. */
  int mirrorChunkPositions();

  @Override
  void close();

  /** Opens one plan per layer and shape. */
  interface Factory {
    AttentionPlan create(String name, TornadoAttentionShape shape);
  }
}
