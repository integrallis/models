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

/** Immutable controls for optional TornadoVM device selection and readiness. */
public record TornadoBackendOptions(
    boolean accelerateDecode,
    boolean eagerReadiness,
    boolean requireAccelerator,
    int executionBatchSize,
    boolean accelerateAttention) {

  public TornadoBackendOptions {
    if (executionBatchSize < 4) {
      throw new IllegalArgumentException("executionBatchSize must be at least 4");
    }
  }

  /** Projection-only controls, for callers written before attention could be accelerated. */
  public TornadoBackendOptions(
      boolean accelerateDecode,
      boolean eagerReadiness,
      boolean requireAccelerator,
      int executionBatchSize) {
    this(accelerateDecode, eagerReadiness, requireAccelerator, executionBatchSize, false);
  }

  /**
   * Selects qualified hardware automatically and falls back to the Java Vector API.
   *
   * <p>Attention stays off by default. Its device-resident KV mirror changes what the accelerator
   * holds for the life of a sequence and how forked branches behave on the device, and neither has
   * been through a real-hardware gate yet; the projection scope has. Opting in is explicit until
   * that changes.
   */
  public static TornadoBackendOptions defaults() {
    return new TornadoBackendOptions(true, true, false, 32, false);
  }

  /** Adds device attention to these controls. */
  public TornadoBackendOptions withAcceleratedAttention() {
    return new TornadoBackendOptions(
        accelerateDecode, eagerReadiness, requireAccelerator, executionBatchSize, true);
  }
}
