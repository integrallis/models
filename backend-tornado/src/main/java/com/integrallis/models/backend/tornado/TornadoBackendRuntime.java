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

import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/** Owns the automatically selected accelerated or Vector API backend. */
public final class TornadoBackendRuntime implements AutoCloseable {
  private PureJavaBackend backend;
  private final TornadoBackendStatus status;
  private final TornadoGgufBatchedMatrixKernel kernel;
  private final TornadoCausalAttentionKernel attentionKernel;

  TornadoBackendRuntime(PureJavaBackend backend, TornadoBackendStatus status) {
    this(backend, status, null, null);
  }

  TornadoBackendRuntime(
      PureJavaBackend backend, TornadoBackendStatus status, TornadoGgufBatchedMatrixKernel kernel) {
    this(backend, status, kernel, null);
  }

  TornadoBackendRuntime(
      PureJavaBackend backend,
      TornadoBackendStatus status,
      TornadoGgufBatchedMatrixKernel kernel,
      TornadoCausalAttentionKernel attentionKernel) {
    this.backend = Objects.requireNonNull(backend, "backend");
    this.status = Objects.requireNonNull(status, "status");
    this.kernel = kernel;
    this.attentionKernel = attentionKernel;
  }

  /** Returns the loaded backend used by the ordinary Models generation pipeline. */
  public PureJavaBackend backend() {
    if (backend == null) {
      throw new IllegalStateException("backend ownership was transferred");
    }
    return backend;
  }

  PureJavaBackend detachBackend() {
    PureJavaBackend detached = backend();
    backend = null;
    return detached;
  }

  /** Returns the device-selection, fallback, and readiness outcome. */
  public TornadoBackendStatus status() {
    return status;
  }

  /**
   * Returns the projections routed to the device so far, keyed by GGUF weight format.
   *
   * <p>Empty when the load fell back to the Vector API. A grouped dispatch counts once per matrix,
   * so a mixed-format model's counts show whether each of its formats reached the device rather
   * than only the majority one.
   */
  public Map<String, Long> routedProjectionsByFormat() {
    return kernel == null ? Map.of() : kernel.routedProjectionsByFormat();
  }

  /** Number of distinct compiled device execution plans, or zero when not accelerated. */
  public int projectionPlanCount() {
    return kernel == null ? 0 : kernel.projectionPlanCount();
  }

  /**
   * What accelerated attention did, when it is installed.
   *
   * <p>Empty means no attention kernel was installed at all, which is a different statement from a
   * kernel that ran and refused every step; the latter reports itself through {@link
   * TornadoAttentionRouting#refusalReasons()}.
   */
  public Optional<TornadoAttentionRouting> attentionRouting() {
    return Optional.ofNullable(attentionKernel).map(TornadoCausalAttentionKernel::routing);
  }

  @Override
  public void close() {
    if (backend != null) {
      backend.close();
      backend = null;
    }
  }
}
