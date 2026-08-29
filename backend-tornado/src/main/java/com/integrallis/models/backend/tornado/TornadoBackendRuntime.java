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
import java.util.Objects;

/** Owns the automatically selected accelerated or Vector API backend. */
public final class TornadoBackendRuntime implements AutoCloseable {
  private PureJavaBackend backend;
  private final TornadoBackendStatus status;

  TornadoBackendRuntime(PureJavaBackend backend, TornadoBackendStatus status) {
    this.backend = Objects.requireNonNull(backend, "backend");
    this.status = Objects.requireNonNull(status, "status");
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

  @Override
  public void close() {
    if (backend != null) {
      backend.close();
      backend = null;
    }
  }
}
