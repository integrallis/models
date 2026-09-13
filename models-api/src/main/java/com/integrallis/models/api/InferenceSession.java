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
package com.integrallis.models.api;

import java.util.OptionalLong;

/** Opaque state for one independent autoregressive sequence. */
public interface InferenceSession extends AutoCloseable {

  /** Returns the next zero-based sequence position. */
  int checkpoint();

  /** Returns whether this session has released its backend-owned state. */
  boolean isClosed();

  /**
   * Returns the backend-reported storage bytes allocated for request-specific inference state, when
   * measurable.
   *
   * <p>The value includes shared storage reachable by this session. Consumers counting multiple
   * branches must subtract storage proven common through a {@link SharedInferencePrefix} exactly
   * once. Implementations may report array or native-buffer payload rather than VM object headers;
   * process RSS remains the complete process-level measurement.
   */
  default OptionalLong allocatedStateBytes() {
    return OptionalLong.empty();
  }

  /** Releases request-specific state. The operation is idempotent. */
  @Override
  void close();
}
