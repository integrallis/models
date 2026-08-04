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

import java.util.Objects;
import java.util.OptionalInt;

/**
 * Immutable view of one loaded inference context's token capacity and current position.
 *
 * @param capacity number of token positions allocated by the active backend
 * @param position next zero-based token position, or empty when the backend cannot report it
 */
public record InferenceContextWindow(int capacity, OptionalInt position) {

  /** Validates the context-window snapshot. */
  public InferenceContextWindow {
    if (capacity <= 0) {
      throw new IllegalArgumentException("capacity must be positive: " + capacity);
    }
    position = Objects.requireNonNull(position, "position");
    if (position.isPresent() && (position.getAsInt() < 0 || position.getAsInt() > capacity)) {
      throw new IllegalArgumentException(
          "position must be in [0, capacity]: " + position.getAsInt() + " and " + capacity);
    }
  }

  /** Returns the unused token positions, or empty when the current position is unavailable. */
  public OptionalInt remaining() {
    return position.isPresent()
        ? OptionalInt.of(capacity - position.getAsInt())
        : OptionalInt.empty();
  }
}
