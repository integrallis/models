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
package com.integrallis.models.router;

import java.util.Optional;
import java.util.OptionalInt;

/** Live operational facts layered over a model's immutable qualification evidence. */
public final class ModelRuntimeState {
  private static final ModelRuntimeState UNKNOWN = new Builder().build();

  private final boolean available;
  private final Boolean resident;
  private final Integer queueDepth;

  private ModelRuntimeState(Builder builder) {
    this.available = builder.available;
    this.resident = builder.resident;
    this.queueDepth = builder.queueDepth;
  }

  /** Returns a healthy state with no residency or load observation. */
  public static ModelRuntimeState unknown() {
    return UNKNOWN;
  }

  public static Builder builder() {
    return new Builder();
  }

  public boolean available() {
    return available;
  }

  public Optional<Boolean> resident() {
    return Optional.ofNullable(resident);
  }

  public OptionalInt queueDepth() {
    return queueDepth == null ? OptionalInt.empty() : OptionalInt.of(queueDepth);
  }

  /** Fluent builder. */
  public static final class Builder {
    private boolean available = true;
    private Boolean resident;
    private Integer queueDepth;

    private Builder() {}

    public Builder available(boolean value) {
      this.available = value;
      return this;
    }

    public Builder resident(boolean value) {
      this.resident = value;
      return this;
    }

    public Builder queueDepth(int value) {
      if (value < 0) {
        throw new IllegalArgumentException("queueDepth must not be negative");
      }
      this.queueDepth = value;
      return this;
    }

    public ModelRuntimeState build() {
      return new ModelRuntimeState(this);
    }
  }
}
