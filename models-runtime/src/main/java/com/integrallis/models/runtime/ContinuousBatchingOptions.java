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
package com.integrallis.models.runtime;

import java.time.Duration;
import java.util.Objects;

/** Capacity and latency controls for opt-in continuous batching of generation sessions. */
public final class ContinuousBatchingOptions {
  private final int maximumBatchSize;
  private final int maximumQueuedRequests;
  private final int maximumPrefillChunkTokens;
  private final boolean batchPrefillAcrossSessions;
  private final Duration batchFormationDelay;

  private ContinuousBatchingOptions(Builder builder) {
    maximumBatchSize = builder.maximumBatchSize;
    maximumQueuedRequests = builder.maximumQueuedRequests;
    maximumPrefillChunkTokens = builder.maximumPrefillChunkTokens;
    batchPrefillAcrossSessions = builder.batchPrefillAcrossSessions;
    batchFormationDelay = builder.batchFormationDelay;
  }

  public static Builder builder() {
    return new Builder();
  }

  /** Explicit active-request cap. */
  public int maximumBatchSize() {
    return maximumBatchSize;
  }

  /** Maximum number of requests waiting outside the active batch. */
  public int maximumQueuedRequests() {
    return maximumQueuedRequests;
  }

  /** Maximum prompt tokens evaluated before active decodes get another scheduling turn. */
  public int maximumPrefillChunkTokens() {
    return maximumPrefillChunkTokens;
  }

  /** Whether compatible prompt chunks may share physical model passes across sessions. */
  public boolean batchPrefillAcrossSessions() {
    return batchPrefillAcrossSessions;
  }

  /** Short delay used only when forming a new batch from an idle scheduler. */
  public Duration batchFormationDelay() {
    return batchFormationDelay;
  }

  public static final class Builder {
    private int maximumBatchSize;
    private int maximumQueuedRequests = 128;
    private int maximumPrefillChunkTokens = 128;
    private boolean batchPrefillAcrossSessions;
    private Duration batchFormationDelay = Duration.ofMillis(1);

    private Builder() {}

    /** Sets the required active-request cap. */
    public Builder maximumBatchSize(int value) {
      if (value <= 0) {
        throw new IllegalArgumentException("maximumBatchSize must be > 0");
      }
      maximumBatchSize = value;
      return this;
    }

    /** Bounds pending work so overload fails explicitly instead of consuming unbounded memory. */
    public Builder maximumQueuedRequests(int value) {
      if (value <= 0) {
        throw new IllegalArgumentException("maximumQueuedRequests must be > 0");
      }
      maximumQueuedRequests = value;
      return this;
    }

    /** Bounds one prompt-ingestion turn so newly admitted work cannot monopolize the model. */
    public Builder maximumPrefillChunkTokens(int value) {
      if (value <= 0) {
        throw new IllegalArgumentException("maximumPrefillChunkTokens must be > 0");
      }
      maximumPrefillChunkTokens = value;
      return this;
    }

    /**
     * Enables model-specific ragged prompt batching.
     *
     * <p>This remains disabled by default because a correct batched kernel is not necessarily
     * faster for every model, batch size, JVM, and host.
     */
    public Builder batchPrefillAcrossSessions(boolean value) {
      batchPrefillAcrossSessions = value;
      return this;
    }

    /** Sets the idle-to-active coalescing window. Zero disables deliberate coalescing. */
    public Builder batchFormationDelay(Duration value) {
      Objects.requireNonNull(value, "value");
      if (value.isNegative()) {
        throw new IllegalArgumentException("batchFormationDelay must not be negative");
      }
      batchFormationDelay = value;
      return this;
    }

    public ContinuousBatchingOptions build() {
      if (maximumBatchSize == 0) {
        throw new IllegalStateException("maximumBatchSize must be set explicitly");
      }
      return new ContinuousBatchingOptions(this);
    }
  }
}
