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
package com.integrallis.models.decisions;

import java.util.Objects;

/**
 * One harvested item: the label the corpus gives, the verdict the base reached through its decode
 * loop, and the hidden state a head reads instead.
 *
 * @param id the corpus item identifier
 * @param split which split the item was drawn into
 * @param label the corpus's own label, the only quantity here that is ground truth
 * @param decode the verdict the base reached autoregressively, which is not ground truth
 * @param truncated whether the context window cut this item's state
 * @param cpuMicros process CPU microseconds attributable to this item
 * @param hidden the base's final normalized hidden state
 */
public record HarvestRecord(
    String id,
    String split,
    boolean label,
    boolean decode,
    boolean truncated,
    long cpuMicros,
    float[] hidden) {

  /** Validates the record and copies the hidden state in. */
  public HarvestRecord {
    Objects.requireNonNull(id, "id");
    Objects.requireNonNull(split, "split");
    Objects.requireNonNull(hidden, "hidden");
    if (id.isBlank()) {
      throw new IllegalArgumentException("id must not be blank");
    }
    if (hidden.length == 0) {
      throw new IllegalArgumentException("hidden state must not be empty");
    }
    hidden = hidden.clone();
  }

  /**
   * Returns a copy of the hidden state.
   *
   * <p>A harvest record outlives the buffer it was built from and is read by more than one caller,
   * so the array is copied in and out. The scoring path, which reads every record once per fit and
   * would double the working set by copying, uses {@link #hiddenInternal()} instead.
   */
  @Override
  public float[] hidden() {
    return hidden.clone();
  }

  /** Returns the hidden state without copying. Callers must not modify it. */
  float[] hiddenInternal() {
    return hidden;
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof HarvestRecord record
        && label == record.label
        && decode == record.decode
        && truncated == record.truncated
        && cpuMicros == record.cpuMicros
        && id.equals(record.id)
        && split.equals(record.split)
        && java.util.Arrays.equals(hidden, record.hidden);
  }

  @Override
  public int hashCode() {
    return Objects.hash(id, split, label, decode, truncated, cpuMicros) * 31
        + java.util.Arrays.hashCode(hidden);
  }
}
