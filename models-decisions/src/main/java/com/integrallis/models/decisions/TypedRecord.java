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

import java.util.Arrays;
import java.util.Objects;

/**
 * One harvested state and its outcome, for an answer space of any width.
 *
 * <p>The binary {@link HarvestRecord} carries a {@code boolean} because a Noul has exactly two
 * outcomes. A Choice or a Score does not, so the outcome here is the index of the true label in
 * the space's declaration order. Keeping these as separate types rather than widening the binary
 * one means a Noul harvest cannot silently acquire a third outcome.
 *
 * <p>The state is copied in and out. A caller holding the array it passed could otherwise mutate a
 * record after it had been assigned to a split, which would move training data into the sealed set
 * without any of the split machinery noticing.
 */
public record TypedRecord(String id, String split, int outcome, boolean truncated, float[] state) {

  /** Validates the record and copies the state in. */
  public TypedRecord {
    Objects.requireNonNull(id, "id");
    Objects.requireNonNull(split, "split");
    Objects.requireNonNull(state, "state");
    if (outcome < 0) {
      throw new IllegalArgumentException("outcome must not be negative: " + outcome);
    }
    if (state.length == 0) {
      throw new IllegalArgumentException("state must not be empty");
    }
    state = state.clone();
  }

  /** The state, copied, so a caller cannot reach back into the record. */
  @Override
  public float[] state() {
    return state.clone();
  }

  /** The state, uncopied, for the training path only. Never hand this to a caller. */
  float[] stateInternal() {
    return state;
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    return other instanceof TypedRecord that
        && outcome == that.outcome
        && truncated == that.truncated
        && id.equals(that.id)
        && split.equals(that.split)
        && Arrays.equals(state, that.state);
  }

  @Override
  public int hashCode() {
    return Objects.hash(id, split, outcome, truncated, Arrays.hashCode(state));
  }

  @Override
  public String toString() {
    return "TypedRecord[id=" + id + ", split=" + split + ", outcome=" + outcome + ", width="
        + state.length + "]";
  }
}
