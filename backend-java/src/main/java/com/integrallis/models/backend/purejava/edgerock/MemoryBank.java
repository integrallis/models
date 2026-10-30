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
package com.integrallis.models.backend.purejava.edgerock;

import java.util.Objects;

/**
 * One memory layer's state: a committed bank of key/value records plus the pending chunk.
 *
 * <p>The bank is immutable between commits. Records stage into {@code pending} for {@code
 * commitChunk} tokens, and only then does {@link #commit()} recompute the kept set from (committed
 * ++ pending). Committing per token would keep a strictly better bank and still be wrong, because
 * it would not be this model.
 *
 * <p>Laid out group-major -- {@code [group][slot][dim]} -- because the read is grouped-query: every
 * head in a group scores against the same keys, so a group's records want to be contiguous.
 */
final class MemoryBank {

  private final int groups;
  private final int capacity;
  private final int commitChunk;
  private final int dim;

  /** Committed records, [group][slot][dim]. Valid only where the selection says so. */
  private final float[][][] keys;

  private final float[][][] values;

  /** Which records are committed, and their admission scores and positions. */
  private MemoryAnchors.Selection selection;

  private final float[][][] pendingKeys;
  private final float[][][] pendingValues;
  private final float[][] pendingScores;
  private final int[] pendingPositions;

  /** Tokens staged since the last commit. */
  private int count;

  /** Valid tokens seen in the whole sequence, which is what positions are derived from. */
  private int total;

  MemoryBank(int groups, int capacity, int commitChunk, int dim) {
    if (groups <= 0 || capacity <= 0 || commitChunk <= 0 || dim <= 0) {
      throw new IllegalArgumentException(
          "groups, capacity, commitChunk and dim must be positive: "
              + groups
              + ", "
              + capacity
              + ", "
              + commitChunk
              + ", "
              + dim);
    }
    if (commitChunk > capacity) {
      throw new IllegalArgumentException(
          "commitChunk " + commitChunk + " exceeds capacity " + capacity);
    }
    this.groups = groups;
    this.capacity = capacity;
    this.commitChunk = commitChunk;
    this.dim = dim;
    this.keys = new float[groups][capacity][dim];
    this.values = new float[groups][capacity][dim];
    this.selection = MemoryAnchors.empty(capacity);
    this.pendingKeys = new float[groups][commitChunk][dim];
    this.pendingValues = new float[groups][commitChunk][dim];
    this.pendingScores = new float[groups][commitChunk];
    this.pendingPositions = new int[commitChunk];
  }

  int groups() {
    return groups;
  }

  int capacity() {
    return capacity;
  }

  int commitChunk() {
    return commitChunk;
  }

  int dim() {
    return dim;
  }

  int staged() {
    return count;
  }

  int total() {
    return total;
  }

  MemoryAnchors.Selection selection() {
    return selection;
  }

  /** A committed record's key, for the read path. */
  float[] key(int group, int slot) {
    return keys[group][slot];
  }

  float[] value(int group, int slot) {
    return values[group][slot];
  }

  /**
   * Stages one token's record and commits if the chunk is full.
   *
   * @param groupKeys per-group key, {@code [group][dim]}
   * @param groupValues per-group value, {@code [group][dim]}
   * @param scores per-group admission score
   * @return whether this token triggered a commit
   */
  boolean stage(float[][] groupKeys, float[][] groupValues, float[] scores) {
    Objects.requireNonNull(groupKeys, "groupKeys");
    Objects.requireNonNull(groupValues, "groupValues");
    Objects.requireNonNull(scores, "scores");
    int slot = count;
    for (int group = 0; group < groups; group++) {
      System.arraycopy(groupKeys[group], 0, pendingKeys[group][slot], 0, dim);
      System.arraycopy(groupValues[group], 0, pendingValues[group][slot], 0, dim);
      pendingScores[group][slot] = scores[group];
    }
    // Position is derived rather than passed: total has not yet counted this token, so the token
    // staged at slot i of this chunk sits at (total - count) + i in the sequence.
    pendingPositions[slot] = total - count + slot;
    count++;
    total++;
    if (count == commitChunk) {
      commit();
      return true;
    }
    return false;
  }

  /**
   * Recomputes the kept set from the committed bank plus the staged chunk.
   *
   * <p>References below {@code capacity} point into the committed bank; the rest point into the
   * pending chunk. That is how one merge can choose from both without copying first.
   */
  void commit() {
    if (count == 0) {
      return;
    }
    float[] pendingScoreRow = new float[count];
    int[] pendingPositionRow = new int[count];
    int[] pendingReferenceRow = new int[count];
    float[][][] nextKeys = new float[groups][capacity][dim];
    float[][][] nextValues = new float[groups][capacity][dim];
    MemoryAnchors.Selection merged = null;

    for (int group = 0; group < groups; group++) {
      for (int i = 0; i < count; i++) {
        pendingScoreRow[i] = pendingScores[group][i];
        pendingPositionRow[i] = pendingPositions[i];
        pendingReferenceRow[i] = capacity + i;
      }
      MemoryAnchors.Selection pending =
          MemoryAnchors.select(pendingScoreRow, pendingPositionRow, pendingReferenceRow, count);
      MemoryAnchors.Selection kept = MemoryAnchors.merge(selection, pending, capacity);
      // Every group admits independently -- the reference calls this independent memory -- but the
      // selection object is shared because scores and positions agree across groups by
      // construction; only the records differ. Gather this group's records by the kept references.
      for (int slot = 0; slot < capacity; slot++) {
        int reference = kept.reference(slot);
        if (reference < 0) {
          continue;
        }
        float[] sourceKey =
            reference < capacity
                ? keys[group][reference]
                : pendingKeys[group][reference - capacity];
        float[] sourceValue =
            reference < capacity
                ? values[group][reference]
                : pendingValues[group][reference - capacity];
        System.arraycopy(sourceKey, 0, nextKeys[group][slot], 0, dim);
        System.arraycopy(sourceValue, 0, nextValues[group][slot], 0, dim);
      }
      merged = kept;
    }

    for (int group = 0; group < groups; group++) {
      for (int slot = 0; slot < capacity; slot++) {
        System.arraycopy(nextKeys[group][slot], 0, keys[group][slot], 0, dim);
        System.arraycopy(nextValues[group][slot], 0, values[group][slot], 0, dim);
      }
    }
    // After a commit the references describe slot order, not provenance.
    float[] scores = merged.scoresCopy();
    int[] positions = merged.positionsCopy();
    int[] references = new int[capacity];
    for (int slot = 0; slot < capacity; slot++) {
      references[slot] = merged.reference(slot) < 0 ? MemoryAnchors.EMPTY : slot;
    }
    selection = MemoryAnchors.select(scores, positions, references, capacity);
    count = 0;
  }

  /** Clears the bank, as a new sequence requires. */
  void reset() {
    selection = MemoryAnchors.empty(capacity);
    count = 0;
    total = 0;
    for (int group = 0; group < groups; group++) {
      for (int slot = 0; slot < capacity; slot++) {
        java.util.Arrays.fill(keys[group][slot], 0f);
        java.util.Arrays.fill(values[group][slot], 0f);
      }
    }
  }
}
