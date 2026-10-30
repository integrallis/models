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

import java.util.Arrays;
import java.util.Comparator;
import java.util.Objects;

/**
 * The admission rule for EdgeRock's Rock Memory Anchors: a bounded, stable top-k over {@code
 * (score, position, reference)} triples.
 *
 * <p>EdgeRock keeps a fixed bank of key/value records -- 1024 slots, from {@code
 * edgerock.memory.capacity} -- and every {@code edgerock.memory.commit_chunk} tokens it decides
 * which records survive. That decision is this class. Everything else in the memory layer is
 * projections and attention that the Qwen3.5 path already has shapes for; this selection has no
 * existing equivalent, which is why it lands first and alone.
 *
 * <p><strong>Two stable passes, in this order.</strong> The reference implementation sorts by
 * position ascending with invalid slots forced last, and only then takes the top {@code capacity}
 * by score descending. Because the second pass is stable, the first one is what breaks ties: among
 * equal scores the earlier position is kept. Scoring first, or sorting unstably, selects a
 * different set of records -- not a rounding difference but different memory contents, which
 * diverge further at every commit. A model that silently remembers different things still produces
 * fluent text, so this is checked by test rather than by inspection.
 *
 * <p>Scores arrive as {@code float} deliberately. The reference computes admission in float32 with
 * autocast disabled even when the model runs in bfloat16, so narrowing them here would change the
 * selection, not merely its precision.
 *
 * <p>See {@code experiments/edgerock-decoder/IMPLEMENTATION.md} for the whole mechanism and the
 * provenance of this reading.
 */
public final class MemoryAnchors {

  /** The sentinel score of an empty slot. Sorts last under descending order. */
  public static final float EMPTY_SCORE = Float.NEGATIVE_INFINITY;

  /** The sentinel position and reference of an empty slot. */
  public static final int EMPTY = -1;

  private MemoryAnchors() {}

  /**
   * A selection of memory slots, always exactly {@code capacity} wide and padded with empties.
   *
   * <p>The three parallel arrays are private and reached by index rather than returned. A record
   * with array components hands every caller a mutable view of the bank -- SpotBugs EI/EI2 -- and
   * cloning on each accessor would put three array copies on the read path of every decoded token.
   * Indexed access costs nothing and a decoder wants it anyway; the {@code *Copy} methods exist for
   * tests and diagnostics, where a copy is the point.
   */
  public static final class Selection {

    private final float[] scores;
    private final int[] positions;
    private final int[] references;

    Selection(float[] scores, int[] positions, int[] references) {
      Objects.requireNonNull(scores, "scores");
      Objects.requireNonNull(positions, "positions");
      Objects.requireNonNull(references, "references");
      if (scores.length != positions.length || scores.length != references.length) {
        throw new IllegalArgumentException(
            "scores, positions and references must agree in length: "
                + scores.length
                + ", "
                + positions.length
                + ", "
                + references.length);
      }
      this.scores = scores.clone();
      this.positions = positions.clone();
      this.references = references.clone();
    }

    public int capacity() {
      return scores.length;
    }

    /** Admission score of a slot, {@link #EMPTY_SCORE} when empty. */
    public float score(int slot) {
      return scores[slot];
    }

    /** Token position of a slot, {@link #EMPTY} when empty. */
    public int position(int slot) {
      return positions[slot];
    }

    /** Source index a slot was gathered from, {@link #EMPTY} when empty. */
    public int reference(int slot) {
      return references[slot];
    }

    /** Whether this slot holds a record. */
    public boolean occupied(int slot) {
      return references[slot] >= 0;
    }

    /** How many slots hold records. Empties are contiguous at the end. */
    public int occupancy() {
      int count = 0;
      while (count < references.length && references[count] >= 0) {
        count++;
      }
      return count;
    }

    public float[] scoresCopy() {
      return scores.clone();
    }

    public int[] positionsCopy() {
      return positions.clone();
    }

    public int[] referencesCopy() {
      return references.clone();
    }
  }

  /**
   * Keeps the {@code capacity} highest-scoring slots, breaking ties by earlier position.
   *
   * <p>A slot whose reference is {@link #EMPTY} is treated as absent whatever its score or
   * position, which is how the reference implementation marks a bank that is not yet full.
   */
  public static Selection select(float[] scores, int[] positions, int[] references, int capacity) {
    Objects.requireNonNull(scores, "scores");
    Objects.requireNonNull(positions, "positions");
    Objects.requireNonNull(references, "references");
    if (scores.length != positions.length || scores.length != references.length) {
      throw new IllegalArgumentException("scores, positions and references must agree in length");
    }
    if (capacity <= 0) {
      throw new IllegalArgumentException("capacity must be positive: " + capacity);
    }

    Integer[] order = new Integer[scores.length];
    for (int i = 0; i < order.length; i++) {
      order[i] = i;
    }
    // Pass one: by position ascending, invalid slots last. Arrays.sort on boxed keys is a stable
    // merge sort, which the tie-breaking below depends on.
    Arrays.sort(
        order,
        Comparator.comparingLong(
            index -> references[index] < 0 ? Long.MAX_VALUE : positions[index]));
    // Pass two: by score descending, stable, so pass one's ordering survives among equal scores.
    // An invalid slot scores as empty whatever its score field holds. The reference keeps that as
    // an invariant -- it pads empties with -inf -- rather than enforcing it here, but depending on
    // an invariant we do not check is how a bank with one stray high-scoring empty slot would
    // evict every real record and still return a well-formed selection.
    Arrays.sort(
        order,
        Comparator.comparingDouble(
                (Integer index) -> references[index] < 0 ? EMPTY_SCORE : scores[index])
            .reversed());

    float[] keptScores = new float[capacity];
    int[] keptPositions = new int[capacity];
    int[] keptReferences = new int[capacity];
    Arrays.fill(keptScores, EMPTY_SCORE);
    Arrays.fill(keptPositions, EMPTY);
    Arrays.fill(keptReferences, EMPTY);
    int kept = Math.min(capacity, order.length);
    for (int slot = 0; slot < kept; slot++) {
      int index = order[slot];
      if (references[index] < 0) {
        // Invalid slots sorted here only because nothing better remains; leave the rest padded.
        break;
      }
      keptScores[slot] = scores[index];
      keptPositions[slot] = positions[index];
      keptReferences[slot] = references[index];
    }
    return new Selection(keptScores, keptPositions, keptReferences);
  }

  /**
   * Merges two selections and keeps the best {@code capacity}.
   *
   * <p>This is the commit step: the committed bank on the left, the pending chunk on the right.
   * Concatenating before selecting makes the operation associative, which is what lets prefill
   * reduce over a tree and decode merge once, with the same result.
   */
  public static Selection merge(Selection left, Selection right, int capacity) {
    Objects.requireNonNull(left, "left");
    Objects.requireNonNull(right, "right");
    int total = left.capacity() + right.capacity();
    float[] scores = new float[total];
    int[] positions = new int[total];
    int[] references = new int[total];
    for (int slot = 0; slot < left.capacity(); slot++) {
      scores[slot] = left.score(slot);
      positions[slot] = left.position(slot);
      references[slot] = left.reference(slot);
    }
    for (int slot = 0; slot < right.capacity(); slot++) {
      int target = left.capacity() + slot;
      scores[target] = right.score(slot);
      positions[target] = right.position(slot);
      references[target] = right.reference(slot);
    }
    return select(scores, positions, references, capacity);
  }

  /** An empty bank of {@code capacity} slots, which is what a fresh sequence starts from. */
  public static Selection empty(int capacity) {
    if (capacity <= 0) {
      throw new IllegalArgumentException("capacity must be positive: " + capacity);
    }
    float[] scores = new float[capacity];
    int[] positions = new int[capacity];
    int[] references = new int[capacity];
    Arrays.fill(scores, EMPTY_SCORE);
    Arrays.fill(positions, EMPTY);
    Arrays.fill(references, EMPTY);
    return new Selection(scores, positions, references);
  }
}
