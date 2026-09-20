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
import java.util.List;
import java.util.Objects;

/**
 * The outcome of one decision: a probability distribution over exactly one answer space.
 *
 * <p>A verdict is bound to the space that produced it, so {@link #winner()} is always one of that
 * space's declared labels. The distribution is validated on construction rather than trusted, and
 * the array is copied in and out so a verdict cannot be altered after the fact.
 *
 * <p>A calibrated probability describes how often decisions at this confidence are right. It says
 * nothing about whether this particular decision is right.
 */
public final class Verdict {

  private static final double SUM_TOLERANCE = 1e-9;

  private final AnswerSpace space;
  private final double[] probabilities;

  /**
   * Creates a verdict over the given space.
   *
   * @param space the answer space the distribution covers
   * @param probabilities one non-negative finite probability per outcome, summing to one
   * @throws IllegalArgumentException if the distribution does not cover exactly the space
   */
  public Verdict(AnswerSpace space, double[] probabilities) {
    this.space = Objects.requireNonNull(space, "space");
    Objects.requireNonNull(probabilities, "probabilities");
    if (probabilities.length != space.size()) {
      throw new IllegalArgumentException(
          "distribution must cover all " + space.size() + " outcomes, got " + probabilities.length);
    }
    double total = 0.0;
    for (double probability : probabilities) {
      if (!Double.isFinite(probability) || probability < 0.0) {
        throw new IllegalArgumentException("probability must be finite and non-negative");
      }
      total += probability;
    }
    if (Math.abs(total - 1.0) > SUM_TOLERANCE) {
      throw new IllegalArgumentException("distribution must sum to one, got " + total);
    }
    this.probabilities = probabilities.clone();
  }

  /** Returns the answer space this verdict covers. */
  public AnswerSpace space() {
    return space;
  }

  /** Returns a copy of the distribution, in the space's declaration order. */
  public double[] probabilities() {
    return probabilities.clone();
  }

  /**
   * Returns the most probable label, resolving ties toward the earlier declaration so repeated runs
   * over identical input agree.
   */
  public String winner() {
    int best = 0;
    for (int index = 1; index < probabilities.length; index++) {
      if (probabilities[index] > probabilities[best]) {
        best = index;
      }
    }
    return space.labels().get(best);
  }

  /**
   * Returns the probability of one declared label.
   *
   * @throws IllegalArgumentException if the label is not part of this verdict's space
   */
  public double probabilityOf(String label) {
    Objects.requireNonNull(label, "label");
    List<String> labels = space.labels();
    int index = labels.indexOf(label);
    if (index < 0) {
      throw new IllegalArgumentException("unknown label " + label + ", expected one of " + labels);
    }
    return probabilities[index];
  }

  /**
   * Returns the probability that a binary proposition holds.
   *
   * @throws IllegalStateException if this verdict does not answer a {@link Noul}
   */
  public double probabilityOfTrue() {
    if (!(space instanceof Noul noul)) {
      throw new IllegalStateException("only a Noul carries a truth probability, not " + space);
    }
    return probabilities[noul.trueIndex()];
  }

  /**
   * Returns how concentrated the distribution is, as one minus its normalized entropy: one when a
   * single outcome takes all the mass, zero when the outcomes are indistinguishable.
   *
   * <p>This measures decisiveness, not correctness.
   */
  public double confidence() {
    double entropy = 0.0;
    for (double probability : probabilities) {
      if (probability > 0.0) {
        entropy -= probability * Math.log(probability);
      }
    }
    double normalized = entropy / Math.log(probabilities.length);
    return Math.clamp(1.0 - normalized, 0.0, 1.0);
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof Verdict verdict
        && space.equals(verdict.space)
        && Arrays.equals(probabilities, verdict.probabilities);
  }

  @Override
  public int hashCode() {
    return 31 * space.hashCode() + Arrays.hashCode(probabilities);
  }

  @Override
  public String toString() {
    return "Verdict["
        + space.question()
        + " -> "
        + winner()
        + " "
        + Arrays.toString(probabilities)
        + "]";
  }
}
