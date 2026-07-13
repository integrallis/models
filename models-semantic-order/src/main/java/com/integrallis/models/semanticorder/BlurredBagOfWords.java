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
package com.integrallis.models.semanticorder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalInt;

/** A sparse L1-normalized bag of words blurred along a semantic order. */
public final class BlurredBagOfWords {
  /** Radius used by the Word Tour reference evaluation. */
  public static final int DEFAULT_RADIUS = 10;

  /** Gaussian denominator used by the Word Tour reference evaluation. */
  public static final double DEFAULT_SIGMA = 10.0;

  private final String orderFingerprint;
  private final int vocabularySize;
  private final int[] ranks;
  private final double[] weights;

  private BlurredBagOfWords(
      String orderFingerprint, int vocabularySize, int[] ranks, double[] weights) {
    this.orderFingerprint = orderFingerprint;
    this.vocabularySize = vocabularySize;
    this.ranks = ranks;
    this.weights = weights;
  }

  /** Encodes terms with the radius and sigma used by the Word Tour reference evaluation. */
  public static BlurredBagOfWords encode(SemanticOrder order, List<String> terms) {
    return encode(order, terms, DEFAULT_RADIUS, DEFAULT_SIGMA);
  }

  /**
   * Encodes terms using weights {@code exp(-(distance^2) / sigma)} and L1 normalization. Unknown
   * terms are ignored.
   */
  public static BlurredBagOfWords encode(
      SemanticOrder order, List<String> terms, int radius, double sigma) {
    Objects.requireNonNull(order, "order");
    Objects.requireNonNull(terms, "terms");
    if (radius < 0) {
      throw new IllegalArgumentException("radius must be >= 0");
    }
    if (!Double.isFinite(sigma) || sigma <= 0.0) {
      throw new IllegalArgumentException("sigma must be finite and > 0");
    }

    Map<Integer, Double> accumulated = new HashMap<>();
    for (String term : terms) {
      Objects.requireNonNull(term, "terms must not contain null");
      OptionalInt rank = order.rank(term);
      if (rank.isEmpty()) {
        continue;
      }
      add(accumulated, rank.getAsInt(), 1.0);
      for (int distance = 1; distance <= radius; distance++) {
        double weight = Math.exp(-((double) distance * distance) / sigma);
        int predecessor = Math.floorMod((long) rank.getAsInt() - distance, order.size());
        int successor = Math.floorMod((long) rank.getAsInt() + distance, order.size());
        add(accumulated, predecessor, weight);
        add(accumulated, successor, weight);
      }
    }

    List<Map.Entry<Integer, Double>> sorted = new ArrayList<>(accumulated.entrySet());
    sorted.sort(Map.Entry.comparingByKey());
    int[] ranks = new int[sorted.size()];
    double[] weights = new double[sorted.size()];
    double total = 0.0;
    for (Map.Entry<Integer, Double> entry : sorted) {
      total += entry.getValue();
    }
    for (int index = 0; index < sorted.size(); index++) {
      Map.Entry<Integer, Double> entry = sorted.get(index);
      ranks[index] = entry.getKey();
      weights[index] = entry.getValue() / total;
    }
    return new BlurredBagOfWords(order.fingerprint(), order.size(), ranks, weights);
  }

  /** Returns the number of nonzero ranks in this sparse representation. */
  public int nonZeroCount() {
    return ranks.length;
  }

  /** Returns the normalized weight at a rank, or zero when the rank is absent. */
  public double weightAtRank(int rank) {
    Objects.checkIndex(rank, vocabularySize);
    int index = Arrays.binarySearch(ranks, rank);
    return index < 0 ? 0.0 : weights[index];
  }

  /** Returns the L1 norm, which is one for a nonempty representation and zero otherwise. */
  public double l1Norm() {
    double result = 0.0;
    for (double weight : weights) {
      result += Math.abs(weight);
    }
    return result;
  }

  /** Returns the sparse Manhattan distance to a representation based on the same exact order. */
  public double l1Distance(BlurredBagOfWords other) {
    Objects.requireNonNull(other, "other");
    if (vocabularySize != other.vocabularySize
        || !orderFingerprint.equals(other.orderFingerprint)) {
      throw new IllegalArgumentException("Representations use a different semantic order");
    }

    int left = 0;
    int right = 0;
    double distance = 0.0;
    while (left < ranks.length && right < other.ranks.length) {
      if (ranks[left] < other.ranks[right]) {
        distance += weights[left++];
      } else if (ranks[left] > other.ranks[right]) {
        distance += other.weights[right++];
      } else {
        distance += Math.abs(weights[left++] - other.weights[right++]);
      }
    }
    while (left < ranks.length) {
      distance += weights[left++];
    }
    while (right < other.ranks.length) {
      distance += other.weights[right++];
    }
    return distance;
  }

  private static void add(Map<Integer, Double> values, int rank, double weight) {
    values.merge(rank, weight, Double::sum);
  }
}
