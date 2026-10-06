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
package com.integrallis.models.bench.fusion;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Numerically careful combination rules and distribution statistics for token-level fusion. */
public final class FusionMath {

  static final double WEIGHT_SUM_TOLERANCE = 1e-6;

  private FusionMath() {}

  /** Returns {@code log_softmax(logits)} in double precision. */
  public static double[] logSoftmax(float[] logits) {
    Objects.requireNonNull(logits, "logits");
    if (logits.length == 0) {
      throw new IllegalArgumentException("logits must not be empty");
    }
    double max = Double.NEGATIVE_INFINITY;
    for (float logit : logits) {
      if (!Float.isFinite(logit)) {
        throw new IllegalArgumentException("logits must be finite");
      }
      max = Math.max(max, logit);
    }
    double sum = 0;
    for (float logit : logits) {
      sum += Math.exp(logit - max);
    }
    double logNormalizer = max + Math.log(sum);
    double[] result = new double[logits.length];
    for (int i = 0; i < logits.length; i++) {
      result[i] = logits[i] - logNormalizer;
    }
    return result;
  }

  /** Returns {@code log_softmax(scores)} for already-combined double scores. */
  public static double[] logSoftmax(double[] scores) {
    double max = Double.NEGATIVE_INFINITY;
    for (double score : scores) {
      max = Math.max(max, score);
    }
    if (!Double.isFinite(max)) {
      throw new IllegalArgumentException("scores must contain a finite maximum");
    }
    double sum = 0;
    for (double score : scores) {
      sum += Math.exp(score - max);
    }
    double logNormalizer = max + Math.log(sum);
    double[] result = new double[scores.length];
    for (int i = 0; i < scores.length; i++) {
      result[i] = scores[i] - logNormalizer;
    }
    return result;
  }

  /**
   * Combines member distributions under one rule.
   *
   * @param rule combination rule
   * @param logits raw member logits, used only by {@link FusionRule#ARTICLE}
   * @param logProbabilities member log-probabilities, used by the log-space rules
   * @param weights simplex weights, one per member
   * @return fused scores {@code s(t)}; not normalized
   */
  public static double[] combine(
      FusionRule rule, float[][] logits, double[][] logProbabilities, double[] weights) {
    Objects.requireNonNull(rule, "rule");
    int members = weights.length;
    if (logits.length != members || logProbabilities.length != members) {
      throw new IllegalArgumentException("one logits row and one weight per member are required");
    }
    int vocabulary = logits[0].length;
    for (int i = 0; i < members; i++) {
      if (logits[i].length != vocabulary || logProbabilities[i].length != vocabulary) {
        throw new IllegalArgumentException(
            "member vocabulary sizes differ: " + logits[i].length + " vs " + vocabulary);
      }
    }
    double[] fused = new double[vocabulary];
    switch (rule) {
      case POE -> {
        for (int i = 0; i < members; i++) {
          double w = weights[i];
          if (w == 0) {
            continue;
          }
          double[] row = logProbabilities[i];
          for (int t = 0; t < vocabulary; t++) {
            fused[t] += w * row[t];
          }
        }
      }
      case ARTICLE -> {
        for (int i = 0; i < members; i++) {
          double w = weights[i];
          if (w == 0) {
            continue;
          }
          float[] row = logits[i];
          for (int t = 0; t < vocabulary; t++) {
            fused[t] += w * row[t];
          }
        }
      }
      case MIXTURE -> {
        List<Integer> active = new ArrayList<>();
        for (int i = 0; i < members; i++) {
          if (weights[i] > 0) {
            active.add(i);
          }
        }
        double[] logWeights = new double[members];
        for (int i : active) {
          logWeights[i] = Math.log(weights[i]);
        }
        for (int t = 0; t < vocabulary; t++) {
          double max = Double.NEGATIVE_INFINITY;
          for (int i : active) {
            max = Math.max(max, logWeights[i] + logProbabilities[i][t]);
          }
          double sum = 0;
          for (int i : active) {
            sum += Math.exp(logWeights[i] + logProbabilities[i][t] - max);
          }
          fused[t] = max + Math.log(sum);
        }
      }
    }
    return fused;
  }

  /** Shannon entropy in nats of a normalized log-probability vector. */
  public static double entropy(double[] logProbabilities) {
    double entropy = 0;
    for (double logP : logProbabilities) {
      if (logP > Double.NEGATIVE_INFINITY) {
        entropy -= Math.exp(logP) * logP;
      }
    }
    return entropy;
  }

  /** {@code KL(p || q)} in nats for normalized log-probability vectors. */
  public static double klDivergence(double[] logP, double[] logQ) {
    if (logP.length != logQ.length) {
      throw new IllegalArgumentException("vocabulary sizes differ");
    }
    double kl = 0;
    for (int t = 0; t < logP.length; t++) {
      if (logP[t] > Double.NEGATIVE_INFINITY) {
        kl += Math.exp(logP[t]) * (logP[t] - logQ[t]);
      }
    }
    return kl;
  }

  /** Index of the largest value; ties resolve to the lowest index. */
  public static int argmax(double[] values) {
    int best = 0;
    for (int i = 1; i < values.length; i++) {
      if (values[i] > values[best]) {
        best = i;
      }
    }
    return best;
  }

  /** Index of the largest value; ties resolve to the lowest index. */
  public static int argmax(float[] values) {
    int best = 0;
    for (int i = 1; i < values.length; i++) {
      if (values[i] > values[best]) {
        best = i;
      }
    }
    return best;
  }

  /** Rejects weights that are not a point on the probability simplex. */
  public static void validateWeights(double[] weights) {
    Objects.requireNonNull(weights, "weights");
    if (weights.length == 0) {
      throw new IllegalArgumentException("at least one weight is required");
    }
    double sum = 0;
    for (double weight : weights) {
      if (!Double.isFinite(weight) || weight < 0) {
        throw new IllegalArgumentException(
            "weights must be finite and non-negative on the simplex");
      }
      sum += weight;
    }
    if (Math.abs(sum - 1.0) > WEIGHT_SUM_TOLERANCE) {
      throw new IllegalArgumentException("weights must lie on the simplex (sum to 1): " + sum);
    }
  }

  /** Every simplex point whose coordinates are multiples of {@code 1/divisions}. */
  public static List<double[]> simplexGrid(int members, int divisions) {
    if (members < 1 || divisions < 1) {
      throw new IllegalArgumentException("members and divisions must be positive");
    }
    List<double[]> points = new ArrayList<>();
    enumerate(new int[members], 0, divisions, divisions, points);
    return points;
  }

  private static void enumerate(
      int[] counts, int index, int remaining, int divisions, List<double[]> points) {
    if (index == counts.length - 1) {
      counts[index] = remaining;
      double[] point = new double[counts.length];
      for (int i = 0; i < counts.length; i++) {
        point[i] = counts[i] / (double) divisions;
      }
      points.add(point);
      return;
    }
    for (int count = remaining; count >= 0; count--) {
      counts[index] = count;
      enumerate(counts, index + 1, remaining - count, divisions, points);
    }
  }
}
