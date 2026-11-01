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
package com.integrallis.models.accelerator;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Objects;

/**
 * In-memory reference implementation of predictive cross-model KV mapping.
 *
 * <p>Each target layer selects source layers by held-out, head-averaged K/V R2. Production maps
 * concatenate every source KV head from the selected layers, allowing any source head to predict
 * any target head. Fitting uses centered affine ridge regression.
 */
public final class PredictiveKvMapper {
  private static final double MIN_OBSERVATIONS_PER_PARAMETER = 4.0;

  /** Mapped, RoPE-free keys and values indexed by target layer, observation, and KV feature. */
  public record MappedCache(float[][][] keys, float[][][] values) {
    public MappedCache {
      Objects.requireNonNull(keys, "keys");
      Objects.requireNonNull(values, "values");
    }
  }

  private record LayerMapper(int[] sourceLayers, RidgeMapper keys, RidgeMapper values) {
    private LayerMapper {
      sourceLayers = sourceLayers.clone();
      Objects.requireNonNull(keys, "keys");
      Objects.requireNonNull(values, "values");
    }
  }

  private final LayerMapper[] layers;
  private final int sourceDimensions;

  private PredictiveKvMapper(LayerMapper[] layers, int sourceDimensions) {
    this.layers = layers;
    this.sourceDimensions = sourceDimensions;
  }

  /** Fits one complete source-to-target cache mapper from disjoint fit and selection sets. */
  public static PredictiveKvMapper fit(
      float[][][] sourceKeys,
      float[][][] sourceValues,
      float[][][] targetKeys,
      float[][][] targetValues,
      float[][][] selectionSourceKeys,
      float[][][] selectionSourceValues,
      float[][][] selectionTargetKeys,
      float[][][] selectionTargetValues,
      int kvHeads,
      int headDimension,
      int topK,
      double regularization) {
    Shape fit =
        validatePair(
            sourceKeys,
            sourceValues,
            targetKeys,
            targetValues,
            kvHeads,
            headDimension,
            "fit");
    Shape selection =
        validatePair(
            selectionSourceKeys,
            selectionSourceValues,
            selectionTargetKeys,
            selectionTargetValues,
            kvHeads,
            headDimension,
            "selection");
    if (fit.sourceLayers() != selection.sourceLayers()
        || fit.targetLayers() != selection.targetLayers()
        || fit.dimensions() != selection.dimensions()) {
      throw new IllegalArgumentException("fit and selection cache shapes differ");
    }
    if (topK < 1 || topK > fit.sourceLayers()) {
      throw new IllegalArgumentException(
          "topK must be between 1 and " + fit.sourceLayers() + ": " + topK);
    }
    int parameters = Math.multiplyExact(topK, fit.dimensions());
    double ratio = fit.observations() / (double) parameters;
    if (ratio < MIN_OBSERVATIONS_PER_PARAMETER) {
      throw new IllegalArgumentException(
          "production fit needs at least "
              + MIN_OBSERVATIONS_PER_PARAMETER
              + " observations per parameter; got "
              + ratio);
    }

    LayerMapper[] mappings = new LayerMapper[fit.targetLayers()];
    for (int targetLayer = 0; targetLayer < mappings.length; targetLayer++) {
      int[] selected =
          selectLayers(
              sourceKeys,
              sourceValues,
              targetKeys[targetLayer],
              targetValues[targetLayer],
              selectionSourceKeys,
              selectionSourceValues,
              selectionTargetKeys[targetLayer],
              selectionTargetValues[targetLayer],
              kvHeads,
              headDimension,
              topK,
              regularization);
      RidgeMapper keyMapper =
          RidgeMapper.fit(concatenate(sourceKeys, selected), targetKeys[targetLayer], regularization);
      RidgeMapper valueMapper =
          RidgeMapper.fit(
              concatenate(sourceValues, selected), targetValues[targetLayer], regularization);
      mappings[targetLayer] = new LayerMapper(selected, keyMapper, valueMapper);
    }
    return new PredictiveKvMapper(mappings, fit.dimensions());
  }

  /** Applies every target-layer mapping in content space. */
  public MappedCache map(float[][][] sourceKeys, float[][][] sourceValues) {
    Shape source = validateSource(sourceKeys, sourceValues, sourceDimensions, "mapping source");
    int requiredSourceLayers =
        Arrays.stream(layers)
            .flatMapToInt(layer -> Arrays.stream(layer.sourceLayers()))
            .max()
            .orElseThrow()
            + 1;
    if (source.sourceLayers() < requiredSourceLayers) {
      throw new IllegalArgumentException(
          "mapping source has "
              + source.sourceLayers()
              + " layers; needs "
              + requiredSourceLayers);
    }
    float[][][] keys = new float[layers.length][][];
    float[][][] values = new float[layers.length][][];
    for (int targetLayer = 0; targetLayer < layers.length; targetLayer++) {
      LayerMapper mapper = layers[targetLayer];
      keys[targetLayer] = mapper.keys().predict(concatenate(sourceKeys, mapper.sourceLayers()));
      values[targetLayer] =
          mapper.values().predict(concatenate(sourceValues, mapper.sourceLayers()));
    }
    return new MappedCache(keys, values);
  }

  /** Returns the selected source layers for one target layer. */
  public int[] sourceLayers(int targetLayer) {
    return layers[targetLayer].sourceLayers().clone();
  }

  /** Mean coefficient of determination across KV heads. */
  public static double meanPerHeadR2(
      float[][] predicted, float[][] target, int heads, int headDimension) {
    requireRows(predicted, heads * headDimension, "predicted");
    requireRows(target, heads * headDimension, "target");
    if (predicted.length != target.length) {
      throw new IllegalArgumentException("predicted and target observation counts differ");
    }
    double score = 0.0;
    for (int head = 0; head < heads; head++) {
      double[] means = new double[headDimension];
      for (float[] row : target) {
        for (int dimension = 0; dimension < headDimension; dimension++) {
          means[dimension] += row[head * headDimension + dimension];
        }
      }
      for (int dimension = 0; dimension < headDimension; dimension++) {
        means[dimension] /= target.length;
      }
      double residual = 0.0;
      double total = 0.0;
      for (int row = 0; row < target.length; row++) {
        for (int dimension = 0; dimension < headDimension; dimension++) {
          int index = head * headDimension + dimension;
          double error = target[row][index] - predicted[row][index];
          double centered = target[row][index] - means[dimension];
          residual += error * error;
          total += centered * centered;
        }
      }
      score += total == 0.0 ? (residual == 0.0 ? 1.0 : Double.NEGATIVE_INFINITY) : 1.0 - residual / total;
    }
    return score / heads;
  }

  private static int[] selectLayers(
      float[][][] fitSourceKeys,
      float[][][] fitSourceValues,
      float[][] fitTargetKeys,
      float[][] fitTargetValues,
      float[][][] heldOutSourceKeys,
      float[][][] heldOutSourceValues,
      float[][] heldOutTargetKeys,
      float[][] heldOutTargetValues,
      int heads,
      int headDimension,
      int topK,
      double regularization) {
    LayerScore[] scores = new LayerScore[fitSourceKeys.length];
    for (int sourceLayer = 0; sourceLayer < scores.length; sourceLayer++) {
      RidgeMapper keyProbe =
          RidgeMapper.fit(fitSourceKeys[sourceLayer], fitTargetKeys, regularization);
      RidgeMapper valueProbe =
          RidgeMapper.fit(fitSourceValues[sourceLayer], fitTargetValues, regularization);
      double keyR2 =
          meanPerHeadR2(
              keyProbe.predict(heldOutSourceKeys[sourceLayer]),
              heldOutTargetKeys,
              heads,
              headDimension);
      double valueR2 =
          meanPerHeadR2(
              valueProbe.predict(heldOutSourceValues[sourceLayer]),
              heldOutTargetValues,
              heads,
              headDimension);
      scores[sourceLayer] = new LayerScore(sourceLayer, (keyR2 + valueR2) / 2.0);
    }
    return Arrays.stream(scores)
        .sorted(Comparator.comparingDouble(LayerScore::score).reversed())
        .limit(topK)
        .mapToInt(LayerScore::layer)
        .toArray();
  }

  private static float[][] concatenate(float[][][] layers, int[] selected) {
    int observations = layers[selected[0]].length;
    int dimensions = layers[selected[0]][0].length;
    float[][] result = new float[observations][Math.multiplyExact(dimensions, selected.length)];
    for (int observation = 0; observation < observations; observation++) {
      for (int selection = 0; selection < selected.length; selection++) {
        float[] source = layers[selected[selection]][observation];
        System.arraycopy(source, 0, result[observation], selection * dimensions, dimensions);
      }
    }
    return result;
  }

  private static Shape validatePair(
      float[][][] sourceKeys,
      float[][][] sourceValues,
      float[][][] targetKeys,
      float[][][] targetValues,
      int heads,
      int headDimension,
      String label) {
    if (heads <= 0 || headDimension <= 0) {
      throw new IllegalArgumentException("KV head geometry must be positive");
    }
    int dimensions = Math.multiplyExact(heads, headDimension);
    Shape source = validateSource(sourceKeys, sourceValues, dimensions, label + " source");
    Shape target = validateSource(targetKeys, targetValues, dimensions, label + " target");
    if (source.observations() != target.observations()) {
      throw new IllegalArgumentException(label + " source and target observation counts differ");
    }
    return new Shape(
        source.sourceLayers(), target.sourceLayers(), source.observations(), dimensions);
  }

  private static Shape validateSource(
      float[][][] keys, float[][][] values, int dimensions, String label) {
    Objects.requireNonNull(keys, label + " keys");
    Objects.requireNonNull(values, label + " values");
    if (keys.length == 0 || keys.length != values.length) {
      throw new IllegalArgumentException(label + " layer counts must match and be positive");
    }
    int observations = requireRows(keys[0], dimensions, label + " keys[0]");
    if (observations == 0) {
      throw new IllegalArgumentException(label + " observations must be positive");
    }
    for (int layer = 0; layer < keys.length; layer++) {
      if (requireRows(keys[layer], dimensions, label + " keys[" + layer + "]") != observations
          || requireRows(values[layer], dimensions, label + " values[" + layer + "]")
              != observations) {
        throw new IllegalArgumentException(label + " observation counts differ by layer");
      }
    }
    return new Shape(keys.length, 0, observations, dimensions);
  }

  private static int requireRows(float[][] rows, int dimensions, String label) {
    Objects.requireNonNull(rows, label);
    for (int row = 0; row < rows.length; row++) {
      if (Objects.requireNonNull(rows[row], label + "[" + row + "]").length != dimensions) {
        throw new IllegalArgumentException(label + " row " + row + " has the wrong width");
      }
    }
    return rows.length;
  }

  private record Shape(int sourceLayers, int targetLayers, int observations, int dimensions) {}

  private record LayerScore(int layer, double score) {}
}
