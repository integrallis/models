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
 * Reference head-local cross-model KV mapper for aligned grouped-query-attention models.
 *
 * <p>Each target head reads only the same architecture-assigned source head from the selected
 * source layers. This bounds coefficient storage and application work by avoiding all-to-all head
 * fan-in while retaining predictive cross-layer selection.
 */
public final class HeadLocalPredictiveKvMapper {
  private static final double MIN_OBSERVATIONS_PER_PARAMETER = 4.0;

  /** Per-target-layer, per-target-head token sensitivities for K and V fitting. */
  public record AttentionSensitivities(double[][][] keys, double[][][] values) {
    public AttentionSensitivities {
      Objects.requireNonNull(keys, "keys");
      Objects.requireNonNull(values, "values");
    }
  }

  /** Mapped, RoPE-free keys and values indexed by target layer and observation. */
  public record MappedCache(float[][][] keys, float[][][] values) {
    public MappedCache {
      Objects.requireNonNull(keys, "keys");
      Objects.requireNonNull(values, "values");
    }
  }

  private record LayerMapper(
      int[] sourceLayers, RidgeMapper[] keyHeads, RidgeMapper[] valueHeads) {
    private LayerMapper {
      sourceLayers = sourceLayers.clone();
      keyHeads = keyHeads.clone();
      valueHeads = valueHeads.clone();
    }
  }

  private record Shape(int layers, int observations, int dimensions) {}

  private record LayerScore(int layer, double score) {}

  private final LayerMapper[] layers;
  private final int heads;
  private final int headDimension;

  private HeadLocalPredictiveKvMapper(
      LayerMapper[] layers, int heads, int headDimension) {
    this.layers = layers;
    this.heads = heads;
    this.headDimension = headDimension;
  }

  /** Fits one mapper from disjoint fitting and source-layer-selection observations. */
  public static HeadLocalPredictiveKvMapper fit(
      float[][][] sourceKeys,
      float[][][] sourceValues,
      float[][][] targetKeys,
      float[][][] targetValues,
      float[][][] selectionSourceKeys,
      float[][][] selectionSourceValues,
      float[][][] selectionTargetKeys,
      float[][][] selectionTargetValues,
      int heads,
      int headDimension,
      int topK,
      double regularization) {
    return fit(
        sourceKeys,
        sourceValues,
        targetKeys,
        targetValues,
        selectionSourceKeys,
        selectionSourceValues,
        selectionTargetKeys,
        selectionTargetValues,
        heads,
        headDimension,
        topK,
        regularization,
        null);
  }

  /** Fits a head-local mapper with receiver-attention-weighted calibration observations. */
  public static HeadLocalPredictiveKvMapper fit(
      float[][][] sourceKeys,
      float[][][] sourceValues,
      float[][][] targetKeys,
      float[][][] targetValues,
      float[][][] selectionSourceKeys,
      float[][][] selectionSourceValues,
      float[][][] selectionTargetKeys,
      float[][][] selectionTargetValues,
      int heads,
      int headDimension,
      int topK,
      double regularization,
      AttentionSensitivities sensitivities) {
    int dimensions = Math.multiplyExact(heads, headDimension);
    Shape source = validate(sourceKeys, sourceValues, dimensions, "fit source");
    Shape target = validate(targetKeys, targetValues, dimensions, "fit target");
    Shape selectionSource =
        validate(selectionSourceKeys, selectionSourceValues, dimensions, "selection source");
    Shape selectionTarget =
        validate(selectionTargetKeys, selectionTargetValues, dimensions, "selection target");
    if (source.observations() != target.observations()
        || source.layers() != selectionSource.layers()
        || target.layers() != selectionTarget.layers()
        || selectionSource.observations() != selectionTarget.observations()) {
      throw new IllegalArgumentException("source and target cache shapes are not aligned");
    }
    if (topK < 1 || topK > source.layers()) {
      throw new IllegalArgumentException(
          "topK must be between 1 and " + source.layers() + ": " + topK);
    }
    int parameters = Math.multiplyExact(topK, headDimension);
    if (source.observations() / (double) parameters < MIN_OBSERVATIONS_PER_PARAMETER) {
      throw new IllegalArgumentException(
          "production fit needs at least "
              + MIN_OBSERVATIONS_PER_PARAMETER
              + " observations per parameter");
    }
    validateSensitivities(sensitivities, target.layers(), heads, source.observations());

    LayerMapper[] mappings = new LayerMapper[target.layers()];
    for (int targetLayer = 0; targetLayer < target.layers(); targetLayer++) {
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
              heads,
              headDimension,
              topK,
              regularization,
              sensitivities,
              targetLayer,
              parameters);
      RidgeMapper[] keyHeads = new RidgeMapper[heads];
      RidgeMapper[] valueHeads = new RidgeMapper[heads];
      for (int head = 0; head < heads; head++) {
        keyHeads[head] =
            RidgeMapper.fit(
                concatenateHead(sourceKeys, selected, head, headDimension),
                headRows(targetKeys[targetLayer], head, headDimension),
                weightsFor(sensitivities, true, targetLayer, head, parameters),
                regularization);
        valueHeads[head] =
            RidgeMapper.fit(
                concatenateHead(sourceValues, selected, head, headDimension),
                headRows(targetValues[targetLayer], head, headDimension),
                weightsFor(sensitivities, false, targetLayer, head, parameters),
                regularization);
      }
      mappings[targetLayer] = new LayerMapper(selected, keyHeads, valueHeads);
    }
    return new HeadLocalPredictiveKvMapper(mappings, heads, headDimension);
  }

  /** Applies the fitted head-local maps in content coordinates. */
  public MappedCache map(float[][][] sourceKeys, float[][][] sourceValues) {
    int dimensions = Math.multiplyExact(heads, headDimension);
    Shape source = validate(sourceKeys, sourceValues, dimensions, "mapping source");
    int requiredLayers =
        Arrays.stream(layers)
                .flatMapToInt(layer -> Arrays.stream(layer.sourceLayers()))
                .max()
                .orElseThrow()
            + 1;
    if (source.layers() < requiredLayers) {
      throw new IllegalArgumentException(
          "mapping source has " + source.layers() + " layers; needs " + requiredLayers);
    }

    float[][][] keys = new float[layers.length][source.observations()][dimensions];
    float[][][] values = new float[layers.length][source.observations()][dimensions];
    for (int targetLayer = 0; targetLayer < layers.length; targetLayer++) {
      LayerMapper mapper = layers[targetLayer];
      for (int head = 0; head < heads; head++) {
        insertHead(
            keys[targetLayer],
            head,
            mapper.keyHeads()[head].predict(
                concatenateHead(sourceKeys, mapper.sourceLayers(), head, headDimension)),
            headDimension);
        insertHead(
            values[targetLayer],
            head,
            mapper.valueHeads()[head].predict(
                concatenateHead(sourceValues, mapper.sourceLayers(), head, headDimension)),
            headDimension);
      }
    }
    return new MappedCache(keys, values);
  }

  /** Returns the selected source layers for one target layer. */
  public int[] sourceLayers(int targetLayer) {
    return layers[targetLayer].sourceLayers().clone();
  }

  private static int[] selectLayers(
      float[][][] fitSourceKeys,
      float[][][] fitSourceValues,
      float[][] fitTargetKeys,
      float[][] fitTargetValues,
      float[][][] selectionSourceKeys,
      float[][][] selectionSourceValues,
      float[][] selectionTargetKeys,
      float[][] selectionTargetValues,
      int heads,
      int headDimension,
      int topK,
      double regularization,
      AttentionSensitivities sensitivities,
      int targetLayer,
      int parameters) {
    LayerScore[] scores = new LayerScore[fitSourceKeys.length];
    for (int sourceLayer = 0; sourceLayer < scores.length; sourceLayer++) {
      double total = 0.0;
      for (int head = 0; head < heads; head++) {
        RidgeMapper keyProbe =
            RidgeMapper.fit(
                headRows(fitSourceKeys[sourceLayer], head, headDimension),
                headRows(fitTargetKeys, head, headDimension),
                weightsFor(sensitivities, true, targetLayer, head, parameters),
                regularization);
        RidgeMapper valueProbe =
            RidgeMapper.fit(
                headRows(fitSourceValues[sourceLayer], head, headDimension),
                headRows(fitTargetValues, head, headDimension),
                weightsFor(sensitivities, false, targetLayer, head, parameters),
                regularization);
        total +=
            r2(
                keyProbe.predict(
                    headRows(selectionSourceKeys[sourceLayer], head, headDimension)),
                headRows(selectionTargetKeys, head, headDimension));
        total +=
            r2(
                valueProbe.predict(
                    headRows(selectionSourceValues[sourceLayer], head, headDimension)),
                headRows(selectionTargetValues, head, headDimension));
      }
      scores[sourceLayer] = new LayerScore(sourceLayer, total / (heads * 2.0));
    }
    return Arrays.stream(scores)
        .sorted(Comparator.comparingDouble(LayerScore::score).reversed())
        .limit(topK)
        .mapToInt(LayerScore::layer)
        .toArray();
  }

  private static double[] weightsFor(
      AttentionSensitivities sensitivities,
      boolean keys,
      int targetLayer,
      int head,
      int parameters) {
    if (sensitivities == null) {
      return null;
    }
    double[] raw = (keys ? sensitivities.keys() : sensitivities.values())[targetLayer][head];
    return AttentionWeighting.shrinkToEffectiveSampleSize(raw, parameters);
  }

  private static void validateSensitivities(
      AttentionSensitivities sensitivities, int targetLayers, int heads, int observations) {
    if (sensitivities == null) {
      return;
    }
    validateSensitivityKind(sensitivities.keys(), "key", targetLayers, heads, observations);
    validateSensitivityKind(sensitivities.values(), "value", targetLayers, heads, observations);
  }

  private static void validateSensitivityKind(
      double[][][] weights, String label, int targetLayers, int heads, int observations) {
    if (weights.length != targetLayers) {
      throw new IllegalArgumentException(label + " sensitivity target-layer count differs");
    }
    for (int layer = 0; layer < targetLayers; layer++) {
      if (weights[layer] == null || weights[layer].length != heads) {
        throw new IllegalArgumentException(label + " sensitivity head count differs at " + layer);
      }
      for (int head = 0; head < heads; head++) {
        if (weights[layer][head] == null || weights[layer][head].length != observations) {
          throw new IllegalArgumentException(
              label + " sensitivity observation count differs at " + layer + "/" + head);
        }
      }
    }
  }

  private static double r2(float[][] predicted, float[][] target) {
    double[] means = new double[target[0].length];
    for (float[] row : target) {
      for (int dimension = 0; dimension < means.length; dimension++) {
        means[dimension] += row[dimension];
      }
    }
    for (int dimension = 0; dimension < means.length; dimension++) {
      means[dimension] /= target.length;
    }
    double residual = 0.0;
    double total = 0.0;
    for (int observation = 0; observation < target.length; observation++) {
      for (int dimension = 0; dimension < means.length; dimension++) {
        double delta = target[observation][dimension] - predicted[observation][dimension];
        double centered = target[observation][dimension] - means[dimension];
        residual += delta * delta;
        total += centered * centered;
      }
    }
    return total == 0.0
        ? (residual == 0.0 ? 1.0 : Double.NEGATIVE_INFINITY)
        : 1.0 - residual / total;
  }

  private static float[][] concatenateHead(
      float[][][] layers, int[] selected, int head, int headDimension) {
    float[][] result = new float[layers[0].length][selected.length * headDimension];
    int sourceOffset = head * headDimension;
    for (int observation = 0; observation < result.length; observation++) {
      for (int selection = 0; selection < selected.length; selection++) {
        System.arraycopy(
            layers[selected[selection]][observation],
            sourceOffset,
            result[observation],
            selection * headDimension,
            headDimension);
      }
    }
    return result;
  }

  private static float[][] headRows(float[][] observations, int head, int headDimension) {
    float[][] result = new float[observations.length][headDimension];
    int offset = head * headDimension;
    for (int observation = 0; observation < observations.length; observation++) {
      System.arraycopy(observations[observation], offset, result[observation], 0, headDimension);
    }
    return result;
  }

  private static void insertHead(
      float[][] destination, int head, float[][] values, int headDimension) {
    int offset = head * headDimension;
    for (int observation = 0; observation < destination.length; observation++) {
      System.arraycopy(values[observation], 0, destination[observation], offset, headDimension);
    }
  }

  private static Shape validate(
      float[][][] keys, float[][][] values, int dimensions, String label) {
    Objects.requireNonNull(keys, label + " keys");
    Objects.requireNonNull(values, label + " values");
    if (keys.length == 0 || keys.length != values.length) {
      throw new IllegalArgumentException(label + " layer counts must match and be positive");
    }
    int observations = rows(keys[0], dimensions, label + " keys[0]");
    if (observations == 0) {
      throw new IllegalArgumentException(label + " observations must be positive");
    }
    for (int layer = 0; layer < keys.length; layer++) {
      if (rows(keys[layer], dimensions, label + " keys[" + layer + "]") != observations
          || rows(values[layer], dimensions, label + " values[" + layer + "]")
              != observations) {
        throw new IllegalArgumentException(label + " observation counts differ by layer");
      }
    }
    return new Shape(keys.length, observations, dimensions);
  }

  private static int rows(float[][] values, int dimensions, String label) {
    Objects.requireNonNull(values, label);
    for (int row = 0; row < values.length; row++) {
      if (Objects.requireNonNull(values[row], label + "[" + row + "]").length != dimensions) {
        throw new IllegalArgumentException(label + " row " + row + " has the wrong width");
      }
    }
    return values.length;
  }
}
