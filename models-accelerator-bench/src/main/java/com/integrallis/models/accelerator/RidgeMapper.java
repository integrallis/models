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

import java.util.Objects;

/** Small Java-only ridge-regression utility for unpublished cache-transfer experiments. */
public final class RidgeMapper {
  private final double[][] coefficients;
  private final double[] bias;
  private final int inputDimensions;
  private final int outputDimensions;

  private RidgeMapper(double[][] coefficients, double[] bias) {
    this.coefficients = coefficients;
    this.bias = bias;
    this.inputDimensions = coefficients.length;
    this.outputDimensions = coefficients[0].length;
  }

  /** Fits {@code Y = XW + b} after centering X and Y. */
  public static RidgeMapper fit(float[][] inputs, float[][] targets, double regularization) {
    return fit(inputs, targets, null, regularization);
  }

  /** Fits weighted {@code Y = XW + b} after rescaling weights to mean one and centering. */
  public static RidgeMapper fit(
      float[][] inputs, float[][] targets, double[] weights, double regularization) {
    Dimensions dimensions = validate(inputs, targets, regularization);
    double[] normalizedWeights = normalizedWeights(weights, inputs.length);
    double[] inputMeans = means(inputs, dimensions.inputs(), normalizedWeights);
    double[] targetMeans = means(targets, dimensions.outputs(), normalizedWeights);
    double[][] gram = new double[dimensions.inputs()][dimensions.inputs()];
    double[][] cross = new double[dimensions.inputs()][dimensions.outputs()];
    for (int sample = 0; sample < inputs.length; sample++) {
      float[] input = inputs[sample];
      float[] target = targets[sample];
      double weight = normalizedWeights[sample];
      for (int row = 0; row < dimensions.inputs(); row++) {
        double x = input[row] - inputMeans[row];
        for (int column = 0; column <= row; column++) {
          gram[row][column] += weight * x * (input[column] - inputMeans[column]);
        }
        for (int output = 0; output < dimensions.outputs(); output++) {
          cross[row][output] += weight * x * (target[output] - targetMeans[output]);
        }
      }
    }
    for (int row = 0; row < dimensions.inputs(); row++) {
      for (int column = 0; column < row; column++) {
        gram[column][row] = gram[row][column];
      }
      gram[row][row] += regularization;
    }
    double[][] coefficients = solveCholesky(gram, cross);
    double[] bias = targetMeans.clone();
    for (int input = 0; input < dimensions.inputs(); input++) {
      for (int output = 0; output < dimensions.outputs(); output++) {
        bias[output] -= inputMeans[input] * coefficients[input][output];
      }
    }
    return new RidgeMapper(coefficients, bias);
  }

  /** Applies the fitted mapping to one matrix of row-major samples. */
  public float[][] predict(float[][] inputs) {
    Objects.requireNonNull(inputs, "inputs");
    float[][] outputs = new float[inputs.length][outputDimensions];
    for (int sample = 0; sample < inputs.length; sample++) {
      float[] input = Objects.requireNonNull(inputs[sample], "inputs[" + sample + "]");
      if (input.length != inputDimensions) {
        throw new IllegalArgumentException(
            "inputs[" + sample + "] length must be " + inputDimensions + ": " + input.length);
      }
      for (int output = 0; output < outputDimensions; output++) {
        double value = bias[output];
        for (int inputIndex = 0; inputIndex < inputDimensions; inputIndex++) {
          value += input[inputIndex] * coefficients[inputIndex][output];
        }
        outputs[sample][output] = (float) value;
      }
    }
    return outputs;
  }

  private static double[] means(float[][] samples, int dimensions, double[] weights) {
    double[] means = new double[dimensions];
    for (int index = 0; index < samples.length; index++) {
      float[] sample = samples[index];
      for (int dimension = 0; dimension < dimensions; dimension++) {
        means[dimension] += weights[index] * sample[dimension];
      }
    }
    for (int dimension = 0; dimension < dimensions; dimension++) {
      means[dimension] /= samples.length;
    }
    return means;
  }

  private static double[] normalizedWeights(double[] weights, int samples) {
    if (weights == null) {
      double[] uniform = new double[samples];
      java.util.Arrays.fill(uniform, 1.0);
      return uniform;
    }
    if (weights.length != samples) {
      throw new IllegalArgumentException(
          "weight count must match training samples: " + weights.length + " != " + samples);
    }
    double sum = 0.0;
    for (int sample = 0; sample < weights.length; sample++) {
      double weight = weights[sample];
      if (!(weight >= 0.0) || !Double.isFinite(weight)) {
        throw new IllegalArgumentException(
            "weights must be finite and nonnegative; weights[" + sample + "]=" + weight);
      }
      sum += weight;
    }
    if (!(sum > 0.0) || !Double.isFinite(sum)) {
      throw new IllegalArgumentException("at least one finite positive weight is required");
    }
    double scale = samples / sum;
    double[] normalized = new double[samples];
    for (int sample = 0; sample < samples; sample++) {
      normalized[sample] = weights[sample] * scale;
    }
    return normalized;
  }

  /** Relative L2 error across two equally shaped sample matrices. */
  public static double relativeL2(float[][] expected, float[][] actual) {
    Objects.requireNonNull(expected, "expected");
    Objects.requireNonNull(actual, "actual");
    if (expected.length != actual.length) {
      throw new IllegalArgumentException("sample counts differ");
    }
    double difference = 0.0;
    double reference = 0.0;
    for (int sample = 0; sample < expected.length; sample++) {
      if (expected[sample].length != actual[sample].length) {
        throw new IllegalArgumentException("sample widths differ at " + sample);
      }
      for (int dimension = 0; dimension < expected[sample].length; dimension++) {
        double target = expected[sample][dimension];
        double delta = target - actual[sample][dimension];
        difference += delta * delta;
        reference += target * target;
      }
    }
    return reference == 0.0 ? Math.sqrt(difference) : Math.sqrt(difference / reference);
  }

  private static Dimensions validate(float[][] inputs, float[][] targets, double regularization) {
    Objects.requireNonNull(inputs, "inputs");
    Objects.requireNonNull(targets, "targets");
    if (inputs.length == 0 || targets.length == 0) {
      throw new IllegalArgumentException("training samples must not be empty");
    }
    if (inputs.length != targets.length) {
      throw new IllegalArgumentException("input and target sample counts differ");
    }
    if (!(regularization >= 0.0) || !Double.isFinite(regularization)) {
      throw new IllegalArgumentException("regularization must be finite and >= 0");
    }
    int inputDimensions = requireWidth(inputs, "inputs");
    int outputDimensions = requireWidth(targets, "targets");
    return new Dimensions(inputDimensions, outputDimensions);
  }

  private static int requireWidth(float[][] samples, String name) {
    int width = Objects.requireNonNull(samples[0], name + "[0]").length;
    if (width == 0) {
      throw new IllegalArgumentException(name + " width must be positive");
    }
    for (int sample = 1; sample < samples.length; sample++) {
      float[] values = Objects.requireNonNull(samples[sample], name + "[" + sample + "]");
      if (values.length != width) {
        throw new IllegalArgumentException(name + " rows have different widths");
      }
    }
    return width;
  }

  private static double[][] solveCholesky(double[][] matrix, double[][] rightHandSides) {
    int size = matrix.length;
    double[][] lower = new double[size][size];
    for (int row = 0; row < size; row++) {
      for (int column = 0; column <= row; column++) {
        double value = matrix[row][column];
        for (int inner = 0; inner < column; inner++) {
          value -= lower[row][inner] * lower[column][inner];
        }
        if (row == column) {
          if (!(value > 0.0) || !Double.isFinite(value)) {
            throw new IllegalArgumentException(
                "regularized covariance is not positive definite at " + row);
          }
          lower[row][column] = Math.sqrt(value);
        } else {
          lower[row][column] = value / lower[column][column];
        }
      }
    }

    int outputs = rightHandSides[0].length;
    double[][] solution = new double[size][outputs];
    double[] intermediate = new double[size];
    for (int output = 0; output < outputs; output++) {
      for (int row = 0; row < size; row++) {
        double value = rightHandSides[row][output];
        for (int inner = 0; inner < row; inner++) {
          value -= lower[row][inner] * intermediate[inner];
        }
        intermediate[row] = value / lower[row][row];
      }
      for (int row = size - 1; row >= 0; row--) {
        double value = intermediate[row];
        for (int inner = row + 1; inner < size; inner++) {
          value -= lower[inner][row] * solution[inner][output];
        }
        solution[row][output] = value / lower[row][row];
      }
    }
    return solution;
  }

  private record Dimensions(int inputs, int outputs) {}
}
