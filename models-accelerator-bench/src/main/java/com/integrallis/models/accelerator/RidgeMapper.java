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
  private final int inputDimensions;
  private final int outputDimensions;

  private RidgeMapper(double[][] coefficients) {
    this.coefficients = coefficients;
    this.inputDimensions = coefficients.length;
    this.outputDimensions = coefficients[0].length;
  }

  /** Fits {@code Y = XW} using {@code W = (X'X + lambda I)^-1 X'Y}. */
  public static RidgeMapper fit(float[][] inputs, float[][] targets, double regularization) {
    Dimensions dimensions = validate(inputs, targets, regularization);
    double[][] gram = new double[dimensions.inputs()][dimensions.inputs()];
    double[][] cross = new double[dimensions.inputs()][dimensions.outputs()];
    for (int sample = 0; sample < inputs.length; sample++) {
      float[] input = inputs[sample];
      float[] target = targets[sample];
      for (int row = 0; row < dimensions.inputs(); row++) {
        double x = input[row];
        for (int column = 0; column <= row; column++) {
          gram[row][column] += x * input[column];
        }
        for (int output = 0; output < dimensions.outputs(); output++) {
          cross[row][output] += x * target[output];
        }
      }
    }
    for (int row = 0; row < dimensions.inputs(); row++) {
      for (int column = 0; column < row; column++) {
        gram[column][row] = gram[row][column];
      }
      gram[row][row] += regularization;
    }
    return new RidgeMapper(solveCholesky(gram, cross));
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
        double value = 0.0;
        for (int inputIndex = 0; inputIndex < inputDimensions; inputIndex++) {
          value += input[inputIndex] * coefficients[inputIndex][output];
        }
        outputs[sample][output] = (float) value;
      }
    }
    return outputs;
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
