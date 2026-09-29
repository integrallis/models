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
package com.integrallis.models.backend.purejava.gemma3n;

/**
 * The arithmetic of Gemma 3n's AltUp, which carries several parallel residual streams.
 *
 * <p>Separated from the graph and kept free of any weight loading, because every mistake available
 * here produces finite, plausible logits rather than an error: a transposed coefficient block, a
 * missing {@code +1}, a correction applied to the wrong stream. Those are testable as pure
 * arithmetic, and they are tested that way.
 *
 * <p>Streams live in one flat array, stream {@code a} occupying {@code [a * dim, (a + 1) * dim)}.
 * One array rather than an array of arrays: the whole state is handed to matmuls and residual adds
 * as slices, and a ragged structure would mean copying at every boundary.
 *
 * <p>Transcribed from {@code llama.cpp/src/models/gemma3n.cpp}. Where a choice there is not implied
 * by the tensor names it is called out below.
 */
final class Gemma3nAltUp {

  private Gemma3nAltUp() {}

  /** The L2 norm of one stream, which is what the reference's {@code calc_magnitude} computes. */
  static float magnitude(float[] values, int offset, int length) {
    double total = 0.0;
    for (int index = offset; index < offset + length; index++) {
      double value = values[index];
      total += value * value;
    }
    return (float) Math.sqrt(total);
  }

  /**
   * Rescales one stream so its magnitude matches {@code target}.
   *
   * <p>Used when the extra streams are created from the active one and again when they are merged
   * back. A zero magnitude leaves the stream alone rather than dividing by zero: that only arises
   * for an all-zero projection, where every scaling is equally meaningless.
   */
  static void rescaleToMagnitude(float[] values, int offset, int length, float target) {
    float current = magnitude(values, offset, length);
    if (current == 0.0f) {
      return;
    }
    float factor = target / current;
    for (int index = offset; index < offset + length; index++) {
      values[index] *= factor;
    }
  }

  /** Elementwise tanh, which is what turns the router projection into modalities. */
  static void tanhInPlace(float[] values, int length) {
    for (int index = 0; index < length; index++) {
      values[index] = (float) Math.tanh(values[index]);
    }
  }

  /**
   * The predict step: every stream gains a learned mixture of all the streams.
   *
   * <p><b>The coefficient layout is the trap.</b> The reference reshapes the {@code altup * altup}
   * projection outputs to {@code [altup, altup]} with the <i>first</i> dimension fastest, and then
   * multiplies so that the output for stream {@code a} reads the coefficients at flat indices
   * {@code [a * altup, (a + 1) * altup)}. Reading it the other way round is a transpose: it mixes
   * the streams with each other's coefficients and produces perfectly plausible output.
   *
   * @param predictions receives {@code altup * dim} values; may not alias {@code streams}
   * @param streams the current state
   * @param coefficients {@code altup * altup} projection outputs
   */
  static void predict(
      float[] predictions, float[] streams, float[] coefficients, int altup, int dim) {
    for (int stream = 0; stream < altup; stream++) {
      int destination = stream * dim;
      int coefficientBase = stream * altup;
      // Starts from this stream's own value: the reference adds `cur` back after the mixture.
      System.arraycopy(streams, destination, predictions, destination, dim);
      for (int source = 0; source < altup; source++) {
        float weight = coefficients[coefficientBase + source];
        if (weight == 0.0f) {
          continue;
        }
        int origin = source * dim;
        for (int index = 0; index < dim; index++) {
          predictions[destination + index] += weight * streams[origin + index];
        }
      }
    }
  }

  /**
   * The correct step: the difference between what the layer produced and what was predicted for the
   * active stream is fed back into every stream, scaled per stream.
   *
   * <p>The coefficients have <b>1.0 added</b> before use, so a zero projection leaves the
   * innovation applied at unit weight rather than discarded. Omitting that is a silent change of
   * behaviour, not an error.
   *
   * @param streams receives the corrected state, and may be the same array as {@code predictions}
   * @param predictions the predict step's output
   * @param activated the active stream after attention and the feed-forward
   * @param coefficients {@code altup} projection outputs, before the {@code +1}
   * @param activeIndex which stream the layer actually computed
   */
  static void correct(
      float[] streams,
      float[] predictions,
      float[] activated,
      float[] coefficients,
      int altup,
      int dim,
      int activeIndex) {
    int activeOffset = activeIndex * dim;
    // The inactive streams first, the active one last. Every stream's innovation is measured
    // against
    // the ACTIVE prediction, so writing that stream first would leave the remaining streams
    // measuring against their own corrected value -- and callers do pass the same array for both,
    // to
    // avoid a per-token copy of the whole state. Ordering makes that safe without a scratch buffer.
    for (int stream = 0; stream < altup; stream++) {
      if (stream != activeIndex) {
        correctStream(streams, predictions, activated, coefficients, dim, stream, activeOffset);
      }
    }
    correctStream(streams, predictions, activated, coefficients, dim, activeIndex, activeOffset);
  }

  private static void correctStream(
      float[] streams,
      float[] predictions,
      float[] activated,
      float[] coefficients,
      int dim,
      int stream,
      int activeOffset) {
    float weight = coefficients[stream] + 1.0f;
    int destination = stream * dim;
    for (int index = 0; index < dim; index++) {
      float innovation = activated[index] - predictions[activeOffset + index];
      streams[destination + index] = predictions[destination + index] + innovation * weight;
    }
  }

  /**
   * Adds the per-layer input contribution to every stream <b>except the active one</b>.
   *
   * <p>The reference is explicit about this -- it concatenates the untouched first slice back onto
   * the modified rest -- and the asymmetry is the point: the active stream already carries the
   * layer's output.
   */
  static void addToInactiveStreams(
      float[] streams, float[] contribution, int altup, int dim, int activeIndex) {
    for (int stream = 0; stream < altup; stream++) {
      if (stream == activeIndex) {
        continue;
      }
      int destination = stream * dim;
      for (int index = 0; index < dim; index++) {
        streams[destination + index] += contribution[index];
      }
    }
  }

  /**
   * Activation sparsity: zeroes everything below a Gaussian cutoff.
   *
   * <p>{@code relu(x - (mean + std * multiplier))}, with the standard deviation taken over the same
   * vector using the {@code n - 1} denominator, as the reference does. The published multiplier for
   * the early layers is 1.6448533535003662, which is the 95th-percentile z-score -- so this keeps
   * roughly the top 5% of activations and shifts them down by the cutoff.
   *
   * <p>Applied <b>before</b> the GELU, not after.
   */
  static void gaussianTopk(float[] values, int length, float standardDeviationMultiplier) {
    if (length < 2) {
      return;
    }
    double total = 0.0;
    for (int index = 0; index < length; index++) {
      total += values[index];
    }
    float mean = (float) (total / length);
    double variance = 0.0;
    for (int index = 0; index < length; index++) {
      double centred = values[index] - mean;
      variance += centred * centred;
    }
    float deviation = (float) Math.sqrt(variance / (length - 1));
    float cutoff = mean + deviation * standardDeviationMultiplier;
    for (int index = 0; index < length; index++) {
      values[index] = Math.max(0.0f, values[index] - cutoff);
    }
  }

  /**
   * Merges the streams back to one, which is the last thing the graph does before the output norm.
   *
   * <p>A mean over all {@code altup} streams, where the inactive ones are first projected through
   * {@code altup_unembd_proj} and rescaled to the active stream's magnitude. The active stream
   * itself is taken as-is -- it is not projected.
   *
   * @param merged receives {@code dim} values
   * @param activeStream the active stream, unprojected
   * @param projected the {@code altup - 1} projected inactive streams, already rescaled
   */
  static void merge(float[] merged, float[] activeStream, float[] projected, int altup, int dim) {
    float scale = 1.0f / altup;
    for (int index = 0; index < dim; index++) {
      float total = activeStream[index];
      for (int stream = 0; stream < altup - 1; stream++) {
        total += projected[stream * dim + index];
      }
      merged[index] = total * scale;
    }
  }
}
