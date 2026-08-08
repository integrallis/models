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
package com.integrallis.models.router;

import com.integrallis.models.api.catalog.DiscoveredModel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Fills in throughput for models nobody has benchmarked on this machine.
 *
 * <p>Leaving them out is worse than estimating: a model absent from the fleet cannot be chosen at
 * all, and the usual reason it lacks a profile is that it is newly installed rather than unusable.
 *
 * <p>The estimate is calibrated rather than assumed wherever it can be. Local generation is bounded
 * by memory bandwidth — every token reads the whole weight file — so {@code tokensPerSecond ×
 * sizeBytes} is roughly constant across models on one machine. Any measured peer therefore fixes
 * that constant for this hardware, and a model's size predicts its rate from there. That is a
 * property of how inference works, not a guess about a particular model.
 *
 * <p>With nothing measured and nothing known about size, there is no honest signal left and the
 * estimate falls back to deliberately pessimistic constants. Pessimism is the safe direction: an
 * estimate that flatters a model wins latency-sensitive routing it has not earned, while one that
 * understates it merely loses ties to models that have actually been measured.
 */
final class PerformanceEstimator {

  /**
   * Bytes read per second when nothing on this machine has been measured.
   *
   * <p>About what a mid-range laptop sustains against DDR4. Used only as a last resort, and chosen
   * low so an unmeasured model does not outrank a measured one on speed it has never shown.
   */
  private static final double FALLBACK_BYTES_PER_SECOND = 8.0e9;

  /** Assumed weight size when a catalog reports none, taken as a 7B at 4-bit. */
  private static final long ASSUMED_SIZE_BYTES = 4L * 1024 * 1024 * 1024;

  /** Latency floor, and the estimate for a model of the assumed size. */
  private static final long FALLBACK_TIME_TO_FIRST_TOKEN_MILLIS = 800;

  private final double bytesPerSecond;
  private final double millisPerByte;
  private final boolean calibrated;

  private PerformanceEstimator(double bytesPerSecond, double millisPerByte, boolean calibrated) {
    this.bytesPerSecond = bytesPerSecond;
    this.millisPerByte = millisPerByte;
    this.calibrated = calibrated;
  }

  /**
   * Derives an estimator from whatever has been measured on this machine.
   *
   * @param measured models that carry both a performance profile and a size
   * @return an estimator calibrated to those models, or a pessimistic one when there are none
   */
  static PerformanceEstimator from(List<DiscoveredModel> measured) {
    List<Double> rates = new ArrayList<>();
    List<Double> latencies = new ArrayList<>();
    for (DiscoveredModel model : measured) {
      if (model.performance() == null || model.sizeBytes() <= 0) {
        continue;
      }
      rates.add(model.performance().tokensPerSecond() * model.sizeBytes());
      latencies.add(model.performance().timeToFirstTokenMillis() / (double) model.sizeBytes());
    }
    if (rates.isEmpty()) {
      return new PerformanceEstimator(
          FALLBACK_BYTES_PER_SECOND,
          FALLBACK_TIME_TO_FIRST_TOKEN_MILLIS / (double) ASSUMED_SIZE_BYTES,
          false);
    }
    // Median, not mean: one model benchmarked while the machine was busy would otherwise drag
    // every estimate with it.
    return new PerformanceEstimator(median(rates), median(latencies), true);
  }

  /** Whether this estimator was calibrated against something measured on this machine. */
  boolean calibrated() {
    return calibrated;
  }

  /**
   * Estimates what a model would do here.
   *
   * @param model the model, whose {@code performance()} is absent
   * @return an estimated profile, never null
   */
  DiscoveredModel.Performance estimate(DiscoveredModel model) {
    long size = model.sizeBytes() > 0 ? model.sizeBytes() : ASSUMED_SIZE_BYTES;
    double tokensPerSecond = bytesPerSecond / size;
    long timeToFirstToken = Math.max(1L, Math.round(millisPerByte * size));
    // Guard the arithmetic rather than trust it: a catalog reporting an absurd size would otherwise
    // produce a zero rate, which the Performance constructor rejects, turning a bad number into a
    // failed discovery.
    if (!(tokensPerSecond > 0) || !Double.isFinite(tokensPerSecond)) {
      tokensPerSecond = 1.0;
    }
    return new DiscoveredModel.Performance(timeToFirstToken, tokensPerSecond);
  }

  private static double median(List<Double> values) {
    Collections.sort(values);
    int middle = values.size() / 2;
    return values.size() % 2 == 1
        ? values.get(middle)
        : (values.get(middle - 1) + values.get(middle)) / 2.0;
  }
}
