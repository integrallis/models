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
package com.integrallis.models.bench;

/** One measured generation request, including failures instead of discarding them. */
public record TrialMeasurement(
    boolean successful,
    double ttftMillis,
    double totalMillis,
    double prefillTokensPerSecond,
    double decodeTokensPerSecond,
    int inputTokens,
    int outputTokens,
    long peakRssBytes,
    double cpuMillis,
    String outputSha256,
    String error) {

  public static TrialMeasurement success(
      double ttftMillis,
      double totalMillis,
      double prefillTokensPerSecond,
      int inputTokens,
      int outputTokens,
      long peakRssBytes,
      double cpuMillis,
      String outputSha256) {
    double postFirstTokenMillis = totalMillis - ttftMillis;
    double decodeTokensPerSecond =
        outputTokens > 1 && postFirstTokenMillis > 0
            ? (outputTokens - 1) * 1_000.0 / postFirstTokenMillis
            : 0;
    return new TrialMeasurement(
        true,
        ttftMillis,
        totalMillis,
        prefillTokensPerSecond,
        decodeTokensPerSecond,
        inputTokens,
        outputTokens,
        peakRssBytes,
        cpuMillis,
        outputSha256,
        null);
  }

  public static TrialMeasurement failure(String error) {
    return new TrialMeasurement(
        false, 0, 0, 0, 0, 0, 0, 0, 0, null, error == null ? "unknown failure" : error);
  }

  public double tpotMillis() {
    return decodeTokensPerSecond > 0 ? 1_000.0 / decodeTokensPerSecond : Double.POSITIVE_INFINITY;
  }
}
