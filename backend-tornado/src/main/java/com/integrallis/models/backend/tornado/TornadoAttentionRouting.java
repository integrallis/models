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
package com.integrallis.models.backend.tornado;

import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * What accelerated attention actually did, per layer-step rather than per request.
 *
 * <p>A kernel that is installed but never accepted a step and a kernel that accepted every step and
 * did not help produce the same end-to-end timings. Without this record the first would be read as
 * the second, and "GPU attention does not help" would be concluded from a measurement of something
 * that never ran. {@link #refusals()} with an empty {@link #deviceLayerSteps()} is the signature of
 * the first; {@link #refusalReasons()} says which gate closed.
 */
public record TornadoAttentionRouting(
    long deviceDecodeLayerSteps,
    long devicePrefillLayerChunks,
    long refusals,
    long mirroredPositions,
    long mirrorRebuilds,
    int planCount,
    double totalMillis,
    Map<String, Long> refusalReasons) {

  public TornadoAttentionRouting {
    refusalReasons = Map.copyOf(refusalReasons);
  }

  /** Layer-steps that ran on the device, decode and prefill together. */
  public long deviceLayerSteps() {
    return deviceDecodeLayerSteps + devicePrefillLayerChunks;
  }

  /** Whether any attention work reached the device at all. */
  public boolean ranOnDevice() {
    return deviceLayerSteps() > 0;
  }

  /** A single line fit for a log or a status report. */
  public String summary() {
    StringBuilder text = new StringBuilder();
    text.append("attention device-layer-steps=")
        .append(deviceLayerSteps())
        .append(" (decode=")
        .append(deviceDecodeLayerSteps)
        .append(", prefill=")
        .append(devicePrefillLayerChunks)
        .append("), refusals=")
        .append(refusals)
        .append(", mirrored-positions=")
        .append(mirroredPositions)
        .append(", mirror-rebuilds=")
        .append(mirrorRebuilds)
        .append(", plans=")
        .append(planCount);
    if (!refusalReasons.isEmpty()) {
      text.append(", refused:");
      refusalReasons.entrySet().stream()
          .sorted(Map.Entry.<String, Long>comparingByValue(Comparator.reverseOrder()))
          .forEach(
              entry ->
                  text.append(' ').append(entry.getKey()).append('=').append(entry.getValue()));
    }
    return text.toString();
  }

  /** The routing of a kernel that was never installed. */
  public static TornadoAttentionRouting none() {
    return new TornadoAttentionRouting(0, 0, 0, 0, 0, 0, 0.0, new LinkedHashMap<>());
  }
}
