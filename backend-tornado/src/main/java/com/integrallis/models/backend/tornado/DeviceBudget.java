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

import java.time.Duration;
import java.util.Locale;
import java.util.Objects;

/**
 * The device and host bytes one {@link DeviceMemoryRequest} costs under one {@link
 * PlanShapeStrategy}, itemised so an ineligibility message can name every term.
 *
 * @param strategy the plan shape this budget was computed for
 * @param deviceWeightBytes model weights retained on the device
 * @param planScratchBytes device activation, scale, and output buffers across all retained plans
 * @param kvCacheBytes KV cache bytes retained on the device
 * @param baseOverheadBytes flat runtime allowance for the TornadoVM context and compiled code
 * @param totalBytes sum of the device terms
 * @param hostWeightCopyBytes off-heap host copies the plans hold, on top of the mapped GGUF
 * @param estimatedPlanCompileTime eager-readiness estimate from the measured plan compile rate
 */
public record DeviceBudget(
    PlanShapeStrategy strategy,
    long deviceWeightBytes,
    long planScratchBytes,
    long kvCacheBytes,
    long baseOverheadBytes,
    long totalBytes,
    long hostWeightCopyBytes,
    Duration estimatedPlanCompileTime) {

  /**
   * Plans compiled per second during eager readiness.
   *
   * <p>Measured, not assumed: the Vultr A40-4Q gate on 2026-08-29 compiled 223 retained plans in
   * 13.554 s ({@code models-accelerator-bench/results/vultr-a40-4q-2026-08-29.md}). The A16-2Q gate
   * on the same artifact took 14.278 s for the same plan set, so this rate is the faster of the two
   * measured profiles and the readiness estimate it produces is optimistic.
   */
  static final double PLANS_COMPILED_PER_SECOND = 223.0 / 13.554;

  public DeviceBudget {
    Objects.requireNonNull(strategy, "strategy");
    Objects.requireNonNull(estimatedPlanCompileTime, "estimatedPlanCompileTime");
  }

  static DeviceBudget of(DeviceMemoryRequest request, PlanShapeStrategy strategy, long baseBytes) {
    Objects.requireNonNull(request, "request");
    Objects.requireNonNull(strategy, "strategy");
    long weights = strategy.deviceWeightBytes(request);
    long scratch = request.planScratchBytes();
    long kv = request.deviceKvCacheBytes();
    long total = Math.addExact(Math.addExact(weights, scratch), Math.addExact(kv, baseBytes));
    Duration readiness =
        Duration.ofMillis(
            Math.round(request.retainedPlanCount() / PLANS_COMPILED_PER_SECOND * 1000.0));
    return new DeviceBudget(
        strategy,
        weights,
        scratch,
        kv,
        baseBytes,
        total,
        strategy.hostWeightCopyBytes(request),
        readiness);
  }

  /** A one-line itemisation suitable for an operator-facing message. */
  public String describe() {
    StringBuilder text = new StringBuilder(128);
    text.append("weights ")
        .append(gib(deviceWeightBytes))
        .append(" (")
        .append(strategy)
        .append("), KV ")
        .append(gib(kvCacheBytes))
        .append(", plan scratch ")
        .append(gib(planScratchBytes))
        .append(", base ")
        .append(gib(baseOverheadBytes));
    return text.toString();
  }

  /** Formats a byte count as GiB with two decimals. */
  public static String gib(long bytes) {
    return String.format(Locale.ROOT, "%.2f GiB", bytes / (double) (1024L * 1024L * 1024L));
  }
}
