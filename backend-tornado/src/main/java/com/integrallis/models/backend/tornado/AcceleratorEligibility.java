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
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Device-vendor and capacity policy applied before constructing accelerator execution plans.
 *
 * <p>The gate adds up what a model would actually place on the device — retained weights under the
 * plan shape the kernel builds, per-plan scratch, and any device-resident KV cache — and compares
 * it against the device's global memory less a safety margin. Where a term is unknown it is refused
 * rather than assumed to be zero: a model whose weights exceed {@link
 * #COARSE_PROFILE_WEIGHT_LIMIT_BYTES} is not admitted on a file-size-only budget, because at that
 * size the unknown terms are larger than the margin.
 */
public final class AcceleratorEligibility {

  /**
   * Flat allowance for the TornadoVM device context, compiled code, and driver bookkeeping.
   *
   * <p>Unmeasured. It is a constant carried over from the published selector, kept so the admitted
   * budget for the qualified A16-2Q and A40-4Q profiles is unchanged. Calibrating it needs a GPU
   * host: {@code TornadoExecutionPlan.getCurrentDeviceMemoryUsage()} reports real usage after
   * readiness, and until that is run this term must not be described as measured.
   */
  static final long BASE_PLAN_OVERHEAD_BYTES = 256L * 1024L * 1024L;

  /** Maximum single allocation assumed for a coarse, file-size-only request. */
  static final long COARSE_MAX_SINGLE_ALLOCATION_BYTES = 512L * 1024L * 1024L;

  /**
   * Hard cap on one uploaded tensor.
   *
   * <p>{@code ByteArray} stores its element count in an {@code int} and {@code
   * ByteArray.fromSegment} computes it as {@code (int) segment.byteSize()}, so a tensor at or above
   * 2 GiB is truncated on construction and the following {@code MemorySegment.copy} of the full
   * tensor fails. Read from {@code tornado-api 5.2.0-jdk25}.
   */
  static final long MAX_TORNADO_ARRAY_BYTES = Integer.MAX_VALUE;

  /**
   * Weight ceiling for admitting a model on a coarse, file-size-only request.
   *
   * <p>Below this a model's KV cache, per-plan scratch, and largest tensor all fit comfortably
   * inside {@link #BASE_PLAN_OVERHEAD_BYTES}; above it they do not, and admitting the model would
   * mean gating on a number known to be incomplete.
   */
  static final long COARSE_PROFILE_WEIGHT_LIMIT_BYTES = 8L * 1024L * 1024L * 1024L;

  /**
   * Eager-readiness ceiling above which a configuration is not deployable.
   *
   * <p>Fixed in advance by gate G5 of the large-model GPU campaign pre-registration.
   */
  static final Duration READINESS_BUDGET = Duration.ofSeconds(120);

  /** Fraction of device global memory the gate refuses to plan into. */
  private static final long SAFETY_DIVISOR = 4L;

  private AcceleratorEligibility() {}

  /**
   * Selects a device using the coarse, file-size-only budget available before a model is parsed.
   *
   * @param devices devices reported by the TornadoVM runtime
   * @param modelSizeBytes size of the GGUF file on disk
   * @param accelerateDecode whether separate single-token decode plans are retained
   */
  public static Decision select(
      List<DeviceCapabilities> devices, long modelSizeBytes, boolean accelerateDecode) {
    Objects.requireNonNull(devices, "devices");
    if (modelSizeBytes <= 0) {
      return Decision.ineligible("model size must be positive", 0, null);
    }
    return select(
        devices, DeviceMemoryRequest.ofModelFile("model", modelSizeBytes, accelerateDecode));
  }

  /** Selects a device for a fully specified device-memory request. */
  public static Decision select(List<DeviceCapabilities> devices, DeviceMemoryRequest request) {
    Objects.requireNonNull(devices, "devices");
    Objects.requireNonNull(request, "request");

    DeviceBudget shipped = budget(request, PlanShapeStrategy.PER_SHAPE_WHOLE_MODEL);

    List<DeviceCapabilities> qualified =
        devices.stream()
            .filter(device -> "PTX".equals(device.backend()) || "CUDA".equals(device.backend()))
            .filter(device -> "GPU".equals(device.type()))
            .sorted(Comparator.comparingLong(DeviceCapabilities::globalMemoryBytes).reversed())
            .toList();
    if (qualified.isEmpty()) {
      return Decision.ineligible(
          "no qualified NVIDIA GPU backend was discovered; needed a PTX or CUDA backend on a GPU"
              + " device, found "
              + describe(devices),
          shipped.totalBytes(),
          shipped);
    }

    if (request.largestAllocationBytes() >= MAX_TORNADO_ARRAY_BYTES) {
      return Decision.ineligible(
          "largest device allocation is "
              + DeviceBudget.gib(request.largestAllocationBytes())
              + " but a TornadoVM ByteArray holds at most "
              + DeviceBudget.gib(MAX_TORNADO_ARRAY_BYTES)
              + " (int element count); the tensor must be split before it can be uploaded",
          shipped.totalBytes(),
          shipped);
    }

    if (!request.detailed() && request.weightBytes() > COARSE_PROFILE_WEIGHT_LIMIT_BYTES) {
      return Decision.ineligible(
          "model '"
              + request.modelLabel()
              + "' has "
              + DeviceBudget.gib(request.weightBytes())
              + " of weights, above the "
              + DeviceBudget.gib(COARSE_PROFILE_WEIGHT_LIMIT_BYTES)
              + " limit for a file-size-only device budget; KV residency, retained-plan count,"
              + " per-plan scratch and largest-tensor size are all unknown in this request."
              + " Build a DeviceMemoryRequest.detailed(...) from the model's tensor inventory"
              + " before admitting a model this size",
          shipped.totalBytes(),
          shipped);
    }

    DeviceCapabilities best = qualified.get(0);
    for (DeviceCapabilities device : qualified) {
      long safeCapacity = safeCapacity(device);
      if (shipped.totalBytes() <= safeCapacity && allocationFits(request, device)) {
        Duration readiness = shipped.estimatedPlanCompileTime();
        if (readiness.compareTo(READINESS_BUDGET) > 0) {
          return Decision.ineligible(
              "device "
                  + device.name()
                  + " fits the plan but the "
                  + request.retainedPlanCount()
                  + " retained plans would need about "
                  + seconds(readiness)
                  + " of eager readiness at the measured "
                  + String.format(Locale.ROOT, "%.1f", DeviceBudget.PLANS_COMPILED_PER_SECOND)
                  + " plans/s (A40-4Q, 223 plans in 13.554 s), above the "
                  + seconds(READINESS_BUDGET)
                  + " deployable ceiling; reduce the retained plan count before admitting this"
                  + " model",
              shipped.totalBytes(),
              shipped);
        }
        return new Decision(
            true,
            device,
            "eligible",
            shipped.totalBytes(),
            shipped,
            PlanShapeStrategy.PER_SHAPE_WHOLE_MODEL);
      }
    }

    for (DeviceCapabilities device : qualified) {
      if (!allocationFits(request, device)) {
        continue;
      }
      long safeCapacity = safeCapacity(device);
      for (PlanShapeStrategy strategy : PlanShapeStrategy.values()) {
        if (strategy.implemented() || !strategy.capacityComputable()) {
          continue;
        }
        DeviceBudget alternative = budget(request, strategy);
        if (alternative.totalBytes() <= safeCapacity) {
          return Decision.ineligible(
              "device "
                  + device.name()
                  + " has "
                  + DeviceBudget.gib(safeCapacity)
                  + " usable of "
                  + DeviceBudget.gib(device.globalMemoryBytes())
                  + " but the shipped plan shape needs "
                  + DeviceBudget.gib(shipped.totalBytes())
                  + " ("
                  + shipped.describe()
                  + "); the model would fit in "
                  + DeviceBudget.gib(alternative.totalBytes())
                  + " under "
                  + strategy
                  + ", which is not implemented: "
                  + strategy.limitation(),
              shipped.totalBytes(),
              shipped);
        }
      }
    }

    DeviceCapabilities target = best;
    long safeCapacity = safeCapacity(target);
    if (!allocationFits(request, target)) {
      return Decision.ineligible(
          "device "
              + target.name()
              + " allows a single allocation of at most "
              + DeviceBudget.gib(target.maxAllocationBytes())
              + " but the plan needs one of "
              + DeviceBudget.gib(largestAllocation(request)),
          shipped.totalBytes(),
          shipped);
    }
    return Decision.ineligible(
        "device "
            + target.name()
            + " has "
            + DeviceBudget.gib(safeCapacity)
            + " usable of "
            + DeviceBudget.gib(target.globalMemoryBytes())
            + " but the plan needs "
            + DeviceBudget.gib(shipped.totalBytes())
            + " ("
            + shipped.describe()
            + "), short by "
            + DeviceBudget.gib(shipped.totalBytes() - safeCapacity)
            + "; the same plans also hold "
            + DeviceBudget.gib(shipped.hostWeightCopyBytes())
            + " of host weight copies",
        shipped.totalBytes(),
        shipped);
  }

  /** Computes the device budget a request costs under one plan shape. */
  public static DeviceBudget budget(DeviceMemoryRequest request, PlanShapeStrategy strategy) {
    return DeviceBudget.of(request, strategy, BASE_PLAN_OVERHEAD_BYTES);
  }

  private static long safeCapacity(DeviceCapabilities device) {
    return device.globalMemoryBytes() - device.globalMemoryBytes() / SAFETY_DIVISOR;
  }

  private static long largestAllocation(DeviceMemoryRequest request) {
    return request.detailed()
        ? request.largestAllocationBytes()
        : Math.min(request.weightBytes(), COARSE_MAX_SINGLE_ALLOCATION_BYTES);
  }

  private static boolean allocationFits(DeviceMemoryRequest request, DeviceCapabilities device) {
    return largestAllocation(request) <= device.maxAllocationBytes();
  }

  private static String describe(List<DeviceCapabilities> devices) {
    if (devices.isEmpty()) {
      return "no devices";
    }
    StringBuilder text = new StringBuilder(64);
    for (DeviceCapabilities device : devices) {
      if (text.length() > 0) {
        text.append(", ");
      }
      text.append(device.name())
          .append(" [")
          .append(device.backend())
          .append('/')
          .append(device.type())
          .append(']');
    }
    return text.toString();
  }

  private static String seconds(Duration duration) {
    return String.format(Locale.ROOT, "%.0f s", duration.toMillis() / 1000.0);
  }

  /** One device as reported by the TornadoVM runtime. */
  public record DeviceCapabilities(
      String name, String backend, String type, long globalMemoryBytes, long maxAllocationBytes) {
    public DeviceCapabilities {
      Objects.requireNonNull(name, "name");
      Objects.requireNonNull(backend, "backend");
      Objects.requireNonNull(type, "type");
    }
  }

  /**
   * The outcome of device selection.
   *
   * @param eligible whether a device was admitted
   * @param device the admitted device, or {@code null}
   * @param reason {@code "eligible"}, or a message naming what was needed and what was available
   * @param requiredBytes device bytes the shipped plan shape would need
   * @param budget the itemised budget behind {@code requiredBytes}, or {@code null} when the
   *     request was rejected before one could be formed
   * @param strategy the plan shape the admitted device was admitted under, or {@code null}
   */
  public record Decision(
      boolean eligible,
      DeviceCapabilities device,
      String reason,
      long requiredBytes,
      DeviceBudget budget,
      PlanShapeStrategy strategy) {

    private static Decision ineligible(String reason, long requiredBytes, DeviceBudget budget) {
      return new Decision(false, null, reason, requiredBytes, budget, null);
    }
  }
}
