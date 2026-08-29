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
import java.util.List;
import java.util.Objects;

/** Device-vendor and capacity policy applied before constructing accelerator execution plans. */
public final class AcceleratorEligibility {
  private static final long PLAN_OVERHEAD_BYTES = 256L * 1024L * 1024L;
  private static final long MAX_ESTIMATED_SINGLE_ALLOCATION_BYTES = 512L * 1024L * 1024L;

  private AcceleratorEligibility() {}

  public static Decision select(
      List<DeviceCapabilities> devices, long modelSizeBytes, boolean accelerateDecode) {
    Objects.requireNonNull(devices, "devices");
    if (modelSizeBytes <= 0) {
      return Decision.ineligible("model size must be positive", 0);
    }
    long retainedCopies = accelerateDecode ? 2L : 1L;
    long requiredBytes =
        Math.addExact(Math.multiplyExact(modelSizeBytes, retainedCopies), PLAN_OVERHEAD_BYTES);
    long largestAllocation = Math.min(modelSizeBytes, MAX_ESTIMATED_SINGLE_ALLOCATION_BYTES);
    List<DeviceCapabilities> qualified =
        devices.stream()
            .filter(device -> "PTX".equals(device.backend()) || "CUDA".equals(device.backend()))
            .filter(device -> "GPU".equals(device.type()))
            .sorted(Comparator.comparingLong(DeviceCapabilities::globalMemoryBytes).reversed())
            .toList();
    if (qualified.isEmpty()) {
      return Decision.ineligible("no qualified NVIDIA GPU backend was discovered", requiredBytes);
    }
    for (DeviceCapabilities device : qualified) {
      long safeCapacity = device.globalMemoryBytes() - device.globalMemoryBytes() / 4L;
      if (requiredBytes <= safeCapacity && largestAllocation <= device.maxAllocationBytes()) {
        return new Decision(true, device, "eligible", requiredBytes);
      }
    }
    return Decision.ineligible(
        "qualified GPU has insufficient device memory or allocation capacity", requiredBytes);
  }

  public record DeviceCapabilities(
      String name, String backend, String type, long globalMemoryBytes, long maxAllocationBytes) {
    public DeviceCapabilities {
      Objects.requireNonNull(name, "name");
      Objects.requireNonNull(backend, "backend");
      Objects.requireNonNull(type, "type");
    }
  }

  public record Decision(
      boolean eligible, DeviceCapabilities device, String reason, long requiredBytes) {
    private static Decision ineligible(String reason, long requiredBytes) {
      return new Decision(false, null, reason, requiredBytes);
    }
  }
}
