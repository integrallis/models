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

import com.integrallis.models.backend.tornado.AcceleratorEligibility;
import com.integrallis.models.backend.tornado.TornadoRuntimeDevices;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/** Prints the exact runtime facts used by automatic accelerator eligibility and fallback. */
public final class DeviceInventoryExperiment {
  private static final double MIB = 1024.0 * 1024.0;

  private DeviceInventoryExperiment() {}

  public static void main(String[] args) {
    if (args.length < 1 || args.length > 2 || (args.length == 2 && !"--decode".equals(args[1]))) {
      throw new IllegalArgumentException(
          "usage: DeviceInventoryExperiment <model-file> [--decode]");
    }
    Path model = Path.of(args[0]).toAbsolutePath().normalize();
    long modelBytes;
    try {
      modelBytes = Files.size(model);
    } catch (IOException exception) {
      throw new UncheckedIOException("could not read model size: " + model, exception);
    }
    boolean decode = Arrays.asList(args).contains("--decode");
    List<AcceleratorEligibility.DeviceCapabilities> devices = TornadoRuntimeDevices.discover();
    for (AcceleratorEligibility.DeviceCapabilities device : devices) {
      System.out.printf(
          Locale.ROOT,
          "device=%s backend=%s type=%s memory=%.1f MiB maxAllocation=%.1f MiB%n",
          device.name(),
          device.backend(),
          device.type(),
          device.globalMemoryBytes() / MIB,
          device.maxAllocationBytes() / MIB);
    }
    AcceleratorEligibility.Decision decision =
        AcceleratorEligibility.select(devices, modelBytes, decode);
    System.out.printf(
        Locale.ROOT,
        "eligible=%s selected=%s required=%.1f MiB reason=%s%n",
        decision.eligible(),
        decision.device() == null ? "none" : decision.device().name(),
        decision.requiredBytes() / MIB,
        decision.reason());
  }
}
